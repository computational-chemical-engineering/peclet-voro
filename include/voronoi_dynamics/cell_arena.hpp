/**
 * @file cell_arena.hpp
 * @brief CSR (Compressed Sparse Row) CellArena and CellView for cache-efficient Voronoi cell
 *        storage.
 *
 * Replaces per-cell heap allocations with a single contiguous arena.
 * CellView is a lightweight non-owning handle (16 bytes) into the arena.
 */

#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

#include "vor_types.hpp"

namespace vor {

/**
 * @brief CSR arena owning all vertex and facet data for all Voronoi cells.
 *
 * Layout (CPU / CSR):
 *   vertexOffsets[c]   = start of cell c's vertex data
 *   facetOffsets[c]    = start of cell c's facet data
 *   vertexPos[3*(vOff+v)..3*(vOff+v)+2]  = vertex v of cell c (x,y,z)
 *   vertexTopo[vOff+v] = topology label array for vertex v of cell c
 *   facetLabel[fOff+f] = starting half-edge label for facet f of cell c
 *   facetNbr[fOff+f]   = neighbour cell id for facet f of cell c
 */
template <typename real_t>
struct CellArena {
  uint32_t numCells{0};

  // Offset arrays (length = numCells + 1)
  std::vector<uint32_t> vertexOffsets;  ///< vertexOffsets[c] = start index for cell c's vertices
  std::vector<uint32_t> facetOffsets;   ///< facetOffsets[c]  = start index for cell c's facets

  // Per-vertex arrays (total length = vertexOffsets[numCells])
  std::vector<real_t> vertexPos;    ///< x,y,z interleaved: element [3*(off+v)+k]
  std::vector<Vertex> vertexTopo;   ///< half-edge labels: element [off+v]

  // Per-facet arrays (total length = facetOffsets[numCells])
  std::vector<uint1> facetLabel;  ///< starting half-edge label for each facet
  std::vector<uint2> facetNbr;    ///< neighbour cell id for each facet

  // Per-cell scalars
  std::vector<uint8_t> numVertices;  ///< number of vertices per cell
  std::vector<uint8_t> numFacets;    ///< number of facets per cell
  std::vector<uint32_t> cellId;      ///< cell (particle) id

  /// Allocate storage given per-cell vertex/facet counts.
  void allocate(uint32_t nCells, const std::vector<uint8_t>& nV, const std::vector<uint8_t>& nF) {
    numCells = nCells;
    numVertices = nV;
    numFacets = nF;
    cellId.resize(nCells);

    vertexOffsets.resize(nCells + 1);
    facetOffsets.resize(nCells + 1);

    vertexOffsets[0] = 0;
    facetOffsets[0] = 0;
    for (uint32_t c = 0; c < nCells; ++c) {
      vertexOffsets[c + 1] = vertexOffsets[c] + nV[c];
      facetOffsets[c + 1] = facetOffsets[c] + nF[c];
    }

    uint32_t totalV = vertexOffsets[nCells];
    uint32_t totalF = facetOffsets[nCells];

    vertexPos.resize(3 * totalV);
    vertexTopo.resize(totalV);
    facetLabel.resize(totalF);
    facetNbr.resize(totalF);
  }

  /// Reset to empty state.
  void clear() {
    numCells = 0;
    vertexOffsets.clear();
    facetOffsets.clear();
    vertexPos.clear();
    vertexTopo.clear();
    facetLabel.clear();
    facetNbr.clear();
    numVertices.clear();
    numFacets.clear();
    cellId.clear();
  }
};

/**
 * @brief Lightweight non-owning view into a CellArena.
 *
 * CellView replaces Cell as the primary handle for Voronoi cell data.
 * It is trivially copyable (16 bytes: one pointer + one index).
 *
 * Provides the same interface as the old Cell class so existing code
 * can be updated with minimal changes.
 */
template <typename real_t>
class CellView {
 public:
  CellView() : m_arena(nullptr), m_cellIdx(0) {}
  CellView(CellArena<real_t>* arena, uint32_t cellIdx) : m_arena(arena), m_cellIdx(cellIdx) {}

  // ── Identity ────────────────────────────────────────────────────────────────

  inline uint2 getID() const { return static_cast<uint2>(m_arena->cellId[m_cellIdx]); }
  inline void setID(uint2 id) { m_arena->cellId[m_cellIdx] = id; }

  // ── Size ────────────────────────────────────────────────────────────────────

  inline uint0 numVertices() const { return m_arena->numVertices[m_cellIdx]; }
  inline uint0 numFacets() const { return m_arena->numFacets[m_cellIdx]; }

  // ── Vertex data ─────────────────────────────────────────────────────────────

  /// Pointer to position array for vertex v (3 consecutive reals: x,y,z).
  inline real_t* vertexPosPtr(uint0 v) { return &m_arena->vertexPos[3 * (vOff() + v)]; }
  inline const real_t* vertexPosPtr(uint0 v) const {
    return &m_arena->vertexPos[3 * (vOff() + v)];
  }

  /// Position of vertex v along coordinate axis k.
  inline real_t& vertexPos(uint0 v, uint0 k) {
    return m_arena->vertexPos[3 * (vOff() + v) + k];
  }
  inline real_t vertexPos(uint0 v, uint0 k) const {
    return m_arena->vertexPos[3 * (vOff() + v) + k];
  }

  /// Half-edge topology for vertex v (3-element Vertex array).
  inline Vertex& vertexTopo(uint0 v) { return m_arena->vertexTopo[vOff() + v]; }
  inline const Vertex& vertexTopo(uint0 v) const { return m_arena->vertexTopo[vOff() + v]; }

  // ── Facet data ───────────────────────────────────────────────────────────────

  inline uint1& facetLabel(uint0 f) { return m_arena->facetLabel[fOff() + f]; }
  inline uint1 facetLabel(uint0 f) const { return m_arena->facetLabel[fOff() + f]; }

  inline uint2& facetNbr(uint0 f) { return m_arena->facetNbr[fOff() + f]; }
  inline uint2 getNbr(uint0 f) const { return m_arena->facetNbr[fOff() + f]; }
  inline const uint2* getNbrs() const { return &m_arena->facetNbr[fOff()]; }

  // ── Topology helpers (replicate Cell methods) ───────────────────────────────

  inline bool hasNoNbr() const {
    for (uint0 i = 0; i < numFacets(); ++i)
      if (getNbr(i) == noNbr) return true;
    return false;
  }

  /// Get next label counter-clockwise on the boundary of a facet.
  inline uint1 getNextLabelCCW(uint1 label) const {
    uint1 facetMasked(label & maskFacet);
    uint1 revLabel(vertexTopo(getVertex(label))[getEdge(label)]);
    uint1 vertexMasked(revLabel & maskVertex);
    uint1 edge(getEdge(revLabel));
    (edge == 0 ? edge = 2 : --edge);
    return (facetMasked | vertexMasked | edge);
  }

  /// Get reverse label (the incoming edge for a half-edge).
  inline uint1 getReverseLabel(uint1 label) const {
    return vertexTopo(getVertex(label))[getEdge(label)];
  }

  // ── Raw pointer access ──────────────────────────────────────────────────────

  CellArena<real_t>* arena() { return m_arena; }
  const CellArena<real_t>* arena() const { return m_arena; }
  uint32_t cellIdx() const { return m_cellIdx; }

 private:
  inline uint32_t vOff() const { return m_arena->vertexOffsets[m_cellIdx]; }
  inline uint32_t fOff() const { return m_arena->facetOffsets[m_cellIdx]; }

  CellArena<real_t>* m_arena;
  uint32_t m_cellIdx;
};

}  // namespace vor
