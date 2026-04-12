/**
 * @file voronoi.hpp
 * @brief Voronoi cell construction, geometry, and dynamic cell-complex
 *        management for moving-particle simulations.
 *
 * Provides the core classes:
 *  - Cell          – topology and neighbor information for a single Voronoi cell
 *  - Cuboid        – helper that initialises a cell as a rectangular cuboid
 *  - CellMaker     – incremental half-plane cutting algorithm
 *  - CellGeometry  – geometric properties (volume, areas, velocity gradients)
 *  - CellComplex   – collection of cells and high-level build/update interface
 *  - NbrsToFacets  – neighbour-to-facet mapping for sparse-matrix assembly
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#ifdef VORONOI_USE_OPENMP
#include <omp.h>
#endif

#include "nbrlist.hpp"
#include "vor_types.hpp"

namespace vor {

/**
 * @brief Process-wide runtime counters for CellMaker topology pressure.
 */
struct CellMakerTelemetry {
  std::atomic<uint64_t> peak_vertices_seen;
  std::atomic<uint64_t> peak_facets_seen;
  std::atomic<uint64_t> peak_vertex_capacity;
  std::atomic<uint64_t> peak_facet_capacity;
  std::atomic<uint64_t> vertex_overflow_events;
  std::atomic<uint64_t> facet_overflow_events;
  std::atomic<uint64_t> vertex_growth_events;
  std::atomic<uint64_t> facet_growth_events;

  CellMakerTelemetry()
      : peak_vertices_seen(0)
      , peak_facets_seen(0)
      , peak_vertex_capacity(0)
      , peak_facet_capacity(0)
      , vertex_overflow_events(0)
      , facet_overflow_events(0)
      , vertex_growth_events(0)
      , facet_growth_events(0) {}
};

inline CellMakerTelemetry& cellMakerTelemetry() {
  static CellMakerTelemetry telemetry;
  return telemetry;
}

inline void resetCellMakerTelemetry() {
  auto& t = cellMakerTelemetry();
  t.peak_vertices_seen.store(0, std::memory_order_relaxed);
  t.peak_facets_seen.store(0, std::memory_order_relaxed);
  t.peak_vertex_capacity.store(0, std::memory_order_relaxed);
  t.peak_facet_capacity.store(0, std::memory_order_relaxed);
  t.vertex_overflow_events.store(0, std::memory_order_relaxed);
  t.facet_overflow_events.store(0, std::memory_order_relaxed);
  t.vertex_growth_events.store(0, std::memory_order_relaxed);
  t.facet_growth_events.store(0, std::memory_order_relaxed);
}

inline void updatePeakCounter(std::atomic<uint64_t>& counter, uint64_t candidate) {
  uint64_t current = counter.load(std::memory_order_relaxed);
  while (current < candidate &&
         !counter.compare_exchange_weak(current, candidate, std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
  }
}

/**
 * @brief make a 1 unsigned integer label that contains information of a neighbor vertex
 * @param facet index of facet
 * @param vertex index of neighbor vertex
 * @param edge index of the edge (0 to 2) of the neighbor vertex that points back to current vertex
 */
inline uint1 makeLabel(uint1 facet, uint1 vertex, uint1 edge) {
  return ((facet << shiftFacet) | (vertex << 2) | edge);
}

/**
 * @brief get the facet index from a label
 * @param label
 * \sa makeLabel
 */
inline uint1 getFacet(uint1 label) {
  return ((label & maskFacet) >> shiftFacet);
}

/**
 * @brief get the neighbor vertex index from a label
 * @param label
 * \sa makeLabel
 */
inline uint1 getVertex(uint1 label) {
  return ((label & maskVertex) >> 2);
}

/**
 * @brief get the neighbor edge index from a label
 * @param label
 * \sa makeLabel
 */
inline uint1 getEdge(uint1 label) {
  return (label & maskEdge);
}

template <typename real_t, bool Weighted = false>
class CellMaker;
template <typename real_t>
class CellGeometry;
template <typename real_t>
class ConstructionArena;
template <typename real_t>
struct CellView;
template <typename real_t>
struct GeometryView;
template <typename real_t>
class TopologyArena;
template <typename real_t>
struct ConnectivityView;
template <typename real_t>
class ConnectivityArena;
template <typename real_t>
class GeometryArena;

struct CellComplexUpdateStats {
  uint2 num_cells = 0;
  uint2 num_non_convex_before = 0;
  uint2 num_rebuild_candidates = 0;
  uint2 num_local_rebuild_cells = 0;
  uint2 num_empty_after_local_rebuild = 0;
  uint2 num_full_rebuild_cells = 0;
  uint2 num_repair_iterations = 0;
  uint2 num_repair_proposals_total = 0;
  uint2 num_repair_target_groups_total = 0;
  uint2 num_repair_cells_changed_total = 0;
  uint2 num_repair_direct_attempts = 0;
  uint2 num_repair_direct_successes = 0;
  uint2 num_repair_indirect_candidates = 0;
  uint2 num_repair_batch_calls = 0;
  uint2 num_repair_batch_changes = 0;
  bool rebuilt_from_scratch = false;
};

struct ParticleRenumberResult {
  std::vector<uint2> old_to_new;
  std::vector<uint2> new_to_old;
};

namespace detail {

template <typename real_t, bool Weighted>
class CellMakerWeightState {};

template <typename real_t>
class CellMakerWeightState<real_t, true> {
 protected:
  const std::vector<real_t>* m_weights = NULL;

  inline void setWeightStorage(const std::vector<real_t>* weights) { m_weights = weights; }
  inline real_t getWeight(uint2 id) const {
    return (m_weights != NULL && id < m_weights->size()) ? (*m_weights)[id] : real_t(0);
  }
};

template <typename real_t, bool Weighted>
class CellComplexWeightState {};

template <typename real_t>
class CellComplexWeightState<real_t, true> {
 protected:
  std::vector<real_t> m_weights;
  std::vector<uint8_t> m_particleHasCell;
  std::vector<uint2> m_cellParticleIds;
  bool m_weightsDirty = false;

  inline void syncWeights(size_t numParticles) {
    if (m_weights.size() < numParticles)
      m_weights.resize(numParticles, real_t(0));
    else if (m_weights.size() > numParticles)
      m_weights.resize(numParticles);
    if (m_particleHasCell.size() < numParticles)
      m_particleHasCell.resize(numParticles, 0u);
    else if (m_particleHasCell.size() > numParticles)
      m_particleHasCell.resize(numParticles);
  }
  inline void markWeightsDirty() { m_weightsDirty = true; }
  inline void clearWeightDirty() { m_weightsDirty = false; }
  inline bool weightsDirty() const { return m_weightsDirty; }
};

}  // namespace detail

/**
 * @class Cell
 * @brief class for storage of a single (Voronoi) cell
 * @tparam real_t real type used for floating point numbers (e.g. real or double)
 */
template <typename real_t>
class Cell {
 public:
  //! @brief constructor
  Cell() : m_id(0), m_numVertices(0), m_numFacets(0) {}
  //! @brief copy constructor
  Cell(const Cell<real_t>& cell);
  //! destructor.
  ~Cell() = default;
  //! @brief copy operator
  //! @param rhs of type Cell
  //! @return reference to the copied cell
  Cell& operator=(const Cell<real_t>& rhs);
  Cell& operator=(const CellView<real_t>& rhs);
  //! @brief copy operator
  //! @param rhs of type CellMaker (\sa CellMaker)
  //! @return reference to the copied cell
  template <bool Weighted>
  Cell& operator=(CellMaker<real_t, Weighted>& rhs);
  //! @brief set the id of a cell
  //! @param id the id to be set
  void setId(uint2 id) { m_id = id; }
  //! @brief reset the internal storage for the cell information
  //! @param numVertices number of vertices of the cell
  //! @param numFacets number of facets of the cell
  void reset(uint0 numVertices = 0, uint0 numFacets = 0);
  //! @brief print the topological information of a cell
  void printTopology() const;
  //! @brief print the facet information (such as cell id's of its neighbors)
  //! @param nbr id of neighbor cell. The facet corresponding to this neighbor is searched in the
  //! current cell
  void printFacet(uint2 nbr) const;
  //! @brief print all the information of the facets of the current cell
  //! @param cells vector of cells that should include the neighbors of the current cell
  void printNbrFacets(const std::vector<Cell<real_t> >& cells) const;
  //! @brief output the cell geometry in a Gnuplot format
  //! @param p coordinate of the center of the cell (Note that internal vertex coordinates are
  //! relative to the center.)
  void drawGnuplot(std::array<real_t, 3> p, FILE* fp) const;
  //! @brief output a facet in a Gnuplot format
  //! @param iFacet index of the facet to be drawn
  //! @param p coordinate of the center of the cell (Note that internal vertex coordinates are
  //! relative to the center.)
  inline void drawFacetGnuplot(uint1 iFacet, std::array<real_t, 3> p, FILE* fp) const;
  //! @brief facet information of a cell with all the verticies on it
  void printFacetInfo(std::array<real_t, 3> p, uint facet_id) const;
  //! @brief get the id if this cell
  //! @return ID of cell
  inline uint2 getID() const { return m_id; }
  //! @brief number of vertices the cell contains
  //! @return number of vertices
  inline uint0 numVertices() const { return m_numVertices; }
  //! @brief number of facets the cell contains
  //! @return number of facets
  inline uint0 numFacets() const { return m_numFacets; }
  //! @brief get the id of the neighbor cell corresponding to a facet index
  //! @param i index of a facet
  //! @return id of neighbor cell
  inline uint2 getNbr(uint1 i) const { return m_nbr[i]; }
  //! @brief get the array of id's of the neighbor cells
  //! @return array of id's of neighbor cells
  inline const uint2* getNbrs() const { return m_nbr; }
  //! @brief check of the cell has a facet that does not correspond to a neighbor cell
  //! Not every facet need necesarrily have an neighbor cell associated to it.
  //! @return true, if there are 1 or more facets without neighbors in a cell. false otherwise
  inline bool hasNoNbr();
  template <typename, bool>
  friend class CellMaker;
  friend class CellGeometry<real_t>;
  friend class TopologyArena<real_t>;

 protected:
  uint2 m_id;
  uint0 m_numFacets;
  uint0 m_numVertices;
  std::array<real_t, 3> m_vertexPos[maxNumVertices];
  Vertex m_vertices[maxNumVertices];
  uint1 m_facets[maxNumFacets];
  uint2 m_nbr[maxNumFacets];
  inline uint1 getNextLabelCCW(uint1 label) const {
    uint1 facetMasked(label & maskFacet);
    uint1 revLabel(m_vertices[getVertex(label)][getEdge(label)]);
    uint1 vertexMasked(revLabel & maskVertex);
    uint1 edge(getEdge(revLabel));
    (edge == 0 ? edge = 2 : --edge);
    return (facetMasked | vertexMasked | edge);
  }
  inline uint1 getReverseLabel(uint1 label) const {
    return m_vertices[getVertex(label)][getEdge(label)];
  }
};

/**
 * @class Cuboid
 * @brief class for creating a cuboid cell
 * This is typically used as a starting cell to carve out a Voronoi cell using plane cuts.
 * @tparam real_t real type used for floating point numbers (e.g. real or double)
 */
template <typename real_t>
class Cuboid : public Cell<real_t> {
 public:
  //! @brief constructor
  //! @param L contains the lengths of the 3 sides of the cuboid
  Cuboid(const std::array<real_t, 3>& L);
};

template <typename real_t>
class ConstructionArena {
 public:
  static constexpr uint1 kDefaultVertexCapacity = 128;
  static constexpr uint1 kDefaultFacetCapacity = 64;

  ConstructionArena(uint1 vertexCapacity = kDefaultVertexCapacity,
                    uint1 facetCapacity = kDefaultFacetCapacity) {
    ensureCapacity(vertexCapacity, facetCapacity);
    m_newVerticesWrk.reserve(20);
    m_facetPrevWrk.reserve(20);
    m_nbrsWrk.reserve(40);
    m_checkGridCell.reserve(64);
    m_vStackWrk.reserve(32);
  }

  void ensureCapacity(uint1 minVertices, uint1 minFacets) {
    size_t vertexCapacity = m_vertexPos.size();
    if (vertexCapacity < minVertices) {
      vertexCapacity = std::max<size_t>(std::max<size_t>(vertexCapacity * 2, minVertices), 1);
      m_vertexPos.resize(vertexCapacity);
      m_rSq.resize(vertexCapacity);
      m_vertices.resize(vertexCapacity);
      m_dist.resize(vertexCapacity);
      m_knownDistGen.resize(vertexCapacity, 0u);
      m_distGC.resize(vertexCapacity);
      m_renumVWrk.resize(vertexCapacity);
      m_aliveV.resize(vertexCapacity, 0u);
      m_freeStackV.resize(vertexCapacity);
    }

    size_t facetCapacity = m_facets.size();
    if (facetCapacity < minFacets) {
      facetCapacity = std::max<size_t>(std::max<size_t>(facetCapacity * 2, minFacets), 1);
      m_facets.resize(facetCapacity);
      m_nbr.resize(facetCapacity);
      m_renumFWrk.resize(facetCapacity);
      m_aliveF.resize(facetCapacity, 0u);
      m_freeStackF.resize(facetCapacity);
    }
  }

  uint1 vertexCapacity() const { return static_cast<uint1>(m_vertexPos.size()); }
  uint1 facetCapacity() const { return static_cast<uint1>(m_facets.size()); }

  void ensureVisitedSize(uint2 count) {
    if (m_visitedGen.size() < count)
      m_visitedGen.resize(count, 0u);
  }

  void startNewCell() {
    ++m_currentGeneration;
    if (m_currentGeneration == 0) {
      std::fill(m_visitedGen.begin(), m_visitedGen.end(), 0u);
      m_currentGeneration = 1;
    }
  }

  bool markAndCheckVisited(uint2 nbrId) {
    if (nbrId >= m_visitedGen.size())
      m_visitedGen.resize(static_cast<size_t>(nbrId) + 1, 0u);
    if (m_visitedGen[nbrId] == m_currentGeneration)
      return true;
    m_visitedGen[nbrId] = m_currentGeneration;
    return false;
  }

  std::array<real_t, 3>* vertexPosData() { return m_vertexPos.data(); }
  real_t* rSqData() { return m_rSq.data(); }
  Vertex* verticesData() { return m_vertices.data(); }
  uint1* facetsData() { return m_facets.data(); }
  uint2* nbrData() { return m_nbr.data(); }
  real_t* distData() { return m_dist.data(); }
  uint2* knownDistGenData() { return m_knownDistGen.data(); }
  real_t* distGCData() { return m_distGC.data(); }
  uint1* renumVWrkData() { return m_renumVWrk.data(); }
  uint1* renumFWrkData() { return m_renumFWrk.data(); }

  uint8_t* aliveVData() { return m_aliveV.data(); }
  uint8_t* aliveFData() { return m_aliveF.data(); }
  uint1* freeStackVData() { return m_freeStackV.data(); }
  uint1* freeStackFData() { return m_freeStackF.data(); }

  std::vector<uint8_t>& aliveV() { return m_aliveV; }
  std::vector<uint8_t>& aliveF() { return m_aliveF; }
  std::vector<uint1>& freeStackV() { return m_freeStackV; }
  std::vector<uint1>& freeStackF() { return m_freeStackF; }
  std::vector<uint2>& checkGridCell() { return m_checkGridCell; }
  std::vector<uint1>& newVerticesWrk() { return m_newVerticesWrk; }
  std::vector<uint1>& facetPrevWrk() { return m_facetPrevWrk; }
  std::vector<PosAndId<uint2, real_t> >& nbrsWrk() { return m_nbrsWrk; }
  std::vector<NbrDist<real_t> >& nbrDistWrk() { return m_nbrDistWrk; }
  std::vector<uint1>& vStackWrk() { return m_vStackWrk; }

 private:
  std::vector<std::array<real_t, 3> > m_vertexPos;
  std::vector<real_t> m_rSq;
  std::vector<Vertex> m_vertices;
  std::vector<uint1> m_facets;
  std::vector<uint2> m_nbr;
  std::vector<real_t> m_dist;
  std::vector<uint2> m_knownDistGen;
  std::vector<real_t> m_distGC;
  std::vector<uint1> m_renumVWrk;
  std::vector<uint1> m_renumFWrk;
  std::vector<uint8_t> m_aliveV;
  std::vector<uint8_t> m_aliveF;
  std::vector<uint1> m_freeStackV;
  std::vector<uint1> m_freeStackF;
  std::vector<uint2> m_checkGridCell;
  std::vector<uint1> m_newVerticesWrk;
  std::vector<uint1> m_facetPrevWrk;
  std::vector<PosAndId<uint2, real_t> > m_nbrsWrk;
  std::vector<NbrDist<real_t> > m_nbrDistWrk;
  std::vector<uint1> m_vStackWrk;
  uint2 m_currentGeneration = 0;
  std::vector<uint2> m_visitedGen;
};

/**
 * @class CellMaker
 * @brief class for making a cell using planar cuts
 * @tparam real_t real type used for floating point numbers (e.g. real or double)
 */
template <typename real_t, bool Weighted>
class CellMaker : private detail::CellMakerWeightState<real_t, Weighted> {
 public:
  //! @brief constructot
  explicit CellMaker(ConstructionArena<real_t>& arena);
  //! @brief destructor
  ~CellMaker();
  //! @brief initialize a cellmaker by equating it to a cell
  //! @param rhs cell used to initialize the cellmaker
  CellMaker& operator=(const Cell<real_t>& rhs);
  CellMaker& operator=(const CellView<real_t>& rhs);
  void setWeights(const std::vector<real_t>* weights) {
    if constexpr (Weighted)
      this->setWeightStorage(weights);
  }
  /**
   * @brief build a Voronoi cell
   *
   * The Voronoi cell of the particle with index id and coordinated pos[id] is build.
   * Typically the vector pos contains all positions of the particles in the system.
   * Therefore a cell list is used to efficiently pick the closest particles and stop when it is
   * guaranteed that there are no particles left that might contribute to the Voronoi cell. Also the
   * periodic simulation box is used to compute the smallest distance between a particle and the
   * center of the Voronoi cell Note that for a small number of particles more than 1 periodic image
   * of a particle can be a neighbor of a cell. Since this is not considered in the current version,
   * this means that the final Voronoi cell is not correct when the number of particles considered
   * is relatively small.
   * @param id the unique id of the cell to be build. It also refers to the particle index.
   * @param pos positions of all particles to be considered
   * @param nbrList neighbor list used to efficiently find the closest particles
   * @param initCell initial cell used to carve out the Voronoi cell
   * @return true if the final cell is different form initCell
   */
  bool build(uint2 id, const std::vector<std::array<real_t, 3> >& pos,
             const NbrList<uint2, real_t>& nbrList, const Cell<real_t>& initCell);
  /**
   * @brief build a Voronoi cell using the standard expanding neighbor search while skipping a set
   * of already-satisfied neighbors
   *
   * This is primarily used by local update paths that seed the builder from an existing convex
   * cell. The neighbor search is identical to the full builder; only the seed cell and the set of
   * skipped neighbors differ.
   *
   * @param id the unique id of the cell to be build. It also refers to the particle index.
   * @param pos positions of all particles to be considered
   * @param nbrList neighbor list used to efficiently find the closest particles
   * @param initCell initial cell used to carve out the Voronoi cell
   * @param skipNbrs sorted ids of neighbors that already define facets of initCell and should be
   * ignored during the build
   * @return true if the final cell is different form initCell
   */
  bool build(uint2 id, const std::vector<std::array<real_t, 3> >& pos,
             const NbrList<uint2, real_t>& nbrList, const Cell<real_t>& initCell,
             const std::vector<uint2>& skipNbrs);
  /**
   * @brief rebuild a Voronoi cell
   *
   * The Voronoi cell is carved out using only the neighbors as stored in initCell.
   * @param pos positions of all particles
   * @param box simulation box used to determine minimal image in the case of periodic boundary
   * conditions
   * @param initCell initial cell used to carve out the Voronoi cell
   * @return true if all neighbors of initCell are associated with a facet of the newly created cell
   */
  bool rebuild(const std::vector<std::array<real_t, 3> >& pos, const Box<real_t>& box,
               const Cell<real_t>& initCell);
  /**
   * @brief further refine the Voronoi cell by performing additional plane cuts corresponding to
   *neighbors The id's and positions of the neighbors are accessed by means of an iterator of type
   *PosAndId
   * @param begin starting value for the iterator
   * @param end end value of the iterator
   * @param pos0 position of the particle corresponding to the Voronoi cell under consideration
   * @return true if any of the neighbors caused a plane cut to add a facet to the cell
   **/
  inline bool processNbrs(typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin,
                          typename std::vector<PosAndId<uint2, real_t> >::const_iterator end,
                          const std::array<real_t, 3> pos0, const Box<real_t>& box);
  //    inline real_t getRsqMax() const {return m_rSq[m_vRsqMax];}
  void getCloseNbrs(NbrInsert& nbrs);
  // void drawGnuplot(FILE *fp) const;
  // void testTopo() const;
  inline uint1 numVertices() const { return m_numVertices; }
  inline uint1 numFacets() const { return m_numFacets; }
  const uint2* getNbrs() const { return m_nbr; }
  friend class Cell<real_t>;
  friend class TopologyArena<real_t>;
  friend class ConnectivityArena<real_t>;
  void renumber();

 protected:
  //! @brief initialize the cell to be cut
  //! @param cell used for the initialization
  void init(const Cell<real_t>& cell);
  void init(const CellView<real_t>& cell);
  inline bool cutCell(const std::array<real_t, 3> p, real_t rSqHalf, uint2 nbr);
  inline bool cutCell2(const std::array<real_t, 3> p, real_t rSqHalf, uint2 nbr);
  //! @brief compute squared distance between vertex i and the center of the cell
  //! Result is stored in private member m_rSq[i]
  //! @param i index of the vertex
  inline void computeRsq(uint1 i);
  //! @brief compute the squared distances of all cells
  void computeAllRsq();
  //! @brief find the vertex with the largest distance from the center
  //! The index of the vertex with the largest distance form the center will be stored in m_vRsqMax.
  void findRsqMax();
  //! @brief compute distance between the vertex i and a plane
  //! @param i index of the vertex
  //! @param p the poisition of a point relative to the cell center. The plane is the perpendicular
  //! plane halfway between the center and p.
  //! @param rSqHalf = 0.5|p|^2
  inline real_t computeDist(uint1 i, const std::array<real_t, 3> p, const real_t rSqHalf);
  //! @brief compute distance between the vertex i and a plane
  //! @param i index of the vertex
  //! @param p the poisition of a point relative to the cell center. The plane is the perpendicular
  //! plane halfway between the center and p.
  inline real_t computeDist(uint1 i, const std::array<real_t, 3> p);
  //! @brief compute distance between a plane and all vertices
  //! @param p the poisition of a point relative to the cell center. The plane is the perpendicular
  //! plane halfway between the center and p.
  //! @param rSqHalf = 0.5|p|^2
  void computeAllDist(const std::array<real_t, 3> p, const real_t rSqHalf);
  //! @brief reset internal variables that indicate distances of vertices to a plane are computed
  inline void resetDist();

  inline void computeGCOrig(uint2 indx, const std::array<real_t, 3>& pos);
  real_t computeRsqMinGC() const;
  std::array<real_t, 3> getClosestPointGC(uint1 indx) const;
  inline real_t computeDistGC(uint1 i);
  inline real_t computeMaxDistGC();
  real_t computeMaxDistGCVerb();
  void getAllDistGCVerb();
  void computeAllDistGC();
  uint1 m_numVertices, m_numFacets;  // only to be used after renumber()!
 private:
  CellMaker(const CellMaker<real_t, Weighted>& rhs);
  CellMaker& operator=(const CellMaker<real_t, Weighted>& rhs);
  inline uint1 getNextLabelCCW(uint1 label) const;
  inline uint1 getReverseLabel(uint1 label) const;
  inline bool applyCut(const std::array<real_t, 3>& p, real_t rSqHalf, uint2 nbr);
  inline real_t cutPlaneOffset(real_t rSqHalf, uint2 nbr) const;
  inline bool candidateMightCut(real_t planeOffset, real_t relNorm) const;
  inline void bindArenaStorage();
  inline void ensureVertexBuffers(uint1 minSize);
  inline void ensureFacetBuffers(uint1 minSize);
  inline uint1 allocVertexChecked(const char* caller);
  inline uint1 allocFacetChecked(const char* caller);
  [[noreturn]] inline void failCapacity(const char* caller, const char* kind, uint1 capacity) const;
  bool buildWithNeighborSearch(uint2 id, const std::vector<std::array<real_t, 3> >& pos,
                               const NbrList<uint2, real_t>& nbrList, const Cell<real_t>& initCell,
                               const std::vector<uint2>* skipNbrs);
  inline bool processNbrsFiltered(
      typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin,
      typename std::vector<PosAndId<uint2, real_t> >::const_iterator end,
      const std::array<real_t, 3> pos0, const Box<real_t>& box, const std::vector<uint2>* skipNbrs);
#ifndef VOR_CELLMAKER_USE_CUTCELL2
#define VOR_CELLMAKER_USE_CUTCELL2 1
#endif
  static constexpr bool kUseCutCell2 = (VOR_CELLMAKER_USE_CUTCELL2 != 0);
  static constexpr uint1 kMaxV = maxNumVertices;
  static constexpr uint1 kMaxF = maxNumFacets;
  static constexpr uint1 kInitialV = 128;
  static constexpr uint1 kInitialF = 64;
  uint2 m_id;
  const NbrList<uint2, real_t>* p_nbrList;
  ConstructionArena<real_t>* m_arena;
  std::array<real_t, 3>* m_vertexPos;
  real_t* m_rSq;
  Vertex* m_vertices;
  uint1* m_facets;
  uint2* m_nbr;
  DenseSlotsView<uint1> m_slotsV;
  DenseSlotsView<uint1> m_slotsF;
  VisitedIndx<uint2> m_visited;
  std::vector<uint2>& m_checkGridCell;  ///< BFS queue (use m_checkGCHead as front index)
  size_t m_checkGCHead;                 ///< index of current BFS queue front
  real_t m_distMax, m_distGCMax;
  uint1 m_vRsqMax;
  bool m_isAllCut;
  uint1 m_vDistMax, m_vDistGCMax;
  real_t* m_dist;
  uint2 m_distGen;        ///< generation counter for lazy distance reset
  uint2* m_knownDistGen;  ///< per-vertex generation stamp
  std::array<real_t, 3> m_relOrigGC, m_dLGC;
  real_t* m_distGC;
  uint1* m_renumVWrk;
  uint1* m_renumFWrk;
  std::vector<uint1>& m_newVerticesWrk;
  std::vector<uint1>& m_facetPrevWrk;
  std::vector<PosAndId<uint2, real_t> >& m_nbrsWrk;
  std::vector<NbrDist<real_t> >& m_nbrDistWrk;
  std::vector<uint1>& m_vStackWrk;  ///< DFS stack (push_back/back/pop_back)
};

template <typename real_t>
struct CellView {
  uint2 id;
  uint2 cellIndex;
  const TopologyArena<real_t>* arena;

  inline uint2 getID() const { return id; }
  inline uint1 numVertices() const { return arena->cellNumVertices(cellIndex); }
  inline uint1 numFacets() const { return arena->cellNumFacets(cellIndex); }
  inline const std::array<real_t, 3>& getVertexPos(uint1 i) const {
    return arena->cellVertexPos(cellIndex, i);
  }
  inline const Vertex& getVertex(uint1 i) const { return arena->cellVertexLabel(cellIndex, i); }
  inline uint1 getFacet(uint1 i) const { return arena->cellFacetLabel(cellIndex, i); }
  inline uint2 getNbr(uint1 i) const { return arena->cellNbr(cellIndex, i); }
  inline bool hasNoNbr() const {
    for (uint1 i = 0; i < numFacets(); ++i)
      if (getNbr(i) == noNbr)
        return true;
    return false;
  }
};

namespace detail {

struct CellTopologyRef {
  uint2 id;
  uint1 vertexCount;
  uint1 facetCount;
  const Vertex* vertices;
  const uint1* facets;
};

inline uint1 getReverseLabel(const CellTopologyRef& cell, uint1 label) {
  return cell.vertices[getVertex(label)][getEdge(label)];
}

inline void failInvalidTopology(const CellTopologyRef& cell, const char* stage, const char* what,
                                uint1 owner, uint1 edge, uint1 label) {
  std::fprintf(stderr,
               "Fatal: invalid cell topology in %s for cell %u: %s "
               "(owner=%u edge=%u facet=%u vertex=%u edgeLabel=%u, numVertices=%u, numFacets=%u)\n",
               stage, static_cast<unsigned>(cell.id), what, static_cast<unsigned>(owner),
               static_cast<unsigned>(edge), static_cast<unsigned>(getFacet(label)),
               static_cast<unsigned>(getVertex(label)), static_cast<unsigned>(getEdge(label)),
               static_cast<unsigned>(cell.vertexCount), static_cast<unsigned>(cell.facetCount));
  std::abort();
}

inline void validateCellTopology(const CellTopologyRef& cell, const char* stage) {
  if (cell.vertexCount > maxNumVertices || cell.facetCount > maxNumFacets) {
    std::fprintf(stderr,
                 "Fatal: invalid cell topology in %s for cell %u: counts exceed fixed storage "
                 "(numVertices=%u, numFacets=%u)\n",
                 stage, static_cast<unsigned>(cell.id), static_cast<unsigned>(cell.vertexCount),
                 static_cast<unsigned>(cell.facetCount));
    std::abort();
  }
  for (uint1 i = 0; i < cell.facetCount; ++i) {
    const uint1 label = cell.facets[i];
    if (getFacet(label) >= cell.facetCount)
      failInvalidTopology(cell, stage, "facet start label references facet out of range", i, 0,
                          label);
    if (getVertex(label) >= cell.vertexCount)
      failInvalidTopology(cell, stage, "facet start label references vertex out of range", i, 0,
                          label);
    if (getEdge(label) >= 3)
      failInvalidTopology(cell, stage, "facet start label references edge out of range", i, 0,
                          label);
  }
  for (uint1 i = 0; i < cell.vertexCount; ++i) {
    for (uint1 k = 0; k < 3; ++k) {
      const uint1 label = cell.vertices[i][k];
      if (getFacet(label) >= cell.facetCount)
        failInvalidTopology(cell, stage, "vertex label references facet out of range", i, k, label);
      if (getVertex(label) >= cell.vertexCount)
        failInvalidTopology(cell, stage, "vertex label references vertex out of range", i, k,
                            label);
      if (getEdge(label) >= 3)
        failInvalidTopology(cell, stage, "vertex label references edge out of range", i, k, label);
      const uint1 reverse = getReverseLabel(cell, label);
      if (getFacet(reverse) >= cell.facetCount)
        failInvalidTopology(cell, stage, "reverse label references facet out of range", i, k,
                            reverse);
      if (getVertex(reverse) != i)
        failInvalidTopology(cell, stage, "reverse label does not point back to owner vertex", i, k,
                            reverse);
      if (getEdge(reverse) != k)
        failInvalidTopology(cell, stage, "reverse label does not point back to owner edge", i, k,
                            reverse);
    }
  }
}

}  // namespace detail

template <typename real_t>
class CellGeometry {
 public:
  CellGeometry();
  CellGeometry(Cell<real_t>& cell);
  CellGeometry(const CellView<real_t>& cell);
  CellGeometry(const CellGeometry<real_t>& rhs);
  CellGeometry& operator=(Cell<real_t>& rhs);
  CellGeometry& operator=(const CellView<real_t>& rhs);
  CellGeometry& operator=(const CellGeometry<real_t>& rhs);
  void computeConnectingVectors(const std::vector<std::array<real_t, 3> >& pos,
                                const Box<real_t>& box);
  void computeEdgeInv();
  void updateVertexPos();
  void computeAreas();
  void computeVolume();
  void diffVolume();
  void computeAll();
  real_t maxConvexViolation() const;
  std::array<std::array<real_t, 3>, 3> velocityGradient(
      const std::vector<std::array<real_t, 3> >& velocity) const;
  void getDelaunayNbrs(uint1 iVertex, std::array<uint2, 3>& nbrs) const;
  void computeDelaunayForces(uint1 iVertex, const std::array<std::array<real_t, 3>, 3>& stress,
                             std::array<std::array<real_t, 3>, 3>& forces);
  std::array<std::array<real_t, 3>, 3> velocityGradientDelaunay(
      uint1 iVertex, const std::array<uint2, 3>& nbrs,
      const std::vector<std::array<real_t, 3> >& velocities) const;
  std::array<real_t, 3> force(
      const std::vector<std::array<std::array<real_t, 3>, 3> >& stresses) const;
  void gradFacetAreaSq(uint1 facetIndx, std::vector<uint2>& indx,
                       std::vector<std::array<real_t, 3> >& grad) const;
  inline const std::vector<std::array<real_t, 3> >& getdV() const { return m_dV; }
  inline const std::vector<std::array<real_t, 3> >& getAreas() const { return m_areas; }
  real_t getVolume() const { return m_vol; }
  const std::vector<real_t>& getVolumeDelaunay() const { return m_volDelaunay; }
  const std::vector<std::array<std::array<std::array<real_t, 3>, 3>, 3> >& getOmega() const {
    return m_omega;
  }
  bool isConvex() const;
  inline Cell<real_t>& getCell() { return *p_cell; }
  inline const Cell<real_t>& getCell() const { return *p_cell; }
  inline const std::vector<std::array<real_t, 3> >& getConnVect() const { return m_connV; }
  inline const std::vector<real_t>& getConnVectSq() const { return m_rSq; }

 protected:
  void resetDerived();
  Cell<real_t> m_ownedCell;
  Cell<real_t>* p_cell;
  std::vector<std::array<real_t, 3> > m_connV;
  std::vector<real_t> m_rSq;
  std::vector<std::array<std::array<real_t, 3>, 3> > m_edgeInv;
  std::vector<real_t> m_volDelaunay;
  std::vector<std::array<real_t, 3> > m_areas;
  real_t m_vol;
  // omega[i][j][l][k]
  // neighbor corresponding to facet i differentiated into j-direction
  // l: displacement direction, k: normal direction
  std::vector<std::array<std::array<std::array<real_t, 3>, 3>, 3> > m_omega;
  std::vector<std::array<real_t, 3> > m_dV;
};

/**
 * @class CellArena
 * @brief Packed storage for cell topology with per-cell offsets.
 *
 * This is an incremental migration target. Existing `Cell` storage remains the
 * source of truth; CellArena is rebuilt from cells after build/update to enable
 * memory-layout experiments and future API transition.
 */
template <typename real_t>
class TopologyArena {
 public:
  static constexpr uint1 PrimaryV = static_cast<uint1>(28);
  static constexpr uint1 PrimaryF = static_cast<uint1>(18);

  void clear() {
    m_ids.clear();
    m_vertexPos.clear();
    m_vertices.clear();
    m_facets.clear();
    m_nbrs.clear();
  }

  void rebuildFromCells(const std::vector<Cell<real_t> >& cells) {
    clear();
    const uint2 numCells = static_cast<uint2>(cells.size());
    m_ids.resize(numCells);
    m_vertexPos.resize(numCells);
    m_vertices.resize(numCells);
    m_facets.resize(numCells);
    m_nbrs.resize(numCells);

    for (size_t i = 0; i < cells.size(); ++i) {
      const Cell<real_t>& cell = cells[i];
      const uint1 nv = static_cast<uint1>(cell.m_numVertices);
      const uint1 nf = static_cast<uint1>(cell.m_numFacets);
      detail::CellTopologyRef ref;
      ref.id = cell.m_id;
      ref.vertexCount = nv;
      ref.facetCount = nf;
      ref.vertices = cell.m_vertices;
      ref.facets = cell.m_facets;
      detail::validateCellTopology(ref, "TopologyArena::rebuildFromCells");
      m_ids[i] = cell.m_id;
      m_vertexPos.insert(static_cast<uint2>(i), cell.m_vertexPos, nv);
      m_vertices.insert(static_cast<uint2>(i), cell.m_vertices, nv);
      m_facets.insert(static_cast<uint2>(i), cell.m_facets, nf);
      m_nbrs.insert(static_cast<uint2>(i), cell.m_nbr, nf);
    }
  }

  template <bool Weighted>
  void insertFromMaker(uint2 cellId, CellMaker<real_t, Weighted>& maker) {
    maker.renumber();
    m_ids[cellId] = maker.m_id;
    m_vertexPos.insert(cellId, maker.m_vertexPos, maker.m_numVertices);
    m_vertices.insert(cellId, maker.m_vertices, maker.m_numVertices);
    m_facets.insert(cellId, maker.m_facets, maker.m_numFacets);
    m_nbrs.insert(cellId, maker.m_nbr, maker.m_numFacets);
  }

  template <bool Weighted>
  void overwriteFromMaker(uint2 cellId, CellMaker<real_t, Weighted>& maker) {
    maker.renumber();
    m_ids[cellId] = maker.m_id;
    m_vertexPos.overwrite(cellId, maker.m_vertexPos, maker.m_numVertices);
    m_vertices.overwrite(cellId, maker.m_vertices, maker.m_numVertices);
    m_facets.overwrite(cellId, maker.m_facets, maker.m_numFacets);
    m_nbrs.overwrite(cellId, maker.m_nbr, maker.m_numFacets);
  }

  void overwriteFromCell(uint2 cellId, const Cell<real_t>& cell) {
    m_ids[cellId] = cell.m_id;
    m_vertexPos.overwrite(cellId, cell.m_vertexPos, static_cast<uint1>(cell.m_numVertices));
    m_vertices.overwrite(cellId, cell.m_vertices, static_cast<uint1>(cell.m_numVertices));
    m_facets.overwrite(cellId, cell.m_facets, static_cast<uint1>(cell.m_numFacets));
    m_nbrs.overwrite(cellId, cell.m_nbr, static_cast<uint1>(cell.m_numFacets));
  }

  void resize(uint2 numCells) {
    clear();
    m_ids.resize(numCells, 0u);
    m_vertexPos.resize(numCells);
    m_vertices.resize(numCells);
    m_facets.resize(numCells);
    m_nbrs.resize(numCells);
  }

  void prepare(uint2 numCells) {
    if (m_ids.size() != numCells)
      m_ids.resize(numCells);
    m_vertexPos.prepare(numCells);
    m_vertices.prepare(numCells);
    m_facets.prepare(numCells);
    m_nbrs.prepare(numCells);
  }

  size_t numCells() const { return m_ids.size(); }
  uint2 cellId(size_t i) const { return m_ids[i]; }
  uint1 cellNumVertices(size_t i) const { return m_vertexPos.count(static_cast<uint2>(i)); }
  uint1 cellNumFacets(size_t i) const { return m_facets.count(static_cast<uint2>(i)); }
  const std::array<real_t, 3>& cellVertexPos(size_t i, uint1 j) const {
    return m_vertexPos.get(static_cast<uint2>(i), j);
  }
  const Vertex& cellVertexLabel(size_t i, uint1 j) const {
    return m_vertices.get(static_cast<uint2>(i), j);
  }
  uint1 cellFacetLabel(size_t i, uint1 j) const { return m_facets.get(static_cast<uint2>(i), j); }
  uint2 cellNbr(size_t i, uint1 j) const { return m_nbrs.get(static_cast<uint2>(i), j); }
  const uint2* cellNbrData(size_t i) const {
    const uint2 cellId = static_cast<uint2>(i);
    if (cellNumFacets(i) <= PrimaryF)
      return m_nbrs.primaryData(cellId);
    thread_local std::vector<uint2> scratch;
    scratch.resize(cellNumFacets(i));
    for (uint1 j = 0; j < cellNumFacets(i); ++j)
      scratch[j] = cellNbr(i, j);
    return scratch.data();
  }

  CellView<real_t> getView(size_t i) const {
    CellView<real_t> view;
    view.id = m_ids[i];
    view.cellIndex = static_cast<uint2>(i);
    view.arena = this;
    return view;
  }

  void materializeCell(size_t i, Cell<real_t>& cell) const {
    cell.reset(cellNumVertices(i), cellNumFacets(i));
    cell.m_id = m_ids[i];
    for (uint1 j = 0; j < cell.m_numVertices; ++j) {
      cell.m_vertexPos[j] = cellVertexPos(i, j);
      cell.m_vertices[j] = cellVertexLabel(i, j);
    }
    for (uint1 j = 0; j < cell.m_numFacets; ++j) {
      cell.m_facets[j] = cellFacetLabel(i, j);
      cell.m_nbr[j] = cellNbr(i, j);
    }
  }

  const std::vector<uint2>& ids() const { return m_ids; }

  void swap(TopologyArena& other) {
    m_ids.swap(other.m_ids);
    m_vertexPos.swap(other.m_vertexPos);
    m_vertices.swap(other.m_vertices);
    m_facets.swap(other.m_facets);
    m_nbrs.swap(other.m_nbrs);
  }

 private:
  std::vector<uint2> m_ids;
  PrimaryOverflowArray<std::array<real_t, 3>, PrimaryV> m_vertexPos;
  PrimaryOverflowArray<Vertex, PrimaryV> m_vertices;
  PrimaryOverflowArray<uint1, PrimaryF> m_facets;
  PrimaryOverflowArray<uint2, PrimaryF> m_nbrs;
};

template <typename real_t>
using CellArena = TopologyArena<real_t>;

template <typename real_t>
struct ConnectivityView {
  uint2 cellIndex;
  const ConnectivityArena<real_t>* arena;

  inline uint1 numDirectNbrs() const { return arena->cellDirectCount(cellIndex); }
  inline uint1 numCandidates() const { return arena->cellCandidateCount(cellIndex); }
  inline uint2 getDirectNbr(uint1 i) const { return arena->cellCandidate(cellIndex, i); }
  inline uint2 getCandidate(uint1 i) const { return arena->cellCandidate(cellIndex, i); }
};

template <typename real_t>
class ConnectivityArena {
 public:
  static constexpr uint1 PrimaryF = TopologyArena<real_t>::PrimaryF;

  void clear() {
    m_directCounts.clear();
    m_candidates.clear();
  }

  void resize(uint2 numCells) {
    clear();
    m_directCounts.resize(numCells, 0u);
    m_candidates.resize(numCells);
  }

  void prepare(uint2 numCells) {
    if (m_directCounts.size() != numCells)
      m_directCounts.resize(numCells);
    std::fill(m_directCounts.begin(), m_directCounts.end(), 0u);
    m_candidates.prepare(numCells);
  }

  size_t numCells() const { return m_directCounts.size(); }

  void overwrite(uint2 cellId, const std::vector<uint2>& directNbrs,
                 const std::vector<uint2>& extraCandidates) {
    std::vector<uint2> merged;
    merged.reserve(directNbrs.size() + extraCandidates.size());
    for (size_t i = 0; i < directNbrs.size(); ++i)
      if (directNbrs[i] != noNbr)
        if (std::find(merged.begin(), merged.end(), directNbrs[i]) == merged.end())
          merged.push_back(directNbrs[i]);
    const uint1 directCount = static_cast<uint1>(merged.size());
    for (size_t i = 0; i < extraCandidates.size(); ++i)
      if (extraCandidates[i] != noNbr)
        if (std::find(merged.begin(), merged.end(), extraCandidates[i]) == merged.end())
          merged.push_back(extraCandidates[i]);
    m_directCounts[cellId] = directCount;
    m_candidates.overwrite(cellId, merged.data(), static_cast<uint1>(merged.size()));
  }

  template <bool Weighted>
  void overwriteFromMaker(uint2 cellId, CellMaker<real_t, Weighted>& maker) {
    std::vector<uint2> directNbrs;
    directNbrs.reserve(maker.m_numFacets);
    for (uint1 i = 0; i < maker.m_numFacets; ++i)
      if (maker.m_nbr[i] != noNbr)
        directNbrs.push_back(maker.m_nbr[i]);
    overwrite(cellId, directNbrs, std::vector<uint2>());
  }

  uint1 cellDirectCount(size_t i) const { return m_directCounts[i]; }
  uint1 cellCandidateCount(size_t i) const { return m_candidates.count(static_cast<uint2>(i)); }
  uint2 cellCandidate(size_t i, uint1 j) const {
    return m_candidates.get(static_cast<uint2>(i), j);
  }

  ConnectivityView<real_t> getView(size_t i) const {
    ConnectivityView<real_t> view;
    view.cellIndex = static_cast<uint2>(i);
    view.arena = this;
    return view;
  }

  void swap(ConnectivityArena& other) {
    m_directCounts.swap(other.m_directCounts);
    m_candidates.swap(other.m_candidates);
  }

 private:
  std::vector<uint1> m_directCounts;
  PrimaryOverflowArray<uint2, PrimaryF> m_candidates;
};

template <typename real_t>
struct GeometryView {
  uint2 id;
  uint2 cellIndex;
  const GeometryArena<real_t>* arena;

  inline real_t getVolume() const { return arena->cellVolume(cellIndex); }
  inline uint1 numFacets() const { return arena->cellFacetCount(cellIndex); }
  inline const std::array<real_t, 3>& getdV(uint1 i) const { return arena->cellDV(cellIndex, i); }
  inline const std::array<real_t, 3>& getArea(uint1 i) const {
    return arena->cellArea(cellIndex, i);
  }
  inline const std::array<real_t, 3>& getConnVect(uint1 i) const {
    return arena->cellConnVect(cellIndex, i);
  }
  inline real_t getConnVectSq(uint1 i) const { return arena->cellConnVectSq(cellIndex, i); }
};

template <typename real_t>
class GeometryArena {
 public:
  void clear() {
    m_ids.clear();
    m_volumes.clear();
    m_dV.clear();
    m_areas.clear();
    m_connV.clear();
    m_connVSq.clear();
  }

  void rebuildFromLegacy(const TopologyArena<real_t>& topology,
                         const std::vector<CellGeometry<real_t> >& geoms);

  void overwriteFromLegacy(uint2 cellId, uint2 id, const CellGeometry<real_t>& geom) {
    const uint1 facetCount = geom.getCell().numFacets();
    m_ids[cellId] = id;
    m_volumes[cellId] = geom.getVolume();
    const std::vector<std::array<real_t, 3> >& dV = geom.getdV();
    const std::vector<std::array<real_t, 3> >& areas = geom.getAreas();
    const std::vector<std::array<real_t, 3> >& connV = geom.getConnVect();
    const std::vector<real_t>& connVSq = geom.getConnVectSq();
    m_dV.overwrite(cellId, dV.data(), facetCount);
    m_areas.overwrite(cellId, areas.data(), facetCount);
    m_connV.overwrite(cellId, connV.data(), facetCount);
    m_connVSq.overwrite(cellId, connVSq.data(), facetCount);
  }

  void resize(uint2 numCells) {
    clear();
    m_ids.resize(numCells, 0u);
    m_volumes.resize(numCells, real_t());
    m_dV.resize(numCells);
    m_areas.resize(numCells);
    m_connV.resize(numCells);
    m_connVSq.resize(numCells);
  }

  void prepare(uint2 numCells) {
    if (m_ids.size() != numCells)
      m_ids.resize(numCells);
    if (m_volumes.size() != numCells)
      m_volumes.resize(numCells);
    m_dV.prepare(numCells);
    m_areas.prepare(numCells);
    m_connV.prepare(numCells);
    m_connVSq.prepare(numCells);
  }

  size_t numCells() const { return m_ids.size(); }
  real_t cellVolume(size_t i) const { return m_volumes[i]; }
  uint1 cellFacetCount(size_t i) const { return m_dV.count(static_cast<uint2>(i)); }
  const std::array<real_t, 3>& cellDV(size_t i, uint1 j) const {
    return m_dV.get(static_cast<uint2>(i), j);
  }
  const std::array<real_t, 3>& cellArea(size_t i, uint1 j) const {
    return m_areas.get(static_cast<uint2>(i), j);
  }
  const std::array<real_t, 3>& cellConnVect(size_t i, uint1 j) const {
    return m_connV.get(static_cast<uint2>(i), j);
  }
  real_t cellConnVectSq(size_t i, uint1 j) const { return m_connVSq.get(static_cast<uint2>(i), j); }

  GeometryView<real_t> getView(size_t i) const {
    GeometryView<real_t> view;
    view.id = m_ids[i];
    view.cellIndex = static_cast<uint2>(i);
    view.arena = this;
    return view;
  }

  const std::vector<uint2>& ids() const { return m_ids; }
  const std::vector<real_t>& volumes() const { return m_volumes; }

  void swap(GeometryArena& other) {
    m_ids.swap(other.m_ids);
    m_volumes.swap(other.m_volumes);
    m_dV.swap(other.m_dV);
    m_areas.swap(other.m_areas);
    m_connV.swap(other.m_connV);
    m_connVSq.swap(other.m_connVSq);
  }

 private:
  std::vector<uint2> m_ids;
  std::vector<real_t> m_volumes;
  PrimaryOverflowArray<std::array<real_t, 3>, TopologyArena<real_t>::PrimaryF> m_dV;
  PrimaryOverflowArray<std::array<real_t, 3>, TopologyArena<real_t>::PrimaryF> m_areas;
  PrimaryOverflowArray<std::array<real_t, 3>, TopologyArena<real_t>::PrimaryF> m_connV;
  PrimaryOverflowArray<real_t, TopologyArena<real_t>::PrimaryF> m_connVSq;
};

template <typename real_t, bool Weighted = false>
class CellComplex : private detail::CellComplexWeightState<real_t, Weighted> {
 public:
  CellComplex(Box<real_t>* box);
  CellComplex(Box<real_t>* box, size_t workerCount);
  ~CellComplex() = default;
  /// Build the packed topology/connectivity. Geometry is computed by default
  /// because update paths depend on it for convexity checks and local refresh,
  /// but it can be skipped and rebuilt lazily later.
  void build(const std::vector<std::array<real_t, 3> >& p, bool computeGeometry = true);
  void build(const std::vector<std::array<real_t, 3> >& p, const std::vector<uint8_t>& active,
             bool computeGeometry = true);
  /// Build geometry data (connecting vectors, edge inverses, volume derivatives)
  /// for all cells. Called automatically by build() when computeGeometry=true;
  /// call separately if needed.
  void buildGeometry(const std::vector<std::array<real_t, 3> >& p);
  /// Default neighbor-list-driven asynchronous sweep update. Cells are claimed
  /// at most once per timestep and repaired using local rebuilds with full
  /// rebuild fallback.
  void update(const std::vector<std::array<real_t, 3> >& p);
  void update(const std::vector<std::array<real_t, 3> >& p, const std::vector<uint8_t>& active);
  size_t numCells() const { return m_cellArena.numCells(); }
  CellView<real_t> getCellView(size_t i) const { return m_cellArena.getView(i); }
  ConnectivityView<real_t> getConnectivityView(size_t i) const { return m_connectivity.getView(i); }
  GeometryView<real_t> getGeometryView(size_t i) const { return m_geometry.getView(i); }
  void materializeCell(size_t i, Cell<real_t>& cell) const { m_cellArena.materializeCell(i, cell); }
  void materializeCells(std::vector<Cell<real_t> >& cells) const {
    cells.resize(m_cellArena.numCells());
    for (size_t i = 0; i < cells.size(); ++i)
      m_cellArena.materializeCell(i, cells[i]);
  }
  std::vector<uint0>& getTypes() { return m_types; }
  const std::vector<uint0>& getTypes() const { return m_types; }
  std::vector<CellGeometry<real_t> >& getGeoms() { return m_geom; }
  const std::vector<CellGeometry<real_t> >& getGeoms() const { return m_geom; }
  GeometryArena<real_t>& getGeometryArena() { return m_geometry; }
  const GeometryArena<real_t>& getGeometryArena() const { return m_geometry; }
  ConnectivityArena<real_t>& getConnectivityArena() { return m_connectivity; }
  const ConnectivityArena<real_t>& getConnectivityArena() const { return m_connectivity; }
  CellArena<real_t>& getCellArena() { return m_cellArena; }
  const CellArena<real_t>& getCellArena() const { return m_cellArena; }
  const NbrList<uint2, real_t>& getNbrList() const { return m_nbrList; }
  const CellComplexUpdateStats& getLastUpdateStats() const { return m_lastUpdateStats; }
  size_t numParticles() const { return m_particleActive.size(); }
  const std::vector<uint8_t>& getParticleActivity() const { return m_particleActive; }
  const std::vector<uint2>& getActiveParticleIds() const { return m_activeParticleIds; }
  const std::vector<uint2>& getCellParticleIds() const {
    if constexpr (Weighted)
      return this->m_cellParticleIds;
    return m_activeParticleIds;
  }
  const std::vector<real_t>& getWeights() const {
    if constexpr (Weighted)
      return this->m_weights;
    static const std::vector<real_t> kEmpty;
    return kEmpty;
  }
  void setWeights(const std::vector<real_t>& weights) {
    if constexpr (Weighted) {
      this->m_weights = weights;
      this->markWeightsDirty();
    } else {
      if (!weights.empty()) {
        std::fprintf(stderr, "CellComplex::setWeights only valid for power-cell specialization\n");
        std::abort();
      }
    }
  }
  bool isParticleActive(uint2 particleId) const {
    return particleId < m_particleActive.size() && m_particleActive[particleId] != 0u;
  }
  bool hasCell(uint2 particleId) const {
    if constexpr (Weighted)
      return particleId < this->m_particleHasCell.size() && this->m_particleHasCell[particleId] != 0u;
    return isParticleActive(particleId);
  }
  uint2 getCellIndexForParticle(uint2 particleId) const {
    return particleId < m_cellIndexByParticle.size() ? m_cellIndexByParticle[particleId] : noNbr;
  }
  void setParticleActivity(const std::vector<uint8_t>& active);
  void activateParticles(const std::vector<uint2>& particleIds);
  void deactivateParticles(const std::vector<uint2>& particleIds);
  void insertParticles(std::vector<std::array<real_t, 3> >& p,
                       const std::vector<std::array<real_t, 3> >& inserted);
  ParticleRenumberResult renumberParticles(std::vector<std::array<real_t, 3> >& p,
                                           bool rebuild = true);
  void drawInterfaceGnuplot(uint0 iType, uint0 jType, const std::vector<std::array<real_t, 3> >& p,
                            FILE* fp) const;

 private:
  class PersistentWorkerTeam {
   public:
    PersistentWorkerTeam() : m_stop(false), m_generation(0), m_completed(0), m_threadCount(0) {}

    ~PersistentWorkerTeam() { stop(); }

    void start(size_t count) {
      stop();
      m_stop = false;
      m_generation = 0;
      m_completed = 0;
      m_threadCount = count;
      for (size_t tid = 0; tid < m_threadCount; ++tid)
        m_threads.emplace_back(&PersistentWorkerTeam::workerLoop, this, tid);
    }

    void stop() {
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
      }
      m_cvStart.notify_all();
      for (size_t i = 0; i < m_threads.size(); ++i)
        if (m_threads[i].joinable())
          m_threads[i].join();
      m_threads.clear();
      m_threadCount = 0;
    }

    size_t threadCount() const { return m_threadCount; }

    template <typename Func>
    void run(Func fn) {
      if (m_threadCount == 0) {
        fn(0, 1);
        return;
      }

      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_task = fn;
        m_exception = nullptr;
        m_completed = 0;
        ++m_generation;
      }
      m_cvStart.notify_all();

      std::unique_lock<std::mutex> lock(m_mutex);
      m_cvDone.wait(lock, [this]() { return m_completed == m_threadCount; });
      std::exception_ptr ex = m_exception;
      lock.unlock();
      if (ex)
        std::rethrow_exception(ex);
    }

   private:
    void workerLoop(size_t tid) {
      size_t generationSeen = 0;
      while (true) {
        std::function<void(size_t, size_t)> task;
        {
          std::unique_lock<std::mutex> lock(m_mutex);
          m_cvStart.wait(
              lock, [this, &generationSeen]() { return m_stop || m_generation != generationSeen; });
          if (m_stop)
            return;
          generationSeen = m_generation;
          task = m_task;
        }

        try {
          task(tid, m_threadCount);
        } catch (...) {
          std::lock_guard<std::mutex> lock(m_mutex);
          if (!m_exception)
            m_exception = std::current_exception();
        }

        {
          std::lock_guard<std::mutex> lock(m_mutex);
          ++m_completed;
          if (m_completed == m_threadCount)
            m_cvDone.notify_one();
        }
      }
    }

    std::mutex m_mutex;
    std::condition_variable m_cvStart;
    std::condition_variable m_cvDone;
    std::vector<std::thread> m_threads;
    std::function<void(size_t, size_t)> m_task;
    std::exception_ptr m_exception;
    bool m_stop;
    size_t m_generation;
    size_t m_completed;
    size_t m_threadCount;
  };

  struct WorkerContext {
    WorkerContext() : arena(), maker(arena) {}

    // Per-thread scratch reused across build/update calls. The CellMaker owns no
    // storage; it is permanently bound to this worker's arena.
    ConstructionArena<real_t> arena;
    CellMaker<real_t, Weighted> maker;
  };

  static size_t defaultPersistentWorkerCount();
  template <typename Func>
  void parallelForPersistent(size_t count, Func fn);
  void ensureWorkerContexts(size_t count);
  void syncParticleActivity(size_t numParticles);
  void rebuildBuiltParticleMaps(size_t numParticles);
  bool allParticlesActive() const;
  static void normalizeActivity(std::vector<uint8_t>& active);
  static void collectActiveParticleIds(const std::vector<uint8_t>& active,
                                       std::vector<uint2>& particleIds);
  void buildFromCurrentActivity(const std::vector<std::array<real_t, 3> >& p, bool computeGeometry);
  void buildWeighted(const std::vector<std::array<real_t, 3> >& p,
                     const std::vector<uint8_t>* enabledMask, bool computeGeometry);
  void clearGeometryCache();
  void rebuildLegacyGeometryCache(const std::vector<std::array<real_t, 3> >& p);
  void commitCellGeometry(uint2 cellId, const TopologyArena<real_t>& cellArena,
                          std::vector<CellGeometry<real_t> >& geomCache,
                          GeometryArena<real_t>& geometryArena,
                          const std::vector<std::array<real_t, 3> >& p);
  void commitCellGeometry(uint2 cellId, const std::vector<std::array<real_t, 3> >& p);
  void initNbrList(const std::vector<std::array<real_t, 3> >& p);
  NbrList<uint2, real_t> m_nbrList;
  std::vector<uint0> m_types;
  TopologyArena<real_t> m_cellArena;
  ConnectivityArena<real_t> m_connectivity;
  GeometryArena<real_t> m_geometry;
  std::vector<CellGeometry<real_t> > m_geom;
  PersistentWorkerTeam m_team;
  std::vector<std::unique_ptr<WorkerContext> > m_workers;
  std::vector<uint8_t> m_particleActive;
  std::vector<uint2> m_activeParticleIds;
  std::vector<uint2> m_cellIndexByParticle;
  CellComplexUpdateStats m_lastUpdateStats;
  bool m_isBuild;
};

template <typename real_t = float>
using PowerCellComplex = CellComplex<real_t, true>;

template <typename real_t = float>
using PowerCellMaker = CellMaker<real_t, true>;

class NbrsToFacets {
 public:
  NbrsToFacets() {}
  template <typename real_t>
  void init(const std::vector<Cell<real_t> >& cells);
  template <typename real_t>
  void init(const CellArena<real_t>& arena);
  void print() const;
  //    void makeMatrixdVdV(const std::vector<real_t> & dV);
 protected:
  template <typename real_t>
  NbrsToFacets transposedV(const std::vector<real_t>& values, std::vector<real_t>& valuesTr) const;
  uint2 m_numCells;
  std::vector<uint2> m_ptr;
  std::vector<uint2> m_nbr;
  std::vector<uint1> m_facet;
};

template <typename real_t>
Cell<real_t>::Cell(const Cell<real_t>& rhs)
    : m_id(rhs.m_id), m_numVertices(rhs.m_numVertices), m_numFacets(rhs.m_numFacets) {
  std::memcpy(m_vertexPos, rhs.m_vertexPos, m_numVertices * sizeof(m_vertexPos[0]));
  std::memcpy(m_vertices, rhs.m_vertices, m_numVertices * sizeof(m_vertices[0]));
  std::memcpy(m_facets, rhs.m_facets, m_numFacets * sizeof(m_facets[0]));
  std::memcpy(m_nbr, rhs.m_nbr, m_numFacets * sizeof(m_nbr[0]));
}

template <typename real_t>
void Cell<real_t>::reset(uint0 numVertices, uint0 numFacets) {
  m_numVertices = numVertices;
  m_numFacets = numFacets;
}

template <typename real_t>
Cell<real_t>& Cell<real_t>::operator=(const Cell<real_t>& rhs) {
  if (&rhs == this)
    return *this;
  m_id = rhs.m_id;
  m_numVertices = rhs.m_numVertices;
  m_numFacets = rhs.m_numFacets;
  std::memcpy(m_vertexPos, rhs.m_vertexPos, m_numVertices * sizeof(m_vertexPos[0]));
  std::memcpy(m_vertices, rhs.m_vertices, m_numVertices * sizeof(m_vertices[0]));
  std::memcpy(m_facets, rhs.m_facets, m_numFacets * sizeof(m_facets[0]));
  std::memcpy(m_nbr, rhs.m_nbr, m_numFacets * sizeof(m_nbr[0]));
  return *this;
}

template <typename real_t>
Cell<real_t>& Cell<real_t>::operator=(const CellView<real_t>& rhs) {
  const uint1 vertexCount = rhs.numVertices();
  const uint1 facetCount = rhs.numFacets();
  if (vertexCount > maxNumVertices || facetCount > maxNumFacets) {
    std::fprintf(stderr, "Fatal: CellView exceeds fixed Cell storage (%u vertices, %u facets).\n",
                 static_cast<unsigned>(vertexCount), static_cast<unsigned>(facetCount));
    std::abort();
  }
  m_id = rhs.id;
  m_numVertices = vertexCount;
  m_numFacets = facetCount;
  for (uint1 i = 0; i < m_numVertices; ++i) {
    m_vertexPos[i] = rhs.getVertexPos(i);
    m_vertices[i] = rhs.getVertex(i);
  }
  for (uint1 i = 0; i < m_numFacets; ++i) {
    m_facets[i] = rhs.getFacet(i);
    m_nbr[i] = rhs.getNbr(i);
  }
  detail::CellTopologyRef ref;
  ref.id = m_id;
  ref.vertexCount = static_cast<uint1>(m_numVertices);
  ref.facetCount = static_cast<uint1>(m_numFacets);
  ref.vertices = m_vertices;
  ref.facets = m_facets;
  detail::validateCellTopology(ref, "Cell::operator=(CellView)");
  return *this;
}

template <typename real_t>
template <bool Weighted>
Cell<real_t>& Cell<real_t>::operator=(CellMaker<real_t, Weighted>& rhs) {
  m_id = rhs.m_id;
  rhs.renumber();
  if (rhs.m_numVertices > maxNumVertices || rhs.m_numFacets > maxNumFacets) {
    std::fprintf(stderr,
                 "Fatal: CellMaker output exceeds fixed Cell storage (%u vertices, %u facets).\n",
                 static_cast<unsigned>(rhs.m_numVertices), static_cast<unsigned>(rhs.m_numFacets));
    std::abort();
  }
  detail::CellTopologyRef ref;
  ref.id = rhs.m_id;
  ref.vertexCount = rhs.m_numVertices;
  ref.facetCount = rhs.m_numFacets;
  ref.vertices = rhs.m_vertices;
  ref.facets = rhs.m_facets;
  detail::validateCellTopology(ref, "Cell::operator=(CellMaker)");
  m_numVertices = rhs.m_numVertices;
  m_numFacets = rhs.m_numFacets;
  std::memcpy(m_vertexPos, rhs.m_vertexPos, m_numVertices * sizeof(m_vertexPos[0]));
  std::memcpy(m_vertices, rhs.m_vertices, m_numVertices * sizeof(m_vertices[0]));
  std::memcpy(m_facets, rhs.m_facets, m_numFacets * sizeof(m_facets[0]));
  std::memcpy(m_nbr, rhs.m_nbr, m_numFacets * sizeof(m_nbr[0]));
  return *this;
}

template <typename real_t>
void Cell<real_t>::printTopology() const {
  printf("cell: %u\n", m_id);
  for (uint1 i(0); i < m_numVertices; ++i) {
    printf("%u: %f ", i, m_vertexPos[i][0]);
    printf("%f ", m_vertexPos[i][1]);
    printf("%f\n", m_vertexPos[i][2]);
  }
  // for (uint1 i(0); i< m_numVertices; ++i){
  //   for (uint0 k(0); k< 3; ++k){
  //  	uint2 label = m_vertices[i][k];
  // 	printf("%u %u %u - ", getFacet(label), getVertex(label), getEdge(label));
  // 	uint2 labelNext(getNextLabelCCW(label));
  // 	printf("%u %u %u, ", getFacet(labelNext), getVertex(labelNext), getEdge(labelNext));
  //  	label = getReverseLabel(label);
  // 	printf("%u %u %u - ", getFacet(label), getVertex(label), getEdge(label));
  // 	labelNext = getNextLabelCCW(label);
  // 	printf("%u %u %u, ", getFacet(labelNext), getVertex(labelNext), getEdge(labelNext));
  //  	label = getReverseLabel(label);
  // 	printf("%u %u %u - ", getFacet(label), getVertex(label), getEdge(label));
  // 	labelNext = getNextLabelCCW(label);
  // 	printf("%u %u %u\n", getFacet(labelNext), getVertex(labelNext), getEdge(labelNext));
  //   }
  // }
  //    printf("number of vertices %u\n", m_numVertices);
  // printf("number of facets %u\n", m_numFacets);
  for (uint1 i(0); i < m_numFacets; ++i) {
    uint1 labelStart(m_facets[i]);
    printf("facet %u, neihgbor %u: ", i, m_nbr[i]);
    uint1 label = labelStart;
    uint1 labelNext = getNextLabelCCW(label);
    while (labelNext != labelStart) {
      printf("%d %d %d - ", getFacet(label), getVertex(label), getEdge(label));
      label = labelNext;
      labelNext = getNextLabelCCW(label);
    }
    printf("%d %d %d\n", getFacet(label), getVertex(label), getEdge(label));
  }
}

template <typename real_t>
void Cell<real_t>::printFacet(uint2 nbr) const {
  uint1 i(0);
  for (; i < m_numFacets; ++i)
    if (m_nbr[i] == nbr)
      break;
  if (i == m_numFacets) {
    printf("facet not found\n");
    return;
  }
  uint1 labelStart(m_facets[i]);
  printf("facet %u, neihgbor %u: ", i, m_nbr[i]);
  uint1 label = labelStart;
  uint1 labelNext = getNextLabelCCW(label);
  while (labelNext != labelStart) {
    printf("%u - ", m_nbr[getFacet(getReverseLabel(label))]);
    label = labelNext;
    labelNext = getNextLabelCCW(label);
  }
  printf("%u\n", m_nbr[getFacet(getReverseLabel(label))]);
}

template <typename real_t>
void Cell<real_t>::printNbrFacets(const std::vector<Cell<real_t> >& cells) const {
  printf("number of facets %u\n", m_numFacets);
  for (uint1 i(0); i < m_numFacets; ++i) {
    uint1 labelStart(m_facets[i]);
    printf("facet %u, neihgbor %u: ", i, m_nbr[i]);
    uint1 label = labelStart;
    uint1 labelNext = getNextLabelCCW(label);
    while (labelNext != labelStart) {
      printf("%u - ", m_nbr[getFacet(getReverseLabel(label))]);
      label = labelNext;
      labelNext = getNextLabelCCW(label);
    }
    printf("%u\n", m_nbr[getFacet(getReverseLabel(label))]);
    printf("neihgbor of facet %u: %u ", i, m_nbr[i]);
    cells[m_nbr[i]].printFacet(m_id);
  }
}

template <typename real_t>
void Cell<real_t>::drawGnuplot(std::array<real_t, 3> p, FILE* fp) const {
  for (uint1 i(0); i < m_numFacets; ++i) {
    drawFacetGnuplot(i, p, fp);
    fputs("\n\n", fp);
  }
}

template <typename real_t>
void Cell<real_t>::drawFacetGnuplot(uint1 iFacet, std::array<real_t, 3> p, FILE* fp) const {
  uint1 labelStart(m_facets[iFacet]);
  uint1 label = labelStart;
  uint1 vertex;
  vertex = getVertex(label);
  fprintf(fp, "%g ", p[0] + m_vertexPos[vertex][0]);
  fprintf(fp, "%g ", p[1] + m_vertexPos[vertex][1]);
  fprintf(fp, "%g\n", p[2] + m_vertexPos[vertex][2]);
  label = getNextLabelCCW(label);
  while (label != labelStart) {
    vertex = getVertex(label);
    fprintf(fp, "%g ", p[0] + m_vertexPos[vertex][0]);
    fprintf(fp, "%g ", p[1] + m_vertexPos[vertex][1]);
    fprintf(fp, "%g\n", p[2] + m_vertexPos[vertex][2]);
    label = getNextLabelCCW(label);
  }
  vertex = getVertex(label);
  fprintf(fp, "%g ", p[0] + m_vertexPos[vertex][0]);
  fprintf(fp, "%g ", p[1] + m_vertexPos[vertex][1]);
  fprintf(fp, "%g\n", p[2] + m_vertexPos[vertex][2]);
}

template <typename real_t>
void Cell<real_t>::printFacetInfo(std::array<real_t, 3> p, uint facet_id) const {
  uint1 labelStart, label, vertex, numfacevertex(1);
  labelStart = m_facets[facet_id];
  label = labelStart;
  vertex = getVertex(label);
  printf("%g ", p[0] + m_vertexPos[vertex][0]);
  printf("%g ", p[1] + m_vertexPos[vertex][1]);
  printf("%g\n", p[2] + m_vertexPos[vertex][2]);
  label = getNextLabelCCW(label);

  while (label != labelStart) {
    vertex = getVertex(label);
    printf("%g ", p[0] + m_vertexPos[vertex][0]);
    printf("%g ", p[1] + m_vertexPos[vertex][1]);
    printf("%g\n", p[2] + m_vertexPos[vertex][2]);
    label = getNextLabelCCW(label);
    numfacevertex++;
  }
  printf("Total verticies on this facet : %u\n\n", numfacevertex);
}

template <typename real_t>
bool Cell<real_t>::hasNoNbr() {
  bool has(false);
  for (uint0 i(0); i < m_numFacets; ++i)
    (m_nbr[i] == noNbr ? has = true : has);
  return has;
}

template <typename real_t>
Cuboid<real_t>::Cuboid(const std::array<real_t, 3>& L) {
  this->reset(8, 6);
  for (uint0 i(0); i < 6; ++i)
    this->m_nbr[i] = noNbr;

  this->m_vertexPos[0][0] = -0.5 * L[0];
  this->m_vertexPos[0][1] = -0.5 * L[1];
  this->m_vertexPos[0][2] = -0.5 * L[2];
  this->m_vertexPos[1][0] = 0.5 * L[0];
  this->m_vertexPos[1][1] = -0.5 * L[1];
  this->m_vertexPos[1][2] = -0.5 * L[2];
  this->m_vertexPos[2][0] = -0.5 * L[0];
  this->m_vertexPos[2][1] = 0.5 * L[1];
  this->m_vertexPos[2][2] = -0.5 * L[2];
  this->m_vertexPos[3][0] = 0.5 * L[0];
  this->m_vertexPos[3][1] = 0.5 * L[1];
  this->m_vertexPos[3][2] = -0.5 * L[2];
  this->m_vertexPos[4][0] = -0.5 * L[0];
  this->m_vertexPos[4][1] = -0.5 * L[1];
  this->m_vertexPos[4][2] = 0.5 * L[2];
  this->m_vertexPos[5][0] = 0.5 * L[0];
  this->m_vertexPos[5][1] = -0.5 * L[1];
  this->m_vertexPos[5][2] = 0.5 * L[2];
  this->m_vertexPos[6][0] = -0.5 * L[0];
  this->m_vertexPos[6][1] = 0.5 * L[1];
  this->m_vertexPos[6][2] = 0.5 * L[2];
  this->m_vertexPos[7][0] = 0.5 * L[0];
  this->m_vertexPos[7][1] = 0.5 * L[1];
  this->m_vertexPos[7][2] = 0.5 * L[2];

  // vertex and edge point to the next label (counter clock wise) on the boundary of a facet
  this->m_facets[0] = makeLabel(0, 0, 0);
  this->m_vertices[0][0] = makeLabel(2, 1, 2);
  this->m_vertices[1][1] = makeLabel(4, 5, 2);
  this->m_vertices[5][1] = makeLabel(5, 4, 0);
  this->m_vertices[4][2] = makeLabel(1, 0, 1);
  this->m_facets[1] = makeLabel(1, 0, 1);
  this->m_vertices[0][1] = makeLabel(0, 4, 2);
  this->m_vertices[4][1] = makeLabel(5, 6, 0);
  this->m_vertices[6][2] = makeLabel(3, 2, 1);
  this->m_vertices[2][0] = makeLabel(2, 0, 2);
  this->m_facets[2] = makeLabel(2, 0, 2);
  this->m_vertices[0][2] = makeLabel(1, 2, 0);
  this->m_vertices[2][2] = makeLabel(3, 3, 0);
  this->m_vertices[3][2] = makeLabel(4, 1, 0);
  this->m_vertices[1][2] = makeLabel(0, 0, 0);
  this->m_facets[3] = makeLabel(3, 3, 0);
  this->m_vertices[3][0] = makeLabel(2, 2, 2);
  this->m_vertices[2][1] = makeLabel(1, 6, 2);
  this->m_vertices[6][1] = makeLabel(5, 7, 0);
  this->m_vertices[7][2] = makeLabel(4, 3, 1);
  this->m_facets[4] = makeLabel(4, 1, 0);
  this->m_vertices[1][0] = makeLabel(2, 3, 2);
  this->m_vertices[3][1] = makeLabel(3, 7, 2);
  this->m_vertices[7][1] = makeLabel(5, 5, 0);
  this->m_vertices[5][2] = makeLabel(0, 1, 1);
  this->m_facets[5] = makeLabel(5, 4, 0);
  this->m_vertices[4][0] = makeLabel(0, 5, 1);
  this->m_vertices[5][0] = makeLabel(4, 7, 1);
  this->m_vertices[7][0] = makeLabel(3, 6, 1);
  this->m_vertices[6][0] = makeLabel(1, 4, 1);
}

template <typename real_t, bool Weighted>
CellMaker<real_t, Weighted>::CellMaker(ConstructionArena<real_t>& arena)
    : m_numVertices(0)
    , m_numFacets(0)
    , m_id(0)
    , p_nbrList(NULL)
    , m_arena(&arena)
    , m_vertexPos(NULL)
    , m_rSq(NULL)
    , m_vertices(NULL)
    , m_facets(NULL)
    , m_nbr(NULL)
    , m_checkGridCell(arena.checkGridCell())
    , m_checkGCHead(0u)
    , m_isAllCut(false)
    , m_dist(NULL)
    , m_distGen(0)
    , m_knownDistGen(NULL)
    , m_distGC(NULL)
    , m_renumVWrk(NULL)
    , m_renumFWrk(NULL)
    , m_newVerticesWrk(arena.newVerticesWrk())
    , m_facetPrevWrk(arena.facetPrevWrk())
    , m_nbrsWrk(arena.nbrsWrk())
    , m_nbrDistWrk(arena.nbrDistWrk())
    , m_vStackWrk(arena.vStackWrk()) {
  bindArenaStorage();
  updatePeakCounter(cellMakerTelemetry().peak_vertex_capacity,
                    static_cast<uint64_t>(m_arena->vertexCapacity()));
  updatePeakCounter(cellMakerTelemetry().peak_facet_capacity,
                    static_cast<uint64_t>(m_arena->facetCapacity()));
  std::fill(m_knownDistGen, m_knownDistGen + m_arena->vertexCapacity(), 0u);
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::bindArenaStorage() {
  m_vertexPos = m_arena->vertexPosData();
  m_rSq = m_arena->rSqData();
  m_vertices = m_arena->verticesData();
  m_facets = m_arena->facetsData();
  m_nbr = m_arena->nbrData();
  m_dist = m_arena->distData();
  m_knownDistGen = m_arena->knownDistGenData();
  m_distGC = m_arena->distGCData();
  m_renumVWrk = m_arena->renumVWrkData();
  m_renumFWrk = m_arena->renumFWrkData();
  m_slotsV.setStorage(m_arena->aliveVData(), m_arena->freeStackVData(), m_arena->vertexCapacity());
  m_slotsF.setStorage(m_arena->aliveFData(), m_arena->freeStackFData(), m_arena->facetCapacity());
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::ensureVertexBuffers(uint1 minSize) {
  m_arena->ensureCapacity(minSize, m_arena->facetCapacity());
  bindArenaStorage();
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::ensureFacetBuffers(uint1 minSize) {
  m_arena->ensureCapacity(m_arena->vertexCapacity(), minSize);
  bindArenaStorage();
}

template <typename real_t, bool Weighted>
CellMaker<real_t, Weighted>::~CellMaker() {}

template <typename real_t, bool Weighted>
CellMaker<real_t, Weighted>& CellMaker<real_t, Weighted>::operator=(const Cell<real_t>& rhs) {
  m_id = rhs.m_id;
  init(rhs);
  return *this;
}

template <typename real_t, bool Weighted>
CellMaker<real_t, Weighted>& CellMaker<real_t, Weighted>::operator=(const CellView<real_t>& rhs) {
  m_id = rhs.id;
  init(rhs);
  return *this;
}

template <typename real_t, bool Weighted>
uint1 CellMaker<real_t, Weighted>::getNextLabelCCW(uint1 label) const {
  uint1 facetMasked(label & maskFacet);
  uint1 revLabel(m_vertices[getVertex(label)][getEdge(label)]);
  uint1 vertexMasked(revLabel & maskVertex);
  uint1 edge(getEdge(revLabel));
  (edge == 0 ? edge = 2 : --edge);
  return (facetMasked | vertexMasked | edge);
}

template <typename real_t, bool Weighted>
uint1 CellMaker<real_t, Weighted>::getReverseLabel(uint1 label) const {
  return m_vertices[getVertex(label)][getEdge(label)];
}

template <typename real_t, bool Weighted>
real_t CellMaker<real_t, Weighted>::cutPlaneOffset(real_t rSqHalf, uint2 nbr) const {
  if constexpr (Weighted)
    return rSqHalf + real_t(0.5) * (this->getWeight(m_id) - this->getWeight(nbr));
  return rSqHalf;
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::candidateMightCut(real_t planeOffset, real_t relNorm) const {
  if constexpr (!Weighted) {
    (void)relNorm;
    return planeOffset < real_t(2) * m_rSq[m_vRsqMax];
  } else {
    (void)planeOffset;
    (void)relNorm;
    return true;
  }
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::applyCut(const std::array<real_t, 3>& p, real_t rSqHalf, uint2 nbr) {
  const real_t planeOffset = cutPlaneOffset(rSqHalf, nbr);
  if (kUseCutCell2)
    return cutCell2(p, planeOffset, nbr);
  return cutCell(p, planeOffset, nbr);
}

template <typename real_t, bool Weighted>
uint1 CellMaker<real_t, Weighted>::allocVertexChecked(const char* caller) {
  uint1 v_new = m_slotsV.getFree();
  if (v_new == DenseSlotsView<uint1>::InvalidIdx) {
    if (m_arena->vertexCapacity() >= kMaxV)
      failCapacity(caller, "vertex", kMaxV);
    size_t newCapacity = std::max<size_t>(1, m_arena->vertexCapacity() * 2);
    newCapacity = std::min<size_t>(newCapacity, static_cast<size_t>(kMaxV));
    if (newCapacity <= m_arena->vertexCapacity())
      failCapacity(caller, "vertex", kMaxV);
    m_arena->ensureCapacity(static_cast<uint1>(newCapacity), m_arena->facetCapacity());
    bindArenaStorage();
    cellMakerTelemetry().vertex_growth_events.fetch_add(1, std::memory_order_relaxed);
    updatePeakCounter(cellMakerTelemetry().peak_vertex_capacity,
                      static_cast<uint64_t>(m_arena->vertexCapacity()));
    v_new = m_slotsV.getFree();
  }
  if (v_new == DenseSlotsView<uint1>::InvalidIdx)
    failCapacity(caller, "vertex", m_arena->vertexCapacity());
  return v_new;
}

template <typename real_t, bool Weighted>
uint1 CellMaker<real_t, Weighted>::allocFacetChecked(const char* caller) {
  uint1 f_new = m_slotsF.getFree();
  if (f_new == DenseSlotsView<uint1>::InvalidIdx) {
    if (m_arena->facetCapacity() >= kMaxF)
      failCapacity(caller, "facet", kMaxF);
    size_t newCapacity = std::max<size_t>(1, m_arena->facetCapacity() * 2);
    newCapacity = std::min<size_t>(newCapacity, static_cast<size_t>(kMaxF));
    if (newCapacity <= m_arena->facetCapacity())
      failCapacity(caller, "facet", kMaxF);
    m_arena->ensureCapacity(m_arena->vertexCapacity(), static_cast<uint1>(newCapacity));
    bindArenaStorage();
    cellMakerTelemetry().facet_growth_events.fetch_add(1, std::memory_order_relaxed);
    updatePeakCounter(cellMakerTelemetry().peak_facet_capacity,
                      static_cast<uint64_t>(m_arena->facetCapacity()));
    f_new = m_slotsF.getFree();
  }
  if (f_new == DenseSlotsView<uint1>::InvalidIdx)
    failCapacity(caller, "facet", m_arena->facetCapacity());
  return f_new;
}

template <typename real_t, bool Weighted>
[[noreturn]] void CellMaker<real_t, Weighted>::failCapacity(const char* caller, const char* kind,
                                                  uint1 capacity) const {
  if (kind[0] == 'v') {
    cellMakerTelemetry().vertex_overflow_events.fetch_add(1, std::memory_order_relaxed);
  } else {
    cellMakerTelemetry().facet_overflow_events.fetch_add(1, std::memory_order_relaxed);
  }
  std::fprintf(stderr,
               "Fatal: CellMaker %s capacity exceeded while processing cell %u in %s. "
               "Configured %s capacity is %u.\n",
               kind, static_cast<unsigned>(m_id), caller, kind, static_cast<unsigned>(capacity));
  std::abort();
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::init(const Cell<real_t>& cell) {
  if (m_arena->vertexCapacity() < cell.m_numVertices) {
    m_arena->ensureCapacity(cell.m_numVertices, m_arena->facetCapacity());
    bindArenaStorage();
  }
  m_slotsV.reset(cell.m_numVertices);
  std::fill(m_arena->aliveV().begin(), m_arena->aliveV().end(), 0u);
  std::fill(m_arena->aliveV().begin(), m_arena->aliveV().begin() + cell.m_numVertices, uint8_t(1));
  std::memcpy(m_vertexPos, cell.m_vertexPos, cell.m_numVertices * sizeof(m_vertexPos[0]));
  std::memcpy(m_vertices, cell.m_vertices, cell.m_numVertices * sizeof(m_vertices[0]));
  if (m_arena->facetCapacity() < cell.m_numFacets) {
    m_arena->ensureCapacity(m_arena->vertexCapacity(), cell.m_numFacets);
    bindArenaStorage();
  }
  m_slotsF.reset(cell.m_numFacets);
  std::fill(m_arena->aliveF().begin(), m_arena->aliveF().end(), 0u);
  std::fill(m_arena->aliveF().begin(), m_arena->aliveF().begin() + cell.m_numFacets, uint8_t(1));
  std::memcpy(m_facets, cell.m_facets, cell.m_numFacets * sizeof(m_facets[0]));
  std::memcpy(m_nbr, cell.m_nbr, cell.m_numFacets * sizeof(m_nbr[0]));
  computeAllRsq();
  resetDist();
  const std::numeric_limits<real_t> lim;
  m_distGCMax = -lim.max();
  m_vDistGCMax = maxNumVertices;
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::init(const CellView<real_t>& cell) {
  const uint1 vertexCount = cell.numVertices();
  const uint1 facetCount = cell.numFacets();
  if (m_arena->vertexCapacity() < vertexCount) {
    m_arena->ensureCapacity(vertexCount, m_arena->facetCapacity());
    bindArenaStorage();
  }
  m_slotsV.reset(vertexCount);
  std::fill(m_arena->aliveV().begin(), m_arena->aliveV().end(), 0u);
  std::fill(m_arena->aliveV().begin(), m_arena->aliveV().begin() + vertexCount, uint8_t(1));
  for (uint1 i = 0; i < vertexCount; ++i) {
    m_vertexPos[i] = cell.getVertexPos(i);
    m_vertices[i] = cell.getVertex(i);
  }
  if (m_arena->facetCapacity() < facetCount) {
    m_arena->ensureCapacity(m_arena->vertexCapacity(), facetCount);
    bindArenaStorage();
  }
  m_slotsF.reset(facetCount);
  std::fill(m_arena->aliveF().begin(), m_arena->aliveF().end(), 0u);
  std::fill(m_arena->aliveF().begin(), m_arena->aliveF().begin() + facetCount, uint8_t(1));
  for (uint1 i = 0; i < facetCount; ++i) {
    m_facets[i] = cell.getFacet(i);
    m_nbr[i] = cell.getNbr(i);
  }
  computeAllRsq();
  resetDist();
  const std::numeric_limits<real_t> lim;
  m_distGCMax = -lim.max();
  m_vDistGCMax = maxNumVertices;
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::computeAllRsq() {
  m_vRsqMax = m_slotsV.firstAlive();
  for (uint1 i = 0; i < m_slotsV.numAllocated(); ++i) {
    if (m_slotsV.isFree(i))
      continue;
    computeRsq(i);
  }
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::computeRsq(uint1 i) {
  m_rSq[i] = m_vertexPos[i][0] * m_vertexPos[i][0];
  for (uint0 k(1); k < 3; ++k)
    m_rSq[i] += m_vertexPos[i][k] * m_vertexPos[i][k];
  if (m_rSq[i] > m_rSq[m_vRsqMax]) {
    m_vRsqMax = i;
  }
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::findRsqMax() {
  real_t rSqMax = 0;
  for (uint1 i = 0; i < m_slotsV.numAllocated(); ++i) {
    if (m_slotsV.isFree(i))
      continue;
    if (m_rSq[i] > rSqMax) {
      rSqMax = m_rSq[i];
      m_vRsqMax = i;
    }
  }
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::resetDist() {
  const std::numeric_limits<real_t> lim;
  m_distMax = -lim.max();
  m_vDistMax = maxNumVertices;
  ++m_distGen;
}

template <typename real_t, bool Weighted>
real_t CellMaker<real_t, Weighted>::computeDist(uint1 i, const std::array<real_t, 3> p,
                                      const real_t rSqHalf) {
  if (m_knownDistGen[i] != m_distGen) {
    m_dist[i] =
        m_vertexPos[i][0] * p[0] + m_vertexPos[i][1] * p[1] + m_vertexPos[i][2] * p[2] - rSqHalf;
    if (m_dist[i] > m_distMax) {
      m_vDistMax = i;
      m_distMax = m_dist[i];
    }
    m_knownDistGen[i] = m_distGen;
  }
  return m_dist[i];
}

template <typename real_t, bool Weighted>
real_t CellMaker<real_t, Weighted>::computeDist(uint1 i, const std::array<real_t, 3> p) {
  if (m_knownDistGen[i] != m_distGen) {
    m_dist[i] = 0;
    for (uint0 k(0); k < 3; ++k)
      m_dist[i] += (m_vertexPos[i][k] - 0.5 * p[k]) * p[k];
    if (m_dist[i] > m_distMax) {
      m_vDistMax = i;
      m_distMax = m_dist[i];
    }
    m_knownDistGen[i] = m_distGen;
  }
  return m_dist[i];
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::computeAllDist(const std::array<real_t, 3> p, const real_t rSqHalf) {
  const std::numeric_limits<real_t> lim;
  m_distMax = -lim.max();
  m_vDistMax = maxNumVertices;
  for (uint1 i = 0; i < m_slotsV.numAllocated(); ++i) {
    if (m_slotsV.isFree(i))
      continue;
    m_dist[i] =
        m_vertexPos[i][0] * p[0] + m_vertexPos[i][1] * p[1] + m_vertexPos[i][2] * p[2] - rSqHalf;
    if (m_dist[i] > m_distMax) {
      m_vDistMax = i;
      m_distMax = m_dist[i];
    }
    m_knownDistGen[i] = m_distGen;
  }
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::renumber() {
  m_numVertices = 0;
  {
    for (uint1 i = 0; i < m_slotsV.numAllocated(); ++i) {
      if (m_slotsV.isFree(i))
        continue;
      m_renumVWrk[i] = m_numVertices;
      if (i != m_numVertices) {
        for (uint0 k(0); k < 3; ++k) {
          m_vertices[m_numVertices][k] = m_vertices[i][k];
          m_vertexPos[m_numVertices][k] = m_vertexPos[i][k];
        }
      }
      ++m_numVertices;
    }
  }
  m_slotsV.reset(m_numVertices);
  std::fill(m_arena->aliveV().begin(), m_arena->aliveV().end(), 0u);
  std::fill(m_arena->aliveV().begin(), m_arena->aliveV().begin() + m_numVertices, uint8_t(1));
  m_numFacets = 0;
  {
    for (uint1 i = 0; i < m_slotsF.numAllocated(); ++i) {
      if (m_slotsF.isFree(i))
        continue;
      m_renumFWrk[i] = m_numFacets;
      if (i != m_numFacets) {
        m_facets[m_numFacets] = m_facets[i];
        m_nbr[m_numFacets] = m_nbr[i];
      }
      ++m_numFacets;
    }
  }
  m_slotsF.reset(m_numFacets);
  std::fill(m_arena->aliveF().begin(), m_arena->aliveF().end(), 0u);
  std::fill(m_arena->aliveF().begin(), m_arena->aliveF().begin() + m_numFacets, uint8_t(1));
  updatePeakCounter(cellMakerTelemetry().peak_vertices_seen, static_cast<uint64_t>(m_numVertices));
  updatePeakCounter(cellMakerTelemetry().peak_facets_seen, static_cast<uint64_t>(m_numFacets));
  m_vRsqMax = m_renumVWrk[m_vRsqMax];
  for (uint1 i(0); i < m_numVertices; ++i) {
    for (uint0 k(0); k < 3; ++k) {
      uint1 f = getFacet(m_vertices[i][k]);
      uint1 v = getVertex(m_vertices[i][k]);
      uint1 e = getEdge(m_vertices[i][k]);
      m_vertices[i][k] = makeLabel(m_renumFWrk[f], m_renumVWrk[v], e);
    }
  }
  for (uint1 i(0); i < m_numFacets; ++i) {
    uint1 f = getFacet(m_facets[i]);
    uint1 v = getVertex(m_facets[i]);
    uint1 e = getEdge(m_facets[i]);
    m_facets[i] = makeLabel(m_renumFWrk[f], m_renumVWrk[v], e);
  }
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::computeGCOrig(uint2 indx, const std::array<real_t, 3>& pos) {
  Indx indcs(p_nbrList->getGrid().expand(indx));
  const std::array<real_t, 3>& L(p_nbrList->getBox().getL());
  for (uint0 k(0); k < 3; ++k) {
    // Wrap the grid cell center (not corner) to the nearest periodic image
    // using floor(x/L + 0.5) which shifts by ±L to minimize |center|.
    // This ensures the grid cell box [corner, corner+dL) does not straddle
    // the periodic boundary, so computeRsqMinGC and computeDistGC work correctly.
    real_t center = (static_cast<real_t>(indcs[k]) + 0.5) * m_dLGC[k] - pos[k];
    center -= L[k] * floor(center / L[k] + 0.5);
    m_relOrigGC[k] = center - 0.5 * m_dLGC[k];
  }
}

template <typename real_t, bool Weighted>
real_t CellMaker<real_t, Weighted>::computeRsqMinGC() const {
  std::array<real_t, 3> dx(m_relOrigGC);
  real_t rSq(0);
  for (uint0 k(0); k < 3; ++k) {
    (dx[k] < 0 ? (dx[k] < -m_dLGC[k] ? dx[k] += m_dLGC[k] : dx[k] = 0) : dx[k]);
    rSq += dx[k] * dx[k];
  }
  return rSq;
}

template <typename real_t, bool Weighted>
real_t CellMaker<real_t, Weighted>::computeDistGC(uint1 indx) {
  real_t vDiffSq = 0, vSq = 0;
  for (uint0 k(0); k < 3; ++k) {
    real_t vDiff(m_vertexPos[indx][k] - m_relOrigGC[k]);
    (vDiff > 0 ? (vDiff > m_dLGC[k] ? vDiff -= m_dLGC[k] : vDiff = 0) : vDiff);
    vDiffSq += vDiff * vDiff;
    vSq += m_vertexPos[indx][k] * m_vertexPos[indx][k];
  }
  m_distGC[indx] = 0.5 * (vSq - vDiffSq);
  if (m_distGC[indx] > m_distGCMax) {
    m_vDistGCMax = indx;
    m_distGCMax = m_distGC[indx];
  }
  return m_distGC[indx];
}

template <typename real_t, bool Weighted>
std::array<real_t, 3> CellMaker<real_t, Weighted>::getClosestPointGC(uint1 indx) const {
  std::array<real_t, 3> pos;
  real_t vDiffSq = 0, vSq = 0;
  for (uint0 k(0); k < 3; ++k) {
    if (m_relOrigGC[k] > m_vertexPos[indx][k])
      pos[k] = m_relOrigGC[k];
    else if (m_relOrigGC[k] + m_dLGC[k] < m_vertexPos[indx][k])
      pos[k] = m_relOrigGC[k] + m_dLGC[k];
    else
      pos[k] = m_vertexPos[indx][k];
  }
  return pos;
}

template <typename real_t, bool Weighted>
real_t CellMaker<real_t, Weighted>::computeMaxDistGC() {
  if (m_slotsV.empty())
    return 0;
  uint1 v1 = m_slotsV.firstAlive();
  real_t distMax = computeDistGC(v1);
  uint0 k(0);
  while (k < 3) {
    uint1 v2 = getVertex(m_vertices[v1][k]);
    real_t dist = computeDistGC(v2);
    if (dist > distMax) {
      distMax = dist;
      v1 = v2;
      k = 0;
    } else {
      ++k;
    }
  }
  return distMax;
}

template <typename real_t, bool Weighted>
real_t CellMaker<real_t, Weighted>::computeMaxDistGCVerb() {
  if (m_slotsV.empty())
    return 0;
  uint1 v1 = m_slotsV.firstAlive();
  std::array<real_t, 3> posGC(getClosestPointGC(v1));
  real_t rSqHalf = 0.5 * (posGC[0] * posGC[0] + posGC[1] * posGC[1] + posGC[2] * posGC[2]);
  real_t distMax = m_vertexPos[v1][0] * posGC[0] + m_vertexPos[v1][1] * posGC[1] +
                   m_vertexPos[v1][2] * posGC[2] - rSqHalf;
  printf("vertex %u, dist %f\n", v1, distMax);
  bool hasMoved;
  do {
    hasMoved = false;
    uint0 k(0);
    do {
      uint1 v2 = getVertex(m_vertices[v1][k]);
      real_t dist = m_vertexPos[v2][0] * posGC[0] + m_vertexPos[v2][1] * posGC[1] +
                    m_vertexPos[v2][2] * posGC[2] - rSqHalf;
      printf("vertex %u, edge %u -> vertex %u,  dist %f, distMax %f\n", v1, k, v2, dist, distMax);
      if (dist > distMax) {
        v1 = v2;
        posGC = getClosestPointGC(v1);
        rSqHalf = 0.5 * (posGC[0] * posGC[0] + posGC[1] * posGC[1] + posGC[2] * posGC[2]);
        distMax = m_vertexPos[v1][0] * posGC[0] + m_vertexPos[v1][1] * posGC[1] +
                  m_vertexPos[v1][2] * posGC[2] - rSqHalf;
        hasMoved = true;
        k = 0;
      } else {
        ++k;
      }
    } while (k < 3);
  } while (hasMoved);
  return distMax;
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::computeAllDistGC() {
  const std::numeric_limits<real_t> lim;
  m_distGCMax = -lim.max();
  m_vDistGCMax = maxNumVertices;
  for (uint1 indx = 0; indx < m_slotsV.numAllocated(); ++indx) {
    if (m_slotsV.isFree(indx))
      continue;
    computeDistGC(indx);
  }
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::getAllDistGCVerb() {
  // compute 0.5 [v^2 - (p-v)^2]
  //  for (uint1 indx = m_freeV.beginIndx(); indx != m_freeV.endIndx() ; indx =
  //  m_freeV.nextIndx(indx))
  //    printf("vertex %u, dist %f\n", indx, m_distGC[indx]);
  printf("maximum vertex: %u, %f, ", m_vDistGCMax, m_distGC[m_vDistGCMax]);
  printf("neighbor vertex: %u, %f, ", getVertex(m_vertices[m_vDistGCMax][0]),
         m_distGC[getVertex(m_vertices[m_vDistGCMax][0])]);
  printf("neighbor vertex: %u, %f, ", getVertex(m_vertices[m_vDistGCMax][1]),
         m_distGC[getVertex(m_vertices[m_vDistGCMax][1])]);
  printf("neighbor vertex: %u, %f\n", getVertex(m_vertices[m_vDistGCMax][2]),
         m_distGC[getVertex(m_vertices[m_vDistGCMax][2])]);
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::cutCell2(const std::array<real_t, 3> p, real_t rSqHalf, uint2 nbr) {
  //    printf("entering cutCell\n");
  resetDist();
  if (m_slotsV.empty())
    return false;
  uint1 v1;
  if (m_vDistMax != maxNumVertices && !m_slotsV.isFree(m_vDistMax))
    v1 = m_vDistMax;
  else
    v1 = m_slotsV.firstAlive();

  // find an edge where the sign of m_dist changes
  bool found(false);
  uint1 edgeStart(0);
  //    printf("computeDist(%u, p, rSqHalf): %f\n", v1, computeDist(v1, p, rSqHalf));
  if (computeDist(v1, p, rSqHalf) > 0) {
    // find the first negative vertex by going down
    uint0 k(0);
    while (m_dist[v1] > 0 && k < 3) {
      k = 0;
      while (k < 3 && computeDist(getVertex(m_vertices[v1][k]), p, rSqHalf) >= m_dist[v1])
        ++k;
      if (k < 3) {
        // smaller value found
        edgeStart = (v1 << 2 | k);
        v1 = getVertex(m_vertices[v1][k]);
      }
    }
    if (m_dist[v1] <= 0) {
      found = true;
    } else {
      // all found distances larger than zero
      // for Voronoi construction this can only occur due to round-off errors...
      // lets do an exhaustive search to find a possible negative distance
      computeAllDist(p, rSqHalf);
      real_t distMax = 0;
      for (uint1 i = 0; i < m_slotsV.numAllocated(); ++i) {
        if (m_slotsV.isFree(i))
          continue;
        if (m_dist[i] <= 0) {
          for (uint1 k(0); k < 3; ++k) {
            uint1 nbrVertex(getVertex(m_vertices[i][k]));
            bool isKnown = (m_knownDistGen[nbrVertex] == m_distGen);
            real_t nbrDist = computeDist(nbrVertex, p, rSqHalf);
            if (nbrDist > 0) {
              found = true;
              real_t newDist(nbrDist - m_dist[i]);
              if (newDist > distMax) {
                edgeStart = m_vertices[i][k];
                distMax = newDist;
              }
            }
          }
        }
      }
    }
    if (!found) {
      // all vertices on negative side of cut-plane -> delete all
      m_slotsV.reset(0);
      m_slotsF.reset(0);
      return true;
    }
  } else {
    uint0 k(0);
    while (m_dist[v1] <= 0 && k < 3) {
      k = 0;
      while (k < 3 && computeDist(getVertex(m_vertices[v1][k]), p, rSqHalf) <= m_dist[v1])
        ++k;
      if (k < 3) {
        // larger value found
        edgeStart = m_vertices[v1][k];
        v1 = getVertex(edgeStart);
      }
    }
    found = (m_dist[v1] > 0);
  }
  if (!found)
    return found;
  //    printf("cutCell %u, found: %i\n", m_id, found);

  // Starting from edgeStart trace out a path that bounds a connected region
  // The edges where the sign changes are traced out
  // If there are disconnected patches with the same sign only one connected region is removed
  // In this way the topology remains consistent
  uint1 label(getReverseLabel(edgeStart));
  uint1 labelRev(getReverseLabel(label));
  const uint1 vDummy = vor::maxNumVertices - 1;
  const uint1 fDummy = vor::maxNumFacets - 1;
  const uint1 fDummyShifted = (fDummy << shiftFacet);
  const uint1 lDummy = makeLabel(fDummy, vDummy, 3);
  m_newVerticesWrk.clear();
  m_facetPrevWrk.clear();
  uint1 v = getVertex(label);
  uint1 e = getEdge(label);
  uint1 fPrev = getFacet(labelRev);
  uint1 vRev = getVertex(labelRev);
  uint1 eRev = getEdge(labelRev);
  do {
    //      printf("startLabel: %u %u %u\n", getFacet(label), getVertex(label), getEdge(label));
    // printf("previous facet: %u\n",fPrev);
    m_facetPrevWrk.push_back(fPrev);
    m_vertices[vRev][eRev] = lDummy;
    uint1 vNew = allocVertexChecked("cutCell2");
    // printf("new vertex %u\n", vNew);
    m_newVerticesWrk.push_back(vNew);
    // compute new positions
    {
      real_t lambda(computeDist(vRev, p, rSqHalf) /
                    (computeDist(vRev, p, rSqHalf) - computeDist(v, p, rSqHalf)));
      for (uint0 k(0); k < 3; ++k)
        m_vertexPos[vNew][k] = lambda * m_vertexPos[v][k] + (1.0 - lambda) * m_vertexPos[vRev][k];
    }
    m_vertices[vNew][0] = label;
    // printf("connect %u %u to %u %u %u\n", v, e, fPrev, vNew, 0);
    m_vertices[v][e] = makeLabel(fPrev, vNew, 0);
    m_facets[fPrev] = m_vertices[v][e];  // avoid that m_facets points to a deleted label
    do {
      v = vRev;
      e = (eRev == 0 ? 2 : eRev - 1);
      label = labelRev;
      labelRev = m_vertices[v][e];
      vRev = getVertex(labelRev);
      eRev = getEdge(labelRev);
      if (vRev == vDummy)
        break;
      fPrev = getFacet(m_vertices[vRev][eRev]);
      // printf("label: %u %u %u\n", getFacet(m_vertices[vRev][eRev]),
      // getVertex(m_vertices[vRev][eRev]), getEdge(m_vertices[vRev][eRev]));
      m_vertices[vRev][eRev] = (fDummyShifted | (m_vertices[vRev][eRev] & (~maskFacet)));
    } while (computeDist(vRev, p, rSqHalf) > 0);
    // printf("test: vRev %u\n", vRev);
    uint1 vSwap = vRev;
    uint1 eSwap = eRev;
    vRev = v;
    eRev = e;
    label = labelRev;
    v = vSwap;
    e = eSwap;
  } while (v != vDummy);

  // form a new facet and interconnect the new vertices
  {
    uint1 facetNew = allocFacetChecked("cutCell2");
    // printf("new facet: %u\n",facetNew);
    uint1 imin = m_newVerticesWrk.size() - 1;
    uint1 numNewV = (uint1)m_newVerticesWrk.size();
    for (uint1 i(0); i < numNewV; ++i) {
      uint1 iplus = i + 1;
      if (iplus == numNewV)
        iplus = 0;
      uint1 vNew(m_newVerticesWrk[i]);
      m_vertices[vNew][1] = makeLabel(m_facetPrevWrk[i], m_newVerticesWrk[imin], 2);
      m_vertices[vNew][2] = makeLabel(facetNew, m_newVerticesWrk[iplus], 1);
      imin = i;
    }
    m_facets[facetNew] = makeLabel(facetNew, m_newVerticesWrk[0], 1);
    m_nbr[facetNew] = nbr;
  }
  for (uint i(0); i < m_newVerticesWrk.size(); ++i) {
    computeRsq(m_newVerticesWrk[i]);
    // computeDistGC(m_newVerticesWrk[i]);
  }

  // remove old vertices and facets using depth-first search
  bool isLargestDeleted(false);
  //    bool isVCloseGCDeleted(false);
  {
    uint1 v(getVertex(edgeStart));
    m_slotsV.release(v);
    // printf("deteled vertex %u\n", v);
    isLargestDeleted = (v == m_vRsqMax ? true : isLargestDeleted);
    //      isVCloseGCDeleted = (v == m_vDistGCMax ? true: isVCloseGCDeleted);
    m_vStackWrk.push_back(v);
    while (!m_vStackWrk.empty()) {
      v = m_vStackWrk.back();
      m_vStackWrk.pop_back();
      for (uint0 k(0); k < 3; ++k) {
        uint1 vNxt(getVertex(m_vertices[v][k]));
        if (vNxt == vDummy)
          continue;
        uint facet(getFacet(m_vertices[v][k]));
        if (facet != fDummy && !m_slotsF.isFree(facet))
          m_slotsF.release(facet);
        if (m_slotsV.isFree(vNxt))
          continue;
        m_slotsV.release(vNxt);
        isLargestDeleted = (vNxt == m_vRsqMax ? true : isLargestDeleted);
        //	  isVCloseGCDeleted = (vNxt == m_vDistGCMax ? true: isVCloseGCDeleted);
        m_vStackWrk.push_back(vNxt);
        //	  printf("deteled vertex %u\n", vNxt);
      }
    }
  }
  if (isLargestDeleted)
    findRsqMax();
  // if (isVCloseGCDeleted){
  //   const std::numeric_limits<real_t> lim;
  //   m_distGCMax = -lim.max();
  //   m_vDistGCMax = maxNumVertices;
  //   for(uint1 i=m_freeV.beginIndx(); i != m_freeV.endIndx() ; i = m_freeV.nextIndx(i)){
  // 	if(m_distGC[i] > m_distGCMax){
  // 	  m_vDistGCMax = i;
  // 	  m_distGCMax = m_distGC[i];
  // 	}
  //   }
  //    }
  return true;
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::cutCell(const std::array<real_t, 3> p, real_t rSqHalf, uint2 nbr) {
  //    printf("entering cutCell\n");
  computeAllDist(p, rSqHalf);
  if (m_distMax <= 0)
    return false;  // no cell cut
  // Find an edge for which the vertices change sign
  // edgeStart will be the edge where the change is largest
  real_t distMax = 0;
  uint1 edgeStart(0);
  for (uint1 i = 0; i < m_slotsV.numAllocated(); ++i) {
    if (m_slotsV.isFree(i))
      continue;
    if (m_dist[i] <= 0)
      continue;
    for (uint1 k(0); k < 3; ++k) {
      uint1 nbrVertex(getVertex(m_vertices[i][k]));
      real_t nbrDist = m_dist[nbrVertex];
      if (nbrDist > 0)
        continue;
      real_t newDist(m_dist[i] - nbrDist);
      if (newDist > distMax) {
        edgeStart = (i << 2 | k);
        distMax = newDist;
      }
    }
  }
  if (distMax == 0) {  // the full cell will be deleted
    m_slotsV.reset(0);
    m_slotsF.reset(0);
    return true;
  }
  // Starting from edgeStart trace out a path that bounds a connected region
  // The edges where the sign changes are traced out
  // If there are disconnected patches with the same sign only one connected region is removed
  // In this way the topology remains consistent
  uint1 label(getReverseLabel(edgeStart));
  uint1 labelRev(getReverseLabel(label));
  const uint1 vDummy = vor::maxNumVertices - 1;
  const uint1 fDummy = vor::maxNumFacets - 1;
  const uint1 fDummyShifted = (fDummy << shiftFacet);
  const uint1 lDummy = makeLabel(fDummy, vDummy, 3);
  m_newVerticesWrk.clear();
  m_facetPrevWrk.clear();
  uint1 v = getVertex(label);
  uint1 e = getEdge(label);
  uint1 fPrev = getFacet(labelRev);
  uint1 vRev = getVertex(labelRev);
  uint1 eRev = getEdge(labelRev);
  do {
    //      printf("startLabel: %u %u %u\n", getFacet(label), getVertex(label), getEdge(label));
    // printf("previous facet: %u\n",fPrev);
    m_facetPrevWrk.push_back(fPrev);
    m_vertices[vRev][eRev] = lDummy;
    uint1 vNew = allocVertexChecked("cutCell");
    // printf("new vertex %u\n", vNew);
    m_newVerticesWrk.push_back(vNew);
    // compute new positions
    {
      real_t lambda(computeDist(vRev, p, rSqHalf) /
                    (computeDist(vRev, p, rSqHalf) - computeDist(v, p, rSqHalf)));
      for (uint0 k(0); k < 3; ++k)
        m_vertexPos[vNew][k] = lambda * m_vertexPos[v][k] + (1.0 - lambda) * m_vertexPos[vRev][k];
    }
    m_vertices[vNew][0] = label;
    // printf("connect %u %u to %u %u %u\n", v, e, fPrev, vNew, 0);
    m_vertices[v][e] = makeLabel(fPrev, vNew, 0);
    m_facets[fPrev] = m_vertices[v][e];  // avoid that m_facets points to a deleted label
    do {
      v = vRev;
      e = (eRev == 0 ? 2 : eRev - 1);
      label = labelRev;
      labelRev = m_vertices[v][e];
      vRev = getVertex(labelRev);
      eRev = getEdge(labelRev);
      if (vRev == vDummy)
        break;
      fPrev = getFacet(m_vertices[vRev][eRev]);
      // printf("label: %u %u %u\n", getFacet(m_vertices[vRev][eRev]),
      // getVertex(m_vertices[vRev][eRev]), getEdge(m_vertices[vRev][eRev]));
      m_vertices[vRev][eRev] = (fDummyShifted | (m_vertices[vRev][eRev] & (~maskFacet)));
    } while (computeDist(vRev, p, rSqHalf) > 0);
    // printf("test: vRev %u\n", vRev);
    uint1 vSwap = vRev;
    uint1 eSwap = eRev;
    vRev = v;
    eRev = e;
    label = labelRev;
    v = vSwap;
    e = eSwap;
  } while (v != vDummy);

  // form a new facet and interconnect the new vertices
  {
    uint1 facetNew = allocFacetChecked("cutCell");
    // printf("new facet: %u\n",facetNew);
    uint1 imin = m_newVerticesWrk.size() - 1;
    uint1 numNewV = (uint1)m_newVerticesWrk.size();
    for (uint1 i(0); i < numNewV; ++i) {
      uint1 iplus = i + 1;
      if (iplus == numNewV)
        iplus = 0;
      uint1 vNew(m_newVerticesWrk[i]);
      m_vertices[vNew][1] = makeLabel(m_facetPrevWrk[i], m_newVerticesWrk[imin], 2);
      m_vertices[vNew][2] = makeLabel(facetNew, m_newVerticesWrk[iplus], 1);
      imin = i;
    }
    m_facets[facetNew] = makeLabel(facetNew, m_newVerticesWrk[0], 1);
    m_nbr[facetNew] = nbr;
  }
  for (uint i(0); i < m_newVerticesWrk.size(); ++i) {
    computeRsq(m_newVerticesWrk[i]);
    // computeDistGC(m_newVerticesWrk[i]);
  }

  // remove old vertices and facets using depth-first search
  bool isLargestDeleted(false);
  //    bool isVCloseGCDeleted(false);
  {
    uint1 v(getVertex(edgeStart));
    m_slotsV.release(v);
    // printf("deteled vertex %u\n", v);
    isLargestDeleted = (v == m_vRsqMax ? true : isLargestDeleted);
    //      isVCloseGCDeleted = (v == m_vDistGCMax ? true: isVCloseGCDeleted);
    m_vStackWrk.push_back(v);
    while (!m_vStackWrk.empty()) {
      v = m_vStackWrk.back();
      m_vStackWrk.pop_back();
      for (uint0 k(0); k < 3; ++k) {
        uint1 vNxt(getVertex(m_vertices[v][k]));
        if (vNxt == vDummy)
          continue;
        uint facet(getFacet(m_vertices[v][k]));
        if (facet != fDummy && !m_slotsF.isFree(facet))
          m_slotsF.release(facet);
        if (m_slotsV.isFree(vNxt))
          continue;
        m_slotsV.release(vNxt);
        isLargestDeleted = (vNxt == m_vRsqMax ? true : isLargestDeleted);
        //	  isVCloseGCDeleted = (vNxt == m_vDistGCMax ? true: isVCloseGCDeleted);
        m_vStackWrk.push_back(vNxt);
        //	  printf("deteled vertex %u\n", vNxt);
      }
    }
  }
  if (isLargestDeleted)
    findRsqMax();
  // if (isVCloseGCDeleted){
  //   const std::numeric_limits<real_t> lim;
  //   m_distGCMax = -lim.max();
  //   m_vDistGCMax = maxNumVertices;
  //   for(uint1 i=m_freeV.beginIndx(); i != m_freeV.endIndx() ; i = m_freeV.nextIndx(i)){
  // 	if(m_distGC[i] > m_distGCMax){
  // 	  m_vDistGCMax = i;
  // 	  m_distGCMax = m_distGC[i];
  // 	}
  //   }
  //    }
  return true;
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::build(uint2 id, const std::vector<std::array<real_t, 3> >& pos,
                              const NbrList<uint2, real_t>& nbrList, const Cell<real_t>& initCell) {
  return buildWithNeighborSearch(id, pos, nbrList, initCell, nullptr);
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::build(uint2 id, const std::vector<std::array<real_t, 3> >& pos,
                              const NbrList<uint2, real_t>& nbrList, const Cell<real_t>& initCell,
                              const std::vector<uint2>& skipNbrs) {
  return buildWithNeighborSearch(id, pos, nbrList, initCell, &skipNbrs);
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::buildWithNeighborSearch(uint2 id,
                                                const std::vector<std::array<real_t, 3> >& pos,
                                                const NbrList<uint2, real_t>& nbrList,
                                                const Cell<real_t>& initCell,
                                                const std::vector<uint2>* skipNbrs) {
  p_nbrList = &nbrList;
  {
    const std::array<real_t, 3>& L(p_nbrList->getBox().getL());
    const Indx& N(p_nbrList->getGrid().getN());
    for (uint0 k(0); k < 3; ++k)
      m_dLGC[k] = L[k] / static_cast<real_t>(N[k]);
  }
  bool isUpdated = false;
  m_id = id;
  this->init(initCell);
  if (p_nbrList->getGrid().numCells() != m_visited.size())
    m_visited.init(p_nbrList->getGrid().numCells());
  else
    m_visited.reset();
  m_checkGridCell.clear();
  m_checkGCHead = 0;
  std::vector<uint2> nbrGC;
  p_nbrList->getGridNbrs(pos[id], nbrGC);
  for (uint2 j(0); j < nbrGC.size(); ++j) {
    if (!m_visited.isVisited(nbrGC[j])) {
      m_checkGridCell.push_back(nbrGC[j]);
      m_visited.set(nbrGC[j]);
    }
  }
  typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin;
  typename std::vector<PosAndId<uint2, real_t> >::const_iterator end;
  // one trial loop without checking of nbr cells
  size_t headEnd = m_checkGridCell.size();
  for (; m_checkGCHead < headEnd; ++m_checkGCHead) {
    uint2 indx = m_checkGridCell[m_checkGCHead];
    p_nbrList->getCellContent(indx, begin, end);
    if (end != begin)
      (processNbrsFiltered(begin, end, pos[m_id], p_nbrList->getBox(), skipNbrs) == true
           ? isUpdated = true
           : isUpdated);
    p_nbrList->getGrid().getNbrs(indx, nbrGC);
    for (uint2 j(0); j < nbrGC.size(); ++j) {
      if (!m_visited.isVisited(nbrGC[j])) {
        m_checkGridCell.push_back(nbrGC[j]);
        m_visited.set(nbrGC[j]);
      }
    }
  }
  // one loop with distance checking for nbr cells
  headEnd = static_cast<uint32_t>(m_checkGridCell.size());
  for (; m_checkGCHead < headEnd; ++m_checkGCHead) {
    uint2 indx = m_checkGridCell[m_checkGCHead];
    if constexpr (!Weighted) {
      computeGCOrig(indx, pos[m_id]);
      real_t rSqMin = computeRsqMinGC();
      if (rSqMin > 4.0 * m_rSq[m_vRsqMax])
        continue;
    }
    p_nbrList->getCellContent(indx, begin, end);
    if (end != begin)
      (processNbrsFiltered(begin, end, pos[m_id], p_nbrList->getBox(), skipNbrs) == true
           ? isUpdated = true
           : isUpdated);
    p_nbrList->getGrid().getNbrs(indx, nbrGC);  // for LE b.c. this needs to be adapted
    for (uint2 j(0); j < nbrGC.size(); ++j) {
      if (!m_visited.isVisited(nbrGC[j])) {
        m_checkGridCell.push_back(nbrGC[j]);
        m_visited.set(nbrGC[j]);
      }
    }
  }
  // outer loop with exhautive nbr cell checking
  while (m_checkGCHead < m_checkGridCell.size()) {
    uint2 indx = m_checkGridCell[m_checkGCHead++];
    if constexpr (!Weighted) {
      computeGCOrig(indx, pos[m_id]);
      real_t rSqMin = computeRsqMinGC();
      if (rSqMin > 4.0 * m_rSq[m_vRsqMax])
        continue;
      computeAllDistGC();
      if (m_distGCMax < 0)
        continue;
    }
    p_nbrList->getCellContent(indx, begin, end);
    if (end != begin)
      (processNbrsFiltered(begin, end, pos[m_id], p_nbrList->getBox(), skipNbrs) == true
           ? isUpdated = true
           : isUpdated);
    p_nbrList->getGrid().getNbrs(indx, nbrGC);
    for (uint2 j(0); j < nbrGC.size(); ++j) {
      if (!m_visited.isVisited(nbrGC[j])) {
        m_checkGridCell.push_back(nbrGC[j]);
        m_visited.set(nbrGC[j]);
      }
    }
  }
  //    printf("cell: %u num checked:%u\n", m_id, numChecked);
  return isUpdated;
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::rebuild(const std::vector<std::array<real_t, 3> >& pos,
                                const Box<real_t>& box, const Cell<real_t>& initCell) {
  m_nbrsWrk.clear();
  for (uint1 i = 0; i < m_slotsF.numAllocated(); ++i) {
    if (m_slotsF.isFree(i))
      continue;
    if (m_nbr[i] != noNbr) {
      PosAndId<uint2, real_t> newNbr;
      newNbr.id = m_nbr[i];
      newNbr.pos = pos[newNbr.id];
      m_nbrsWrk.push_back(newNbr);
    }
  }
  m_isAllCut = true;
  // for(uint2 i(0); i< m_indcsNbrsWrk.size(); ++i)
  //   printf("cell %u, neigb: %u\n", m_id, m_indcsNbrsWrk[i]);
  // printf("\n");
  this->init(initCell);
  // typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin = m_indcsNbrsWrk.begin();
  // typename std::vector<PosAndId<uint2, real_t> >::const_iterator end = m_indcsNbrsWrk.end();
  processNbrs(m_nbrsWrk.begin(), m_nbrsWrk.end(), pos[m_id], box);
  return m_isAllCut;
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::processNbrs(
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin,
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator end,
    const std::array<real_t, 3> pos0, const Box<real_t>& box) {
  //    printf("number of nbrs: %lu\n", end-begin);
  bool isCut(false);
  m_nbrDistWrk.clear();
  m_nbrDistWrk.reserve(end - begin);
  NbrDist<real_t> nbrDist;
  for (typename std::vector<PosAndId<uint2, real_t> >::const_iterator itr(begin); itr != end;
       ++itr) {
    if (itr->id == m_id)
      continue;
    nbrDist.id = itr->id;
    //      printf("nbr: %u ", itr->id);
    std::array<real_t, 3> relPos;
    for (uint0 k(0); k < 3; ++k)
      relPos[k] = (itr->pos)[k] - pos0[k];
    box.makeShortestDistance(relPos);
    for (uint0 k(0); k < 3; ++k)
      nbrDist[k] = relPos[k];
    nbrDist.rSqHalf = 0.5 * (relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2]);
    const real_t relNorm = std::sqrt(std::max(real_t(0), real_t(2) * nbrDist.rSqHalf));
    if constexpr (Weighted) {
      m_nbrDistWrk.push_back(nbrDist);
    } else if (candidateMightCut(cutPlaneOffset(nbrDist.rSqHalf, nbrDist.id), relNorm)) {
      m_nbrDistWrk.push_back(nbrDist);
    }
  }
  std::sort(m_nbrDistWrk.begin(), m_nbrDistWrk.end(), CompareNbrDist<real_t>());
  for (size_t i(0); i < m_nbrDistWrk.size(); ++i) {
    const NbrDist<real_t>& p(m_nbrDistWrk[i]);
    if constexpr (Weighted) {
      (applyCut(p, p.rSqHalf, p.id) ? isCut = true : m_isAllCut = false);
    } else {
      const real_t relNorm = std::sqrt(std::max(real_t(0), real_t(2) * p.rSqHalf));
      if (!candidateMightCut(cutPlaneOffset(p.rSqHalf, p.id), relNorm))
        break;
      (applyCut(p, p.rSqHalf, p.id) ? isCut = true : m_isAllCut = false);
    }
  }
  return isCut;
}

template <typename real_t, bool Weighted>
bool CellMaker<real_t, Weighted>::processNbrsFiltered(
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin,
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator end,
    const std::array<real_t, 3> pos0, const Box<real_t>& box, const std::vector<uint2>* skipNbrs) {
  if (skipNbrs == nullptr || skipNbrs->empty())
    return processNbrs(begin, end, pos0, box);

  m_nbrsWrk.clear();
  m_nbrsWrk.reserve(static_cast<size_t>(end - begin));
  for (typename std::vector<PosAndId<uint2, real_t> >::const_iterator itr(begin); itr != end;
       ++itr) {
    if (std::binary_search(skipNbrs->begin(), skipNbrs->end(), itr->id))
      continue;
    m_nbrsWrk.push_back(*itr);
  }
  if (m_nbrsWrk.empty())
    return false;
  return processNbrs(m_nbrsWrk.begin(), m_nbrsWrk.end(), pos0, box);
}

template <typename real_t, bool Weighted>
void CellMaker<real_t, Weighted>::getCloseNbrs(NbrInsert& nbrs) {
  for (uint k(0); k < 3; ++k) {
    uint1 facet(getFacet(m_vertices[m_vDistMax][k]));
    nbrs[k] = m_nbr[facet];
  }
}

template <typename real_t>
CellGeometry<real_t>::CellGeometry() : p_cell(&m_ownedCell), m_vol(0.0) {}

template <typename real_t>
CellGeometry<real_t>::CellGeometry(Cell<real_t>& cell) : CellGeometry() {
  *this = cell;
}

template <typename real_t>
CellGeometry<real_t>::CellGeometry(const CellView<real_t>& cell) : CellGeometry() {
  *this = cell;
}

template <typename real_t>
CellGeometry<real_t>::CellGeometry(const CellGeometry<real_t>& rhs) : CellGeometry() {
  *this = rhs;
}

template <typename real_t>
void CellGeometry<real_t>::resetDerived() {
  m_connV.clear();
  m_rSq.clear();
  m_edgeInv.clear();
  m_volDelaunay.clear();
  m_areas.clear();
  m_vol = 0.0;
  m_omega.clear();
  m_dV.clear();
}

template <typename real_t>
CellGeometry<real_t>& CellGeometry<real_t>::operator=(Cell<real_t>& rhs) {
  m_ownedCell = rhs;
  p_cell = &m_ownedCell;
  resetDerived();
  return *this;
}

template <typename real_t>
CellGeometry<real_t>& CellGeometry<real_t>::operator=(const CellView<real_t>& rhs) {
  m_ownedCell = rhs;
  p_cell = &m_ownedCell;
  resetDerived();
  return *this;
}

template <typename real_t>
CellGeometry<real_t>& CellGeometry<real_t>::operator=(const CellGeometry<real_t>& rhs) {
  if (&rhs == this)
    return *this;
  this->m_ownedCell = rhs.m_ownedCell;
  this->p_cell = &this->m_ownedCell;
  this->m_connV = rhs.m_connV;
  this->m_rSq = rhs.m_rSq;
  this->m_edgeInv = rhs.m_edgeInv;
  this->m_volDelaunay = rhs.m_volDelaunay;
  this->m_areas = rhs.m_areas;
  this->m_vol = rhs.m_vol;
  this->m_omega = rhs.m_omega;
  this->m_dV = rhs.m_dV;
  return *this;
}

template <typename real_t>
void CellGeometry<real_t>::computeConnectingVectors(const std::vector<std::array<real_t, 3> >& pos,
                                                    const Box<real_t>& box) {
  m_connV.resize(p_cell->m_numFacets);
  m_rSq.resize(p_cell->m_numFacets);
  for (uint1 i(0); i < p_cell->m_numFacets; ++i) {
    for (uint0 k(0); k < 3; ++k)
      m_connV[i][k] = pos[p_cell->m_nbr[i]][k] - pos[(p_cell->m_id)][k];
    box.makeShortestDistance(m_connV[i]);
    m_rSq[i] = 0.5 * (m_connV[i][0] * m_connV[i][0] + m_connV[i][1] * m_connV[i][1] +
                      m_connV[i][2] * m_connV[i][2]);
  }
}

template <typename real_t>
void CellGeometry<real_t>::computeEdgeInv() {
  m_edgeInv.resize(p_cell->m_numVertices);
  m_volDelaunay.resize(p_cell->m_numVertices);
  for (uint1 i(0); i < p_cell->m_numVertices; ++i)
    for (uint0 k(0); k < 3; ++k) {
      uint1 label0(p_cell->m_vertices[i][k]);
      uint1 label1(p_cell->getReverseLabel(label0));
      if (label0 > label1)
        continue;
      uint1 e0(getEdge(label0));
      uint1 v0(getVertex(label0));
      uint1 f0(getFacet(label0));
      uint1 e1(getEdge(label1));
      uint1 v1(getVertex(label1));
      uint1 f1(getFacet(label1));
      if (f0 >= p_cell->m_numFacets || f1 >= p_cell->m_numFacets) {
        detail::CellTopologyRef ref;
        ref.id = p_cell->m_id;
        ref.vertexCount = static_cast<uint1>(p_cell->m_numVertices);
        ref.facetCount = static_cast<uint1>(p_cell->m_numFacets);
        ref.vertices = p_cell->m_vertices;
        ref.facets = p_cell->m_facets;
        detail::validateCellTopology(ref, "CellGeometry::computeEdgeInv");
      }
      m_edgeInv[v0][e0][0] = m_connV[f0][1] * m_connV[f1][2] - m_connV[f0][2] * m_connV[f1][1];
      m_edgeInv[v0][e0][1] = m_connV[f0][2] * m_connV[f1][0] - m_connV[f0][0] * m_connV[f1][2];
      m_edgeInv[v0][e0][2] = m_connV[f0][0] * m_connV[f1][1] - m_connV[f0][1] * m_connV[f1][0];
      m_edgeInv[v1][e1][0] = -m_edgeInv[v0][e0][0];
      m_edgeInv[v1][e1][1] = -m_edgeInv[v0][e0][1];
      m_edgeInv[v1][e1][2] = -m_edgeInv[v0][e0][2];
    }
  for (uint1 i(0); i < p_cell->m_numVertices; ++i) {
    uint1 indxF;
    indxF = getFacet(p_cell->m_vertices[i][2]);
    if (indxF >= p_cell->m_numFacets) {
      detail::CellTopologyRef ref;
      ref.id = p_cell->m_id;
      ref.vertexCount = static_cast<uint1>(p_cell->m_numVertices);
      ref.facetCount = static_cast<uint1>(p_cell->m_numFacets);
      ref.vertices = p_cell->m_vertices;
      ref.facets = p_cell->m_facets;
      detail::validateCellTopology(ref, "CellGeometry::computeEdgeInv");
    }
    real_t vol(m_connV[indxF][0] * m_edgeInv[i][0][0] + m_connV[indxF][1] * m_edgeInv[i][0][1] +
               m_connV[indxF][2] * m_edgeInv[i][0][2]);
    m_volDelaunay[i] = vol / (-6.0);
    for (uint0 m(0); m < 3; ++m)
      for (uint0 k(0); k < 3; ++k)
        m_edgeInv[i][m][k] /= vol;
  }
}

template <typename real_t>
void CellGeometry<real_t>::updateVertexPos() {
  for (uint1 i(0); i < p_cell->m_numVertices; ++i) {
    std::array<uint1, 3> indxF;
    indxF[0] = getFacet(p_cell->m_vertices[i][2]);
    indxF[1] = getFacet(p_cell->m_vertices[i][0]);
    indxF[2] = getFacet(p_cell->m_vertices[i][1]);
    for (uint0 k(0); k < 3; ++k)
      p_cell->m_vertexPos[i][k] =
          (m_rSq[indxF[0]] * m_edgeInv[i][0][k] + m_rSq[indxF[1]] * m_edgeInv[i][1][k] +
           m_rSq[indxF[2]] * m_edgeInv[i][2][k]);
  }
}

template <typename real_t>
bool CellGeometry<real_t>::isConvex() const {
  bool valid(true);
  for (uint1 i(0); i < p_cell->m_numVertices; ++i) {
    for (uint0 k(0); k < 3; ++k) {
      uint1 v(getVertex(p_cell->m_vertices[i][k]));
      uint0 kmin(k == 0 ? 2 : k - 1);
      uint1 f(getFacet(p_cell->m_vertices[i][kmin]));
      real_t dist(p_cell->m_vertexPos[v][0] * m_connV[f][0] +
                  p_cell->m_vertexPos[v][1] * m_connV[f][1] +
                  p_cell->m_vertexPos[v][2] * m_connV[f][2] - m_rSq[f]);
      (dist > 0 ? valid = false : valid);
    }
  }
  return valid;
}

template <typename real_t>
real_t CellGeometry<real_t>::maxConvexViolation() const {
  real_t maxDist = 0;
  for (uint1 i(0); i < p_cell->m_numVertices; ++i) {
    for (uint0 k(0); k < 3; ++k) {
      uint1 v(getVertex(p_cell->m_vertices[i][k]));
      uint0 kmin(k == 0 ? 2 : k - 1);
      uint1 f(getFacet(p_cell->m_vertices[i][kmin]));
      real_t dist(p_cell->m_vertexPos[v][0] * m_connV[f][0] +
                  p_cell->m_vertexPos[v][1] * m_connV[f][1] +
                  p_cell->m_vertexPos[v][2] * m_connV[f][2] - m_rSq[f]);
      if (dist > maxDist)
        maxDist = dist;
    }
  }
  return maxDist;
}

template <typename real_t>
void CellGeometry<real_t>::computeVolume() {
  m_vol = 0;
  for (uint1 i(0); i < p_cell->m_numVertices; ++i)
    for (uint0 k(0); k < 3; ++k) {
      uint1 label0(p_cell->m_vertices[i][k]);
      uint1 label1(p_cell->getReverseLabel(label0));
      if (label0 > label1)
        continue;
      uint1 e0(getEdge(label0));
      uint1 v0(getVertex(label0));
      uint1 f0(getFacet(label0));
      uint1 e1(getEdge(label1));
      uint1 v1(getVertex(label1));
      uint1 f1(getFacet(label1));
      m_vol += (m_connV[f0][0] - m_connV[f1][0]) *
               (p_cell->m_vertexPos[v0][1] * p_cell->m_vertexPos[v1][2] -
                p_cell->m_vertexPos[v0][2] * p_cell->m_vertexPos[v1][1]);
      m_vol += (m_connV[f0][1] - m_connV[f1][1]) *
               (p_cell->m_vertexPos[v0][2] * p_cell->m_vertexPos[v1][0] -
                p_cell->m_vertexPos[v0][0] * p_cell->m_vertexPos[v1][2]);
      m_vol += (m_connV[f0][2] - m_connV[f1][2]) *
               (p_cell->m_vertexPos[v0][0] * p_cell->m_vertexPos[v1][1] -
                p_cell->m_vertexPos[v0][1] * p_cell->m_vertexPos[v1][0]);
    }
  m_vol /= 12.0;
}

template <typename real_t>
void CellGeometry<real_t>::computeAreas() {
  m_vol = 0;
  m_areas.clear();
  m_areas.resize(p_cell->m_numFacets);
  for (uint i(0); i < p_cell->m_numFacets; ++i)
    for (uint0 k(0); k < 3; ++k)
      m_areas[i][k] = 0;
  std::array<real_t, 3> dA;
  for (uint1 i(0); i < p_cell->m_numVertices; ++i)
    for (uint0 k(0); k < 3; ++k) {
      uint1 label0(p_cell->m_vertices[i][k]);
      uint1 label1(p_cell->getReverseLabel(label0));
      if (label0 > label1)
        continue;
      uint1 e0(getEdge(label0));
      uint1 v0(getVertex(label0));
      uint1 f0(getFacet(label0));
      uint1 e1(getEdge(label1));
      uint1 v1(getVertex(label1));
      uint1 f1(getFacet(label1));
      dA[0] = p_cell->m_vertexPos[v0][1] * p_cell->m_vertexPos[v1][2] -
              p_cell->m_vertexPos[v0][2] * p_cell->m_vertexPos[v1][1];
      dA[1] = p_cell->m_vertexPos[v0][2] * p_cell->m_vertexPos[v1][0] -
              p_cell->m_vertexPos[v0][0] * p_cell->m_vertexPos[v1][2];
      dA[2] = p_cell->m_vertexPos[v0][0] * p_cell->m_vertexPos[v1][1] -
              p_cell->m_vertexPos[v0][1] * p_cell->m_vertexPos[v1][0];
      for (uint0 k(0); k < 3; ++k) {
        m_areas[f0][k] += dA[k];
        m_areas[f1][k] -= dA[k];
      }
    }
  for (uint1 i(0); i < m_areas.size(); ++i) {
    for (uint0 k(0); k < 3; ++k) {
      m_areas[i][k] *= 0.5;
      m_vol += m_areas[i][k] * m_connV[i][k];
    }
  }
  m_vol /= 6.0;
  printf("volume: %f\n", m_vol);
}

template <typename real_t>
void CellGeometry<real_t>::diffVolume() {
  m_vol = 0;
  m_dV.clear();
  m_dV.resize(p_cell->m_numFacets);
  m_areas.resize(p_cell->m_numFacets);
  for (uint1 i(0); i < p_cell->m_numFacets; ++i)
    for (uint0 k(0); k < 3; ++k) {
      m_dV[i][k] = 0;
      m_areas[i][k] = 0;
    }
  real_t dA[3];
  real_t ddA[3];
  real_t dVertex[3][3][3];
  //    real_t vol(0);
  uint1 f[3];
  uint1 vNbr[3];
  const uint0 eOpp[3] = {2, 0, 1};  // edge opposite to facet
  real_t dv[3][3];
  for (uint1 vc(0); vc < p_cell->m_numVertices; ++vc) {
    vNbr[0] = getVertex(p_cell->m_vertices[vc][0]);
    f[2] = getFacet(p_cell->m_vertices[vc][0]);
    vNbr[1] = getVertex(p_cell->m_vertices[vc][1]);
    f[0] = getFacet(p_cell->m_vertices[vc][1]);
    vNbr[2] = getVertex(p_cell->m_vertices[vc][2]);
    f[1] = getFacet(p_cell->m_vertices[vc][2]);
    for (uint0 k(0); k < 3; ++k) {
      dv[0][k] = p_cell->m_vertexPos[vNbr[0]][k] - p_cell->m_vertexPos[vNbr[1]][k];
      dv[1][k] = p_cell->m_vertexPos[vNbr[1]][k] - p_cell->m_vertexPos[vNbr[2]][k];
      dv[2][k] = p_cell->m_vertexPos[vNbr[2]][k] - p_cell->m_vertexPos[vNbr[0]][k];
    }
    // vertex vc (direction l) differentiated to position of m_connV[f[i]] (direction j)
    for (uint0 j(0); j < 3; ++j)
      for (uint0 i(0); i < 3; ++i)
        for (uint0 l(0); l < 3; ++l) {
          dVertex[j][i][l] =
              m_edgeInv[vc][eOpp[i]][l] * (m_connV[f[i]][j] - p_cell->m_vertexPos[vc][j]);
          //	    dVertex[j][i][l] = (std::fabs(dVertex[j][i][l])>5e0? copysign(0, dVertex[j][i][l])
          //: dVertex[j][i][l]);
        }
    for (uint0 m(0); m < 3; ++m) {
      dA[0] = p_cell->m_vertexPos[vc][1] * dv[m][2] - p_cell->m_vertexPos[vc][2] * dv[m][1];
      dA[1] = p_cell->m_vertexPos[vc][2] * dv[m][0] - p_cell->m_vertexPos[vc][0] * dv[m][2];
      dA[2] = p_cell->m_vertexPos[vc][0] * dv[m][1] - p_cell->m_vertexPos[vc][1] * dv[m][0];
      for (uint0 k(0); k < 3; ++k) {
        m_areas[f[m]][k] += 0.25 * dA[k];
        //	  vol += m_connV[f[m]][k]*dA[k]/24.0;
      }
      for (uint0 j(0); j < 3; ++j) {
        for (uint0 i(0); i < 3; ++i) {
          ddA[0] = dVertex[j][i][1] * dv[m][2] - dVertex[j][i][2] * dv[m][1];
          ddA[1] = dVertex[j][i][2] * dv[m][0] - dVertex[j][i][0] * dv[m][2];
          ddA[2] = dVertex[j][i][0] * dv[m][1] - dVertex[j][i][1] * dv[m][0];
          for (uint0 k(0); k < 3; ++k)
            m_dV[f[i]][j] += m_connV[f[m]][k] * ddA[k] / 12.0;
        }
        m_dV[f[m]][j] += dA[j] / 24.0;
      }
    }
  }
  for (uint1 i(0); i < p_cell->m_numFacets; ++i)
    for (uint0 k(0); k < 3; ++k) {
      m_vol += m_areas[i][k] * m_connV[i][k];
    }
  m_vol /= 6.0;
  //    printf("connecting vector: %f %f %f\n", m_connV[0][0], m_connV[0][1], m_connV[0][2]);
  // printf("volume: %f %f\n", m_vol, vol);
  //    printf("nbr: %u\n",p_cell->m_nbr[0]);
}

template <typename real_t>
void CellGeometry<real_t>::computeAll() {
  bool isDetected = false;
  const uint0 numFacets(p_cell->m_numFacets);
  const uint0 numVertices(p_cell->m_numVertices);
  m_vol = 0;
  m_dV.resize(numFacets);
  m_areas.resize(numFacets);
  m_omega.resize(numFacets);
  std::vector<std::array<std::array<real_t, 3>, 3> > dA(numVertices);
  std::vector<std::array<std::array<real_t, 3>, 3> > dv(numVertices);
  std::vector<std::array<real_t, 3> > xcm(numFacets);
  std::vector<real_t> volFacet(numFacets, 0.0);
  uint1 f[3];
  uint1 vNbr[3];
  //    real_t omega[numFacets][3][3][3];
  real_t dx[3];
  for (uint1 i(0); i < numFacets; ++i) {
    for (uint0 k(0); k < 3; ++k) {
      m_areas[i][k] = 0;
      xcm[i][k] = 0;
      for (uint0 l(0); l < 3; ++l)
        for (uint0 m(0); m < 3; ++m)
          m_omega[i][k][l][m] = 0.0;
    }
  }
  const uint0 eOpp[3] = {2, 0, 1};  // edge opposite to facet
  for (uint1 vc(0); vc < p_cell->m_numVertices; ++vc) {
    vNbr[0] = getVertex(p_cell->m_vertices[vc][0]);
    f[2] = getFacet(p_cell->m_vertices[vc][0]);
    vNbr[1] = getVertex(p_cell->m_vertices[vc][1]);
    f[0] = getFacet(p_cell->m_vertices[vc][1]);
    vNbr[2] = getVertex(p_cell->m_vertices[vc][2]);
    f[1] = getFacet(p_cell->m_vertices[vc][2]);
    for (uint0 k(0); k < 3; ++k) {
      dv[vc][0][k] = p_cell->m_vertexPos[vNbr[0]][k] - p_cell->m_vertexPos[vNbr[1]][k];
      dv[vc][1][k] = p_cell->m_vertexPos[vNbr[1]][k] - p_cell->m_vertexPos[vNbr[2]][k];
      dv[vc][2][k] = p_cell->m_vertexPos[vNbr[2]][k] - p_cell->m_vertexPos[vNbr[0]][k];
    }
    for (uint0 m(0); m < 3; ++m) {
      for (uint0 k(0); k < 3; ++k)
        dx[k] = p_cell->m_vertexPos[vc][k] - 0.5 * m_connV[f[m]][k];
      dA[vc][m][0] = dx[1] * dv[vc][m][2] - dx[2] * dv[vc][m][1];
      dA[vc][m][1] = dx[2] * dv[vc][m][0] - dx[0] * dv[vc][m][2];
      dA[vc][m][2] = dx[0] * dv[vc][m][1] - dx[1] * dv[vc][m][0];
      real_t dVol = 0;
      for (uint0 k(0); k < 3; ++k) {
        m_areas[f[m]][k] += dA[vc][m][k];
        dVol += m_connV[f[m]][k] * dA[vc][m][k];
      }
      volFacet[f[m]] += dVol;
      for (uint0 k(0); k < 3; ++k)
        xcm[f[m]][k] += dVol * p_cell->m_vertexPos[vc][k];
    }
  }
  for (uint1 i(0); i < numFacets; ++i) {
    for (uint0 k(0); k < 3; ++k) {
      xcm[i][k] /= volFacet[i];
      m_areas[i][k] *= 0.25;
    }
    m_vol += volFacet[i];
  }
  m_vol /= 24;
  real_t dAtot[3];
  for (uint1 vc(0); vc < p_cell->m_numVertices; ++vc) {
    f[2] = getFacet(p_cell->m_vertices[vc][0]);
    f[0] = getFacet(p_cell->m_vertices[vc][1]);
    f[1] = getFacet(p_cell->m_vertices[vc][2]);
    for (uint0 m(0); m < 3; ++m) {
      for (uint0 k(0); k < 3; ++k)
        dx[k] = 0.5 * m_connV[f[m]][k] - xcm[f[m]][k];
      dA[vc][m][0] += dx[1] * dv[vc][m][2] - dx[2] * dv[vc][m][1];
      dA[vc][m][1] += dx[2] * dv[vc][m][0] - dx[0] * dv[vc][m][2];
      dA[vc][m][2] += dx[0] * dv[vc][m][1] - dx[1] * dv[vc][m][0];
    }
    for (uint0 k(0); k < 3; ++k)
      dAtot[k] = 0.25 * (dA[vc][0][k] + dA[vc][1][k] + dA[vc][2][k]);
    for (uint0 i(0); i < 3;
         ++i)  // facet indicating nbr particle-position towards what is differentiated
      for (uint0 j(0); j < 3; ++j) {    // direction of derivative
        for (uint0 l(0); l < 3; ++l) {  // direction of displacement
          // vertex vc (direction l) differentiated to position of m_connV[f[i]] (direction j)
          real_t dVertex =
              m_edgeInv[vc][eOpp[i]][l] * (m_connV[f[i]][j] - p_cell->m_vertexPos[vc][j]);
          for (uint0 k(0); k < 3; ++k)  // normal direction
            m_omega[f[i]][j][l][k] += dVertex * dAtot[k];
        }
      }
  }
  for (uint1 i(0); i < numFacets; ++i) {
    for (uint0 k(0); k < 3; ++k) {
      m_dV[i][k] = 0;
      for (uint0 m(0); m < 3; ++m)
        m_dV[i][k] += m_omega[i][k][m][m];
    }
  }

  //    printf("volume: %f\n", m_vol);
  //    printf("nbr: %u\n",p_cell->m_nbr[0]);
}

// template<typename real_t>
// std::array<std::array<real_t, 3>, 3> CellGeometry<real_t>::velocityGradient(const
// std::vector<std::array<real_t, 3> > & velocities) const
// {
//   std::array<std::array<real_t, 3>, 3> gradV; //gradV[i][j] = dv[i]/dx[j]
//   std::array<real_t, 3> vCenter = velocities[p_cell->m_id];
//   // omega[i][j][l][k]
//   // neighbor corresponding to facet i differentiated into j-direction
//   // l: displacement direction, k: normal direction
//   for(int l(0); l<3; ++l)
//     for(int k(0); k<3; ++k)
// 	gradV[l][k] = 0.0;
//   for(int i(0); i< m_omega.size(); ++i){
//     std::array<real_t, 3> v = velocities[p_cell->m_nbr[i]];
//     for(int j(0); j<3; ++j){
// 	real_t dv = v[j] - vCenter[j];
// 	  for(int l(0); l<3; ++l)
// 	    for(int k(0); k<3; ++k){
// 	      gradV[l][k] += m_omega[i][j][l][k]*dv;
// 	      //printf("cell: %d, dv: %f, omega[%d][%d][%d][%d]: %f\n", p_cell->getID(), dv, i,j, l, k,
// m_omega[i][j][l][k]);
// 	    }
//     }
//   }
//   for(int l(0); l<3; ++l)
//     for(int k(0); k<3; ++k){
// 	gradV[l][k] /= m_vol;
// 	//	printf("cell %d gradVel[%d][%d]: %f\n", p_cell->getID(), l, k,  gradV[l][k]);
//     }
//   return gradV;
// }

template <typename real_t>
std::array<std::array<real_t, 3>, 3> CellGeometry<real_t>::velocityGradient(
    const std::vector<std::array<real_t, 3> >& velocities) const {
  std::array<std::array<real_t, 3>, 3> gradV;  // gradV[i][j] = dv[i]/dx[j]
  for (int l(0); l < 3; ++l)
    for (int k(0); k < 3; ++k)
      gradV[l][k] = 0.0;
  for (int i(0); i < m_areas.size(); ++i) {
    std::array<real_t, 3> v = velocities[p_cell->m_nbr[i]];
    for (int j(0); j < 3; ++j)
      for (int k(0); k < 3; ++k)
        gradV[j][k] += m_areas[i][j] * v[k];
  }
  real_t factor = 0.5 / m_vol;
  for (int j(0); j < 3; ++j)
    for (int k(0); k < 3; ++k) {
      gradV[j][k] *= factor;
      //	printf("cell %d gradVel[%d][%d]: %f\n", p_cell->getID(), l, k,  gradV[l][k]);
    }
  return gradV;
}

template <typename real_t>
void CellGeometry<real_t>::getDelaunayNbrs(uint1 iVertex, std::array<uint2, 3>& nbrs) const {
  std::array<uint1, 3> indxF;
  indxF[0] = getFacet(p_cell->m_vertices[iVertex][2]);
  indxF[1] = getFacet(p_cell->m_vertices[iVertex][0]);
  indxF[2] = getFacet(p_cell->m_vertices[iVertex][1]);
  for (uint0 m(0); m < 3; ++m)
    nbrs[m] = p_cell->getNbrs()[indxF[m]];
}

template <typename real_t>
std::array<std::array<real_t, 3>, 3> CellGeometry<real_t>::velocityGradientDelaunay(
    uint1 iVertex, const std::array<uint2, 3>& nbrs,
    const std::vector<std::array<real_t, 3> >& velocities) const {
  std::array<std::array<real_t, 3>, 3> gradV;  // gradV[i][j] = dv[i]/dx[j]
  for (int l(0); l < 3; ++l)
    for (int k(0); k < 3; ++k)
      gradV[l][k] = 0.0;
  std::array<real_t, 3> v0 = velocities[p_cell->getID()];
  for (uint0 m(0); m < 3; ++m) {
    std::array<real_t, 3> v = velocities[nbrs[m]];
    std::array<real_t, 3> dv;
    for (uint0 k(0); k < 3; ++k)
      dv[k] = v[k] - v0[k];
    for (uint0 l(0); l < 3; ++l)
      for (uint0 k(0); k < 3; ++k)
        gradV[l][k] += m_edgeInv[iVertex][m][l] * dv[k];
  }
  return gradV;
}

template <typename real_t>
void CellGeometry<real_t>::computeDelaunayForces(uint1 iVertex,
                                                 const std::array<std::array<real_t, 3>, 3>& stress,
                                                 std::array<std::array<real_t, 3>, 3>& forces) {
  for (uint0 m(0); m < 3; ++m)
    for (uint0 k(0); k < 3; ++k) {
      forces[m][k] = 0;
      for (uint0 l(0); l < 3; ++l)
        forces[m][k] += stress[k][l] * m_edgeInv[iVertex][m][l];
      forces[m][k] *= -m_volDelaunay[iVertex];
    }
}

template <typename real_t>
std::array<real_t, 3> CellGeometry<real_t>::force(
    const std::vector<std::array<std::array<real_t, 3>, 3> >& stresses) const {
  std::array<real_t, 3> f;  // gradV[i][j] = dv[i]/dx[j]
  std::array<std::array<real_t, 3>, 3> stressCenter = stresses[p_cell->id];
  // omega[i][j][l][k]
  // neighbor corresponding to facet i differentiated into j-direction
  // l: displacement direction, k: normal direction
  for (int j(0); j < 3; ++j)
    f[j] = 0.0;
  for (int i(0); i < m_omega.size(); ++i) {
    std::array<std::array<real_t, 3>, 3> stress = stresses[p_cell->m_nbr[i]];
    std::array<std::array<real_t, 3>, 3> dStress;
    for (int l(0); l < 3; ++l)
      for (int k(0); k < 3; ++k)
        dStress[l][k] = stress[l][k] - stressCenter[l][k];
    for (int j(0); j < 3; ++j)
      for (int l(0); l < 3; ++l)
        for (int k(0); k < 3; ++k)
          f[j] = -m_omega[i][j][l][k] * dStress[l][k];
  }
  return f;
}

template <typename real_t>
void CellGeometry<real_t>::gradFacetAreaSq(uint1 indxFacet, std::vector<uint2>& indxFacets,
                                           std::vector<std::array<real_t, 3> >& grad) const {
  std::vector<uint1> labels;
  labels.reserve(10);
  uint1 labelStart(p_cell->m_facets[indxFacet]);
  labels.push_back(labelStart);
  uint1 labelNext = p_cell->getNextLabelCCW(labelStart);
  while (labelNext != labelStart) {
    labels.push_back(labelNext);
    labelNext = p_cell->getNextLabelCCW(labelNext);
  }
  uint1 numV(labels.size());
  uint1 numF(numV + 1);
  indxFacets.resize(numF);
  indxFacets[0] = indxFacet;
  for (uint1 i(0); i < numV; ++i)
    indxFacets[i + 1] = getFacet(p_cell->getReverseLabel(labels[i]));
  grad.resize(numF);
  for (uint1 i(0); i < numF; ++i)
    for (uint0 k(0); k < 3; ++k)
      grad[i][k] = 0;
  uint1 nbrPrev = numV;
  uint1 nbrNext = 1;
  real_t dv[3], dVertex[3], dv_out_A[3];
  for (uint1 i(0); i < numV; ++i) {
    uint1 vc = getVertex(labels[i]);
    uint1 e0 = getEdge(labels[i]);
    uint1 e1 = (e0 == 2 ? 0 : e0 + 1);
    uint1 e2 = (e0 == 0 ? 2 : e0 - 1);
    uint1 f0 = getFacet(p_cell->m_vertices[vc][e1]);
    uint1 f1 = getFacet(p_cell->m_vertices[vc][e2]);
    uint1 f2 = getFacet(p_cell->m_vertices[vc][e0]);
    uint1 v0 = getVertex(p_cell->m_vertices[vc][e0]);
    uint1 v1 = getVertex(p_cell->m_vertices[vc][e1]);
    for (uint0 k(0); k < 3; ++k)
      dv[k] = p_cell->m_vertexPos[v0][k] - p_cell->m_vertexPos[v1][k];
    dv_out_A[0] = dv[1] * m_areas[indxFacet][2] - dv[2] * m_areas[indxFacet][1];
    dv_out_A[1] = dv[2] * m_areas[indxFacet][0] - dv[0] * m_areas[indxFacet][2];
    dv_out_A[2] = dv[0] * m_areas[indxFacet][1] - dv[1] * m_areas[indxFacet][0];
    real_t sum = 0;
    for (uint0 l(0); l < 3; ++l)                  // coordinate of vertex
      sum += m_edgeInv[vc][e0][l] * dv_out_A[l];  // vertex vc (direction l) differentiated to
                                                  // position of m_connV[fs] (direction j)
    for (uint0 j(0); j < 3; ++j)                  // direction of displacement
      grad[nbrPrev][j] += (m_connV[f1][j] - p_cell->m_vertexPos[vc][j]) * sum;
    sum = 0;
    for (uint0 l(0); l < 3; ++l)  // coordinate of vertex
      sum += m_edgeInv[vc][e1][l] * dv_out_A[l];
    for (uint0 j(0); j < 3; ++j)  // direction of displacement
      grad[nbrNext][j] += (m_connV[f2][j] - p_cell->m_vertexPos[vc][j]) * sum;
    sum = 0;
    for (uint0 l(0); l < 3; ++l)  // coordinate of vertex
      sum += m_edgeInv[vc][e2][l] * dv_out_A[l];
    for (uint0 j(0); j < 3; ++j)  // direction of displacement
      grad[0][j] += (m_connV[f0][j] - p_cell->m_vertexPos[vc][j]) * sum;
    nbrPrev = nbrNext;
    ++nbrNext;
  }
}

template <typename real_t>
void GeometryArena<real_t>::rebuildFromLegacy(const TopologyArena<real_t>& topology,
                                              const std::vector<CellGeometry<real_t> >& geoms) {
  clear();
  const size_t numCells = std::min(topology.numCells(), geoms.size());
  resize(static_cast<uint2>(numCells));
  for (size_t i = 0; i < numCells; ++i) {
    const uint2 cellId = topology.cellId(i);
    const uint1 facetCount = topology.cellNumFacets(i);
    m_ids[i] = cellId;
    m_volumes[i] = geoms[i].getVolume();
    const std::vector<std::array<real_t, 3> >& dV = geoms[i].getdV();
    const std::vector<std::array<real_t, 3> >& areas = geoms[i].getAreas();
    const std::vector<std::array<real_t, 3> >& connV = geoms[i].getConnVect();
    const std::vector<real_t>& connVSq = geoms[i].getConnVectSq();
    m_dV.insert(static_cast<uint2>(i), dV.data(), facetCount);
    m_areas.insert(static_cast<uint2>(i), areas.data(), facetCount);
    m_connV.insert(static_cast<uint2>(i), connV.data(), facetCount);
    m_connVSq.insert(static_cast<uint2>(i), connVSq.data(), facetCount);
  }
}

template <typename real_t, bool Weighted>
CellComplex<real_t, Weighted>::CellComplex(Box<real_t>* box)
    : CellComplex(box, defaultPersistentWorkerCount()) {}

template <typename real_t, bool Weighted>
CellComplex<real_t, Weighted>::CellComplex(Box<real_t>* box, size_t workerCount)
    : m_nbrList(box), m_isBuild(false) {
  m_team.start(workerCount);
  ensureWorkerContexts(std::max<size_t>(workerCount, 1));
}

template <typename real_t, bool Weighted>
size_t CellComplex<real_t, Weighted>::defaultPersistentWorkerCount() {
#ifdef VORONOI_USE_OPENMP
  const char* ompEnv = std::getenv("OMP_NUM_THREADS");
  const int ompThreads = omp_get_max_threads();
  if (ompEnv != NULL && ompEnv[0] != '\0' && ompThreads > 0)
    return static_cast<size_t>(ompThreads);
#endif
  const unsigned hw = std::thread::hardware_concurrency();
  if (hw > 8u && (hw % 2u) == 0u)
    return static_cast<size_t>(hw / 2u);
  return (hw > 1u) ? static_cast<size_t>(hw) : 0u;
}

template <typename real_t, bool Weighted>
template <typename Func>
void CellComplex<real_t, Weighted>::parallelForPersistent(size_t count, Func fn) {
  if (count == 0)
    return;

  if (m_team.threadCount() == 0) {
    for (size_t i = 0; i < count; ++i)
      fn(i, 0u);
    return;
  }

  m_team.run([count, &fn](size_t tid, size_t teamSize) {
    const size_t begin = (count * tid) / teamSize;
    const size_t end = (count * (tid + 1)) / teamSize;
    for (size_t i = begin; i < end; ++i)
      fn(i, tid);
  });
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::normalizeActivity(std::vector<uint8_t>& active) {
  for (size_t i = 0; i < active.size(); ++i)
    active[i] = (active[i] != 0u ? 1u : 0u);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::collectActiveParticleIds(const std::vector<uint8_t>& active,
                                                   std::vector<uint2>& particleIds) {
  particleIds.clear();
  particleIds.reserve(active.size());
  for (size_t i = 0; i < active.size(); ++i)
    if (active[i] != 0u)
      particleIds.push_back(static_cast<uint2>(i));
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::syncParticleActivity(size_t numParticles) {
  if constexpr (Weighted)
    this->syncWeights(numParticles);
  if (m_particleActive.empty())
    m_particleActive.assign(numParticles, 1u);
  else if (m_particleActive.size() < numParticles)
    m_particleActive.resize(numParticles, 1u);
  else if (m_particleActive.size() > numParticles)
    m_particleActive.resize(numParticles);
  normalizeActivity(m_particleActive);
}

template <typename real_t, bool Weighted>
bool CellComplex<real_t, Weighted>::allParticlesActive() const {
  for (size_t i = 0; i < m_particleActive.size(); ++i)
    if (m_particleActive[i] == 0u)
      return false;
  return true;
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::rebuildBuiltParticleMaps(size_t numParticles) {
  collectActiveParticleIds(m_particleActive, m_activeParticleIds);
  m_cellIndexByParticle.assign(numParticles, noNbr);
  if constexpr (Weighted) {
    this->m_cellParticleIds.resize(m_cellArena.numCells());
    if (this->m_particleHasCell.size() != numParticles)
      this->m_particleHasCell.resize(numParticles, 0u);
    else
      std::fill(this->m_particleHasCell.begin(), this->m_particleHasCell.end(), 0u);
  }
  for (size_t i = 0; i < m_cellArena.numCells(); ++i) {
    const uint2 particleId = m_cellArena.cellId(i);
    if constexpr (Weighted)
      this->m_cellParticleIds[i] = particleId;
    else
      m_activeParticleIds[i] = particleId;
    if (particleId < m_cellIndexByParticle.size())
      m_cellIndexByParticle[particleId] = static_cast<uint2>(i);
    if constexpr (Weighted) {
      if (particleId < this->m_particleHasCell.size())
        this->m_particleHasCell[particleId] = 1u;
    }
  }
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::setParticleActivity(const std::vector<uint8_t>& active) {
  m_particleActive = active;
  normalizeActivity(m_particleActive);
  collectActiveParticleIds(m_particleActive, m_activeParticleIds);
  if constexpr (Weighted) {
    this->syncWeights(m_particleActive.size());
    for (size_t i = 0; i < m_particleActive.size(); ++i)
      if (m_particleActive[i] == 0u)
        this->m_particleHasCell[i] = 0u;
  }
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::activateParticles(const std::vector<uint2>& particleIds) {
  for (size_t i = 0; i < particleIds.size(); ++i) {
    if (particleIds[i] >= m_particleActive.size()) {
      std::fprintf(stderr, "CellComplex::activateParticles: particle id %u out of range (%zu)\n",
                   static_cast<unsigned>(particleIds[i]), m_particleActive.size());
      std::abort();
    }
    m_particleActive[particleIds[i]] = 1u;
  }
  collectActiveParticleIds(m_particleActive, m_activeParticleIds);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::deactivateParticles(const std::vector<uint2>& particleIds) {
  for (size_t i = 0; i < particleIds.size(); ++i) {
    if (particleIds[i] >= m_particleActive.size()) {
      std::fprintf(stderr, "CellComplex::deactivateParticles: particle id %u out of range (%zu)\n",
                   static_cast<unsigned>(particleIds[i]), m_particleActive.size());
      std::abort();
    }
    m_particleActive[particleIds[i]] = 0u;
    if constexpr (Weighted)
      this->m_particleHasCell[particleIds[i]] = 0u;
  }
  collectActiveParticleIds(m_particleActive, m_activeParticleIds);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::insertParticles(std::vector<std::array<real_t, 3> >& p,
                                          const std::vector<std::array<real_t, 3> >& inserted) {
  syncParticleActivity(p.size());
  const size_t oldSize = p.size();
  p.insert(p.end(), inserted.begin(), inserted.end());
  m_particleActive.resize(p.size(), 1u);
  if constexpr (Weighted) {
    this->m_weights.resize(p.size(), real_t(0));
    this->m_particleHasCell.resize(p.size(), 0u);
    this->markWeightsDirty();
  }
  if (!m_types.empty() && m_types.size() == oldSize)
    m_types.resize(p.size(), 0u);
  collectActiveParticleIds(m_particleActive, m_activeParticleIds);
}

template <typename real_t, bool Weighted>
ParticleRenumberResult CellComplex<real_t, Weighted>::renumberParticles(
    std::vector<std::array<real_t, 3> >& p, bool rebuild) {
  syncParticleActivity(p.size());
  ParticleRenumberResult result;
  result.old_to_new.assign(p.size(), noNbr);
  result.new_to_old.reserve(p.size());

  std::vector<std::array<real_t, 3> > compactPos;
  compactPos.reserve(p.size());
  std::vector<uint0> compactTypes;
  const bool compactTypesEnabled = (!m_types.empty() && m_types.size() == p.size());
  if (compactTypesEnabled)
    compactTypes.reserve(p.size());
  std::vector<real_t> compactWeights;
  if constexpr (Weighted)
    compactWeights.reserve(p.size());
  std::vector<uint8_t> compactHasCell;
  if constexpr (Weighted)
    compactHasCell.reserve(p.size());

  for (size_t oldId = 0; oldId < p.size(); ++oldId) {
    if (m_particleActive[oldId] == 0u)
      continue;
    const uint2 newId = static_cast<uint2>(compactPos.size());
    result.old_to_new[oldId] = newId;
    result.new_to_old.push_back(static_cast<uint2>(oldId));
    compactPos.push_back(p[oldId]);
    if (compactTypesEnabled)
      compactTypes.push_back(m_types[oldId]);
    if constexpr (Weighted) {
      compactWeights.push_back(this->m_weights[oldId]);
      compactHasCell.push_back(this->m_particleHasCell[oldId]);
    }
  }

  p.swap(compactPos);
  if (compactTypesEnabled)
    m_types.swap(compactTypes);
  else if (!m_types.empty() && m_types.size() > p.size())
    m_types.resize(p.size());
  if constexpr (Weighted) {
    this->m_weights.swap(compactWeights);
    this->m_particleHasCell.swap(compactHasCell);
    this->m_cellParticleIds.clear();
  }

  m_particleActive.assign(p.size(), 1u);
  collectActiveParticleIds(m_particleActive, m_activeParticleIds);
  m_cellIndexByParticle.clear();
  m_cellArena.clear();
  m_connectivity.clear();
  clearGeometryCache();
  m_isBuild = false;
  if constexpr (Weighted)
    std::fill(this->m_particleHasCell.begin(), this->m_particleHasCell.end(), 0u);

  if (rebuild) {
    if constexpr (Weighted)
      buildWeighted(p, NULL, true);
    else
      buildFromCurrentActivity(p, true);
  }

  return result;
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::initNbrList(const std::vector<std::array<real_t, 3> >& p) {
  syncParticleActivity(p.size());
  const std::array<real_t, 3>& L(m_nbrList.getBox().getL());
  if constexpr (Weighted) {
    if (p.empty()) {
      std::vector<uint2> emptyIds;
      m_nbrList.setupSubset(p, emptyIds, real_t(1));
      return;
    }
    real_t density = real_t(p.size()) / (L[0] * L[1] * L[2]);
    real_t rcut = real_t(1.75) * (std::pow(density, real_t(-1.0 / 3.0)));
#ifdef VORONOI_USE_OPENMP
    if (omp_in_parallel())
      m_nbrList.setupCurrentTeam(p, rcut);
    else
      m_nbrList.setup(p, rcut);
#else
    m_nbrList.setup(p, rcut);
#endif
    return;
  }
  std::vector<uint2> activeParticleIds;
  collectActiveParticleIds(m_particleActive, activeParticleIds);
  if (activeParticleIds.empty()) {
    m_nbrList.setupSubset(p, activeParticleIds, real_t(1));
    return;
  }
  real_t density = real_t(activeParticleIds.size()) / (L[0] * L[1] * L[2]);
  real_t rcut = 1.75 * (std::pow(density, -1.0 / 3.0));
  const bool useAllParticles = activeParticleIds.size() == p.size();
#ifdef VORONOI_USE_OPENMP
  if (useAllParticles && omp_in_parallel())
    m_nbrList.setupCurrentTeam(p, rcut);
  else if (useAllParticles)
    m_nbrList.setup(p, rcut);
  else
    m_nbrList.setupSubset(p, activeParticleIds, rcut);
#else
  if (useAllParticles)
    m_nbrList.setup(p, rcut);
  else
    m_nbrList.setupSubset(p, activeParticleIds, rcut);
#endif
  //    printf("number grid cells: %u\n", m_nbrList.getGrid().getN()[0]);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::ensureWorkerContexts(size_t count) {
  while (m_workers.size() < count)
    m_workers.emplace_back(new WorkerContext());
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::buildFromCurrentActivity(
    const std::vector<std::array<real_t, 3> >& p, bool computeGeometry) {
  syncParticleActivity(p.size());
  std::vector<uint2> activeParticleIds;
  collectActiveParticleIds(m_particleActive, activeParticleIds);
  initNbrList(p);
  m_cellArena.prepare(static_cast<uint2>(activeParticleIds.size()));
  m_connectivity.prepare(static_cast<uint2>(activeParticleIds.size()));
  if (!activeParticleIds.empty()) {
    const std::array<real_t, 3>& L(m_nbrList.getBox().getL());
    Cuboid<real_t> cub(L);
    parallelForPersistent(activeParticleIds.size(), [&](size_t i, size_t workerId) {
      CellMaker<real_t, Weighted>& maker = m_workers[workerId]->maker;
      m_workers[workerId]->arena.ensureVisitedSize(static_cast<uint2>(p.size()));
      if constexpr (Weighted)
        maker.setWeights(&this->m_weights);
      else
        maker.setWeights(NULL);
      maker.build(activeParticleIds[i], p, m_nbrList, cub);
      m_cellArena.insertFromMaker(static_cast<uint2>(i), maker);
      m_connectivity.overwriteFromMaker(static_cast<uint2>(i), maker);
    });
  }
  m_nbrList.clear();
  if (computeGeometry)
    buildGeometry(p);
  else
    clearGeometryCache();
  m_isBuild = true;
  if constexpr (Weighted)
    this->clearWeightDirty();
  rebuildBuiltParticleMaps(p.size());
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::buildWeighted(const std::vector<std::array<real_t, 3> >& p,
                                                  const std::vector<uint8_t>* enabledMask,
                                                  bool computeGeometry) {
  syncParticleActivity(p.size());
  std::vector<uint8_t> enabled(p.size(), 1u);
  if (enabledMask != NULL) {
    if (enabledMask->size() != p.size()) {
      std::fprintf(stderr,
                   "CellComplex::build: activity mask has %zu entries, expected %zu\n",
                   enabledMask->size(), p.size());
      std::abort();
    }
    enabled = *enabledMask;
    normalizeActivity(enabled);
  }

  const std::array<real_t, 3>& L(m_nbrList.getBox().getL());
  std::vector<uint2> enabledIds;
  collectActiveParticleIds(enabled, enabledIds);
  if (enabledIds.empty()) {
    m_particleActive.assign(p.size(), 0u);
    this->m_particleHasCell.assign(p.size(), 0u);
    this->m_cellParticleIds.clear();
    m_activeParticleIds.clear();
    m_cellArena.clear();
    m_connectivity.clear();
    clearGeometryCache();
    m_nbrList.setupSubset(p, enabledIds, real_t(1));
    m_nbrList.clear();
    m_isBuild = true;
    this->clearWeightDirty();
    rebuildBuiltParticleMaps(p.size());
    return;
  }

  real_t density = real_t(enabledIds.size()) / (L[0] * L[1] * L[2]);
  real_t rcut = real_t(1.75) * (std::pow(density, real_t(-1.0 / 3.0)));
  if (enabledIds.size() == p.size()) {
#ifdef VORONOI_USE_OPENMP
    if (omp_in_parallel())
      m_nbrList.setupCurrentTeam(p, rcut);
    else
      m_nbrList.setup(p, rcut);
#else
    m_nbrList.setup(p, rcut);
#endif
  } else {
    m_nbrList.setupSubset(p, enabledIds, rcut);
  }

  Cuboid<real_t> cub(L);
  std::vector<Cell<real_t> > builtCells(p.size());
  std::vector<uint8_t> nonEmpty(p.size(), 0u);
  parallelForPersistent(p.size(), [&](size_t i, size_t workerId) {
    if (enabled[i] == 0u)
      return;
    CellMaker<real_t, Weighted>& maker = m_workers[workerId]->maker;
    maker.setWeights(&this->m_weights);
    maker.build(static_cast<uint2>(i), p, m_nbrList, cub);
    maker.renumber();
    if (maker.numVertices() == 0 || maker.numFacets() == 0)
      return;
    bool hasNoNbr = false;
    for (uint1 facet = 0; facet < maker.numFacets(); ++facet) {
      if (maker.getNbrs()[facet] == noNbr) {
        hasNoNbr = true;
        break;
      }
    }
    if (hasNoNbr)
      return;
    builtCells[i] = maker;
    nonEmpty[i] = 1u;
  });
  m_nbrList.clear();

  std::vector<Cell<real_t> > packedCells;
  packedCells.reserve(enabledIds.size());
  m_particleActive = enabled;
  this->m_particleHasCell.assign(p.size(), 0u);
  for (size_t i = 0; i < p.size(); ++i) {
    if (enabled[i] == 0u || nonEmpty[i] == 0u)
      continue;
    this->m_particleHasCell[i] = 1u;
    packedCells.push_back(builtCells[i]);
  }

  m_cellArena.rebuildFromCells(packedCells);
  m_connectivity.resize(static_cast<uint2>(packedCells.size()));
  for (size_t i = 0; i < packedCells.size(); ++i) {
    std::vector<uint2> directNbrs;
    directNbrs.reserve(packedCells[i].numFacets());
    for (uint1 facet = 0; facet < packedCells[i].numFacets(); ++facet) {
      const uint2 nbrId = packedCells[i].getNbr(facet);
      if (nbrId != noNbr)
        directNbrs.push_back(nbrId);
    }
    m_connectivity.overwrite(static_cast<uint2>(i), directNbrs, std::vector<uint2>());
  }

  if (computeGeometry)
    buildGeometry(p);
  else
    clearGeometryCache();
  m_isBuild = true;
  this->clearWeightDirty();
  rebuildBuiltParticleMaps(p.size());
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::build(const std::vector<std::array<real_t, 3> >& p,
                                bool computeGeometry) {
  if constexpr (Weighted) {
    buildWeighted(p, NULL, computeGeometry);
    return;
  }
  buildFromCurrentActivity(p, computeGeometry);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::build(const std::vector<std::array<real_t, 3> >& p,
                                const std::vector<uint8_t>& active, bool computeGeometry) {
  if constexpr (Weighted) {
    buildWeighted(p, &active, computeGeometry);
    return;
  }
  setParticleActivity(active);
  if (active.size() != p.size()) {
    std::fprintf(stderr, "CellComplex::build: activity mask has %zu entries, expected %zu\n",
                 active.size(), p.size());
    std::abort();
  }
  buildFromCurrentActivity(p, computeGeometry);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::clearGeometryCache() {
  m_geom.clear();
  m_geometry.clear();
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::commitCellGeometry(uint2 cellId, const TopologyArena<real_t>& cellArena,
                          std::vector<CellGeometry<real_t> >& geomCache,
                          GeometryArena<real_t>& geometryArena,
                          const std::vector<std::array<real_t, 3> >& p) {
  CellGeometry<real_t> geom;
  geom = cellArena.getView(cellId);
  geom.computeConnectingVectors(p, m_nbrList.getBox());
  geom.computeEdgeInv();
  geom.diffVolume();
  geomCache[cellId] = geom;
  geometryArena.overwriteFromLegacy(cellId, cellArena.cellId(cellId), geom);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::commitCellGeometry(uint2 cellId,
                                             const std::vector<std::array<real_t, 3> >& p) {
  commitCellGeometry(cellId, m_cellArena, m_geom, m_geometry, p);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::rebuildLegacyGeometryCache(const std::vector<std::array<real_t, 3> >& p) {
  m_geom.resize(m_cellArena.numCells());
  m_geometry.prepare(static_cast<uint2>(m_cellArena.numCells()));
  parallelForPersistent(m_cellArena.numCells(),
                        [&](size_t i, size_t) { commitCellGeometry(static_cast<uint2>(i), p); });
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::buildGeometry(const std::vector<std::array<real_t, 3> >& p) {
  rebuildLegacyGeometryCache(p);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::update(const std::vector<std::array<real_t, 3> >& p) {
  m_lastUpdateStats = CellComplexUpdateStats();
  syncParticleActivity(p.size());
  std::vector<uint2> desiredActiveIds;
  collectActiveParticleIds(m_particleActive, desiredActiveIds);
  std::vector<uint2> currentCellParticleIds;
  if constexpr (Weighted)
    collectActiveParticleIds(this->m_particleHasCell, currentCellParticleIds);
  m_lastUpdateStats.num_cells =
      static_cast<uint2>(Weighted ? currentCellParticleIds.size() : desiredActiveIds.size());
  bool weightsDirtyNow = false;
  if constexpr (Weighted)
    weightsDirtyNow = this->weightsDirty();
  if (!m_isBuild) {
    m_lastUpdateStats.rebuilt_from_scratch = true;
    if constexpr (Weighted)
      buildWeighted(p, NULL, true);
    else
      buildFromCurrentActivity(p, true);
    return;
  }
  if (m_geom.size() != m_cellArena.numCells() || m_geometry.numCells() != m_cellArena.numCells())
    buildGeometry(p);
  if (desiredActiveIds.empty()) {
    m_cellArena.clear();
    m_connectivity.clear();
    clearGeometryCache();
    m_isBuild = true;
    rebuildBuiltParticleMaps(p.size());
    return;
  }
  if constexpr (Weighted) {
    if (m_cellArena.numCells() != currentCellParticleIds.size()) {
      m_lastUpdateStats.rebuilt_from_scratch = true;
      buildWeighted(p, NULL, true);
      return;
    }
  } else {
    if (!allParticlesActive() || m_cellArena.numCells() != desiredActiveIds.size()) {
      m_lastUpdateStats.rebuilt_from_scratch = true;
      buildFromCurrentActivity(p, true);
      return;
    }
  }

  initNbrList(p);
  const Box<real_t>& box(m_nbrList.getBox());
  const std::array<real_t, 3>& L(box.getL());
  Cuboid<real_t> cub(L);
  const size_t workerCount = std::max<size_t>(m_team.threadCount(), 1u);

  struct AsyncTask {
    uint2 particleId;
    uint2 depth;
  };
  enum CellTaskState : uint8_t {
    kTaskUnseen = 0u,
    kTaskQueued = 1u,
    kTaskProcessing = 2u,
    kTaskDone = 3u
  };

  std::vector<uint2> nonConvexByWorker(workerCount, 0u);

  auto collectDirectNbrs = [](const uint2* nbrs, uint1 numFacets, std::vector<uint2>& out,
                              bool& hasNoNbr) {
    out.clear();
    hasNoNbr = false;
    out.reserve(numFacets);
    for (uint1 facet = 0; facet < numFacets; ++facet) {
      const uint2 nbrId = nbrs[facet];
      if (nbrId == noNbr) {
        hasNoNbr = true;
        continue;
      }
      if (std::find(out.begin(), out.end(), nbrId) == out.end())
        out.push_back(nbrId);
    }
    std::sort(out.begin(), out.end());
  };

  parallelForPersistent(m_cellArena.numCells(), [&](size_t i, size_t workerId) {
    CellGeometry<real_t> geom = m_geom[i];
    geom.computeConnectingVectors(p, box);
    geom.computeEdgeInv();
    geom.updateVertexPos();
    if (!geom.isConvex()) {
      ++nonConvexByWorker[workerId];
      return;
    }

    geom.diffVolume();
    m_geom[i] = geom;
    m_geometry.overwriteFromLegacy(static_cast<uint2>(i), m_cellArena.cellId(i), geom);
  });

  for (size_t workerId = 0; workerId < workerCount; ++workerId)
    m_lastUpdateStats.num_non_convex_before += nonConvexByWorker[workerId];

  std::vector<std::atomic<uint8_t> > taskState(p.size());
  for (size_t i = 0; i < taskState.size(); ++i)
    taskState[i].store(kTaskUnseen, std::memory_order_relaxed);

  std::vector<std::atomic<uint8_t> > forceFullState(p.size());
  std::vector<std::atomic<uint8_t> > liveActiveState(p.size());
  std::vector<std::atomic<uint8_t> > liveHasCellState(p.size());
  std::unique_ptr<std::atomic<uint8_t>[]> reactivationState;
  std::vector<std::atomic<uint8_t> > insertedState(p.size());
  std::vector<Cell<real_t> > insertedCells(p.size());
  const std::vector<uint2> oldCellIndexByParticle = m_cellIndexByParticle;
  std::vector<uint8_t> oldHasCell;
  if constexpr (Weighted)
    oldHasCell = this->m_particleHasCell;
  for (size_t i = 0; i < p.size(); ++i) {
    forceFullState[i].store(0u, std::memory_order_relaxed);
    liveActiveState[i].store(m_particleActive[i], std::memory_order_relaxed);
    uint8_t hasCell = m_particleActive[i];
    if constexpr (Weighted)
      hasCell = this->m_particleHasCell[i];
    liveHasCellState[i].store(hasCell, std::memory_order_relaxed);
    insertedState[i].store(0u, std::memory_order_relaxed);
  }
  if constexpr (Weighted) {
    reactivationState.reset(new std::atomic<uint8_t>[p.size()]);
    for (size_t i = 0; i < p.size(); ++i)
      reactivationState[i].store(0u, std::memory_order_relaxed);
  }

  size_t initialQueued = 0;
  std::vector<std::vector<AsyncTask> > workQueues(workerCount);
  std::vector<std::mutex> queueMutex(workerCount);
  if constexpr (Weighted) {
    if (weightsDirtyNow) {
      for (size_t i = 0; i < desiredActiveIds.size(); ++i) {
        const uint2 particleId = desiredActiveIds[i];
        taskState[particleId].store(kTaskQueued, std::memory_order_relaxed);
        forceFullState[particleId].store(1u, std::memory_order_relaxed);
        workQueues[i % workerCount].push_back(AsyncTask{particleId, 1u});
        ++initialQueued;
      }
    }
  }
  if (initialQueued == 0) {
    for (size_t i = 0; i < m_geom.size(); ++i) {
      CellGeometry<real_t> geom = m_geom[i];
      geom.computeConnectingVectors(p, box);
      geom.computeEdgeInv();
      geom.updateVertexPos();
      if (geom.isConvex())
        continue;
      const uint2 particleId =
          (Weighted ? m_cellArena.cellId(static_cast<uint2>(i)) : static_cast<uint2>(i));
      taskState[particleId].store(kTaskQueued, std::memory_order_relaxed);
      workQueues[i % workerCount].push_back(AsyncTask{particleId, 1u});
      ++initialQueued;
    }
  }
  m_lastUpdateStats.num_rebuild_candidates = static_cast<uint2>(initialQueued);
  if (initialQueued == 0)
    return;

  std::atomic<size_t> outstanding(initialQueued);
  std::vector<uint2> queuedByWorker(workerCount, 0u);
  std::vector<uint2> localByWorker(workerCount, 0u);
  std::vector<uint2> fullByWorker(workerCount, 0u);
  std::vector<uint2> emptyAfterLocalByWorker(workerCount, 0u);
  std::vector<uint2> changedByWorker(workerCount, 0u);
  std::vector<uint2> proposalByWorker(workerCount, 0u);
  std::vector<uint2> processedByWorker(workerCount, 0u);
  std::vector<uint2> maxDepthByWorker(workerCount, 0u);

  auto tryQueueCandidate = [&](uint2 particleId, uint2 depth, size_t ownerTid,
                               bool forceFull) -> bool {
    if (particleId == noNbr || particleId >= taskState.size())
      return false;
    if (forceFull)
      forceFullState[particleId].store(1u, std::memory_order_release);
    uint8_t expected = kTaskUnseen;
    if (!taskState[particleId].compare_exchange_strong(expected, kTaskQueued,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_relaxed)) {
      if (!forceFull || expected != kTaskDone)
        return false;
      expected = kTaskDone;
      if (!taskState[particleId].compare_exchange_strong(expected, kTaskQueued,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_relaxed))
        return false;
    }

    outstanding.fetch_add(1u, std::memory_order_acq_rel);
    ++queuedByWorker[ownerTid];
    {
      std::lock_guard<std::mutex> lock(queueMutex[ownerTid]);
      workQueues[ownerTid].push_back(AsyncTask{particleId, depth});
    }
    return true;
  };

  auto tryReactivateNearbyInactive = [&](uint2 particleId, CellMaker<real_t, Weighted>& maker,
                                         size_t ownerTid, uint2 depth) {
    if constexpr (!Weighted)
      return false;

    std::vector<uint2> nbrGC;
    m_nbrList.getGridNbrs(p[particleId], nbrGC);
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin;
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator end;
    for (size_t i = 0; i < nbrGC.size(); ++i) {
      m_nbrList.getCellContent(nbrGC[i], begin, end);
      for (typename std::vector<PosAndId<uint2, real_t> >::const_iterator itr(begin); itr != end;
           ++itr) {
        const uint2 candidateId = itr->id;
        if (candidateId == particleId || candidateId >= liveActiveState.size() ||
            liveActiveState[candidateId].load(std::memory_order_acquire) == 0u ||
            liveHasCellState[candidateId].load(std::memory_order_acquire) != 0u)
          continue;
        uint8_t expected = 0u;
        if (!reactivationState[candidateId].compare_exchange_strong(expected, 1u,
                                                                    std::memory_order_acq_rel,
                                                                    std::memory_order_relaxed))
          continue;
        maker.build(candidateId, p, m_nbrList, cub);
        maker.renumber();
        bool hasNoNbr = false;
        for (uint1 facet = 0; facet < maker.numFacets(); ++facet) {
          if (maker.getNbrs()[facet] == noNbr) {
            hasNoNbr = true;
            break;
          }
        }
        if (maker.numVertices() == 0 || maker.numFacets() == 0 || hasNoNbr)
          continue;
        insertedCells[candidateId] = maker;
        insertedState[candidateId].store(1u, std::memory_order_release);
        liveHasCellState[candidateId].store(1u, std::memory_order_release);
        const uint2 nextDepth = depth + 1u;
        for (uint1 facet = 0; facet < maker.numFacets(); ++facet)
          tryQueueCandidate(maker.getNbrs()[facet], nextDepth, ownerTid, true);
        proposalByWorker[ownerTid] += maker.numFacets();
        return true;
      }
    }
    return false;
  };

  auto popLocalTask = [&](size_t tid, AsyncTask& task) -> bool {
    std::lock_guard<std::mutex> lock(queueMutex[tid]);
    if (workQueues[tid].empty())
      return false;
    task = workQueues[tid].back();
    workQueues[tid].pop_back();
    return true;
  };

  auto stealTask = [&](size_t tid, AsyncTask& task) -> bool {
    for (size_t offset = 1; offset < workerCount; ++offset) {
      const size_t src = (tid + offset) % workerCount;
      std::lock_guard<std::mutex> lock(queueMutex[src]);
      if (workQueues[src].empty())
        continue;
      task = workQueues[src].back();
      workQueues[src].pop_back();
      return true;
    }
    return false;
  };

  m_team.run([&](size_t tid, size_t /*teamSize*/) {
    AsyncTask task{0u, 0u};
    while (true) {
      if (!popLocalTask(tid, task) && !stealTask(tid, task)) {
        if (outstanding.load(std::memory_order_acquire) == 0u)
          break;
        std::this_thread::yield();
        continue;
      }

      uint8_t expected = kTaskQueued;
      if (!taskState[task.particleId].compare_exchange_strong(
              expected, kTaskProcessing, std::memory_order_acq_rel, std::memory_order_relaxed))
        continue;

      ++processedByWorker[tid];
      if (task.depth > maxDepthByWorker[tid])
        maxDepthByWorker[tid] = task.depth;

      WorkerContext& worker = *m_workers[tid];
      CellMaker<real_t, Weighted>& maker = worker.maker;
      if constexpr (Weighted)
        maker.setWeights(&this->m_weights);
      else
        maker.setWeights(NULL);
      const uint2 particleId = task.particleId;
      const bool forceFull =
          (forceFullState[particleId].exchange(0u, std::memory_order_acq_rel) != 0u);
      if constexpr (Weighted) {
        if (liveActiveState[particleId].load(std::memory_order_acquire) == 0u) {
          taskState[particleId].store(kTaskDone, std::memory_order_release);
          outstanding.fetch_sub(1u, std::memory_order_acq_rel);
          continue;
        }
      }
      const uint2 cellId =
          (Weighted ? m_cellIndexByParticle[particleId] : static_cast<uint2>(particleId));
      std::vector<uint2> oldDirect;
      CellGeometry<real_t> geom;
      bool isConvex = false;
      const Cell<real_t>* pBaseCell = NULL;
      if constexpr (Weighted) {
        if (cellId != noNbr) {
          geom = m_geom[cellId];
          geom.computeConnectingVectors(p, box);
          geom.computeEdgeInv();
          geom.updateVertexPos();
          isConvex = geom.isConvex();
          pBaseCell = &geom.getCell();

          const ConnectivityView<real_t> connView = m_connectivity.getView(cellId);
          oldDirect.reserve(connView.numDirectNbrs());
          for (uint1 facet = 0; facet < connView.numDirectNbrs(); ++facet)
            oldDirect.push_back(connView.getDirectNbr(facet));
          std::sort(oldDirect.begin(), oldDirect.end());
        }
      } else {
        geom = m_geom[cellId];
        geom.computeConnectingVectors(p, box);
        geom.computeEdgeInv();
        geom.updateVertexPos();
        isConvex = geom.isConvex();
        pBaseCell = &geom.getCell();

        const ConnectivityView<real_t> connView = m_connectivity.getView(cellId);
        oldDirect.reserve(connView.numDirectNbrs());
        for (uint1 facet = 0; facet < connView.numDirectNbrs(); ++facet)
          oldDirect.push_back(connView.getDirectNbr(facet));
        std::sort(oldDirect.begin(), oldDirect.end());
      }

      std::vector<uint2> newDirect;
      bool hasNoNbr = false;
      bool usedLocal = false;
      bool localCutChanged = false;

      if (!forceFull && isConvex) {
        usedLocal = true;
        ++localByWorker[tid];
        localCutChanged = maker.build(particleId, p, m_nbrList, *pBaseCell, oldDirect);
        maker.renumber();
        collectDirectNbrs(maker.getNbrs(), maker.numFacets(), newDirect, hasNoNbr);

        if (!localCutChanged && newDirect == oldDirect && !hasNoNbr) {
          geom.diffVolume();
          m_geom[cellId] = geom;
          m_geometry.overwriteFromLegacy(cellId, m_cellArena.cellId(cellId), geom);
          m_connectivity.overwrite(cellId, oldDirect, std::vector<uint2>());
          taskState[particleId].store(kTaskDone, std::memory_order_release);
          outstanding.fetch_sub(1u, std::memory_order_acq_rel);
          continue;
        }

        bool keepLocal = !(maker.numVertices() == 0 || maker.numFacets() == 0 || hasNoNbr);
        if (keepLocal) {
          Cell<real_t> checkedCell;
          checkedCell = maker;
          CellGeometry<real_t> checkedGeom(checkedCell);
          checkedGeom.computeConnectingVectors(p, box);
          checkedGeom.computeEdgeInv();
          checkedGeom.updateVertexPos();
          checkedGeom.computeVolume();
          keepLocal = checkedGeom.isConvex() && (checkedGeom.getVolume() > real_t(0));
        }
        if (!keepLocal) {
          ++emptyAfterLocalByWorker[tid];
          usedLocal = false;
        }
      }

      if (!isConvex || !usedLocal) {
        maker.build(particleId, p, m_nbrList, cub);
        maker.renumber();
        collectDirectNbrs(maker.getNbrs(), maker.numFacets(), newDirect, hasNoNbr);
        ++fullByWorker[tid];
      }

      if constexpr (Weighted) {
        if (maker.numVertices() == 0 || maker.numFacets() == 0 || hasNoNbr) {
          liveHasCellState[particleId].store(0u, std::memory_order_release);
          const uint2 nextDepth = task.depth + 1u;
          for (size_t i = 0; i < oldDirect.size(); ++i)
            tryQueueCandidate(oldDirect[i], nextDepth, tid, true);
          taskState[particleId].store(kTaskDone, std::memory_order_release);
          outstanding.fetch_sub(1u, std::memory_order_acq_rel);
          continue;
        }
      }

      if (!usedLocal && !forceFull && isConvex && newDirect == oldDirect && !hasNoNbr) {
        geom.diffVolume();
        m_geom[cellId] = geom;
        m_geometry.overwriteFromLegacy(cellId, m_cellArena.cellId(cellId), geom);
        m_connectivity.overwrite(cellId, oldDirect, std::vector<uint2>());
        taskState[particleId].store(kTaskDone, std::memory_order_release);
        outstanding.fetch_sub(1u, std::memory_order_acq_rel);
        continue;
      }

      const bool topologyChanged = hasNoNbr || (newDirect != oldDirect);
      if (topologyChanged)
        ++changedByWorker[tid];

      bool insertedPackedCell = false;
      if constexpr (Weighted) {
        if (cellId == noNbr) {
          insertedCells[particleId] = maker;
          insertedState[particleId].store(1u, std::memory_order_release);
          liveHasCellState[particleId].store(1u, std::memory_order_release);
          insertedPackedCell = true;
        }
      }
      if (!insertedPackedCell) {
        m_cellArena.overwriteFromMaker(cellId, maker);
        m_connectivity.overwrite(cellId, newDirect, std::vector<uint2>());
        commitCellGeometry(cellId, p);
      }

      if constexpr (Weighted) {
        if (topologyChanged)
          tryReactivateNearbyInactive(particleId, maker, tid, task.depth);
      }

      if (task.depth == 1u || topologyChanged) {
        proposalByWorker[tid] += static_cast<uint2>(oldDirect.size() + newDirect.size());
        const uint2 nextDepth = task.depth + 1u;
        for (size_t i = 0; i < oldDirect.size(); ++i)
          tryQueueCandidate(oldDirect[i], nextDepth, tid, (Weighted && topologyChanged));
        for (size_t i = 0; i < newDirect.size(); ++i)
          tryQueueCandidate(newDirect[i], nextDepth, tid, (Weighted && topologyChanged));
      }

      taskState[particleId].store(kTaskDone, std::memory_order_release);
      outstanding.fetch_sub(1u, std::memory_order_acq_rel);
    }
  });

  uint2 maxDepth = 0u;
  for (size_t workerId = 0; workerId < workerCount; ++workerId) {
    m_lastUpdateStats.num_rebuild_candidates += queuedByWorker[workerId];
    m_lastUpdateStats.num_local_rebuild_cells += localByWorker[workerId];
    m_lastUpdateStats.num_full_rebuild_cells += fullByWorker[workerId];
    m_lastUpdateStats.num_empty_after_local_rebuild += emptyAfterLocalByWorker[workerId];
    m_lastUpdateStats.num_repair_cells_changed_total += changedByWorker[workerId];
    m_lastUpdateStats.num_repair_proposals_total += proposalByWorker[workerId];
    m_lastUpdateStats.num_repair_target_groups_total += processedByWorker[workerId];
    if (maxDepthByWorker[workerId] > maxDepth)
      maxDepth = maxDepthByWorker[workerId];
  }
  m_lastUpdateStats.num_repair_iterations = maxDepth;
  if constexpr (Weighted) {
    bool activityChanged = false;
    bool cellMembershipChanged = false;
    for (size_t i = 0; i < p.size(); ++i) {
      const uint8_t nextActive = liveActiveState[i].load(std::memory_order_acquire);
      const uint8_t nextHasCell = liveHasCellState[i].load(std::memory_order_acquire);
      if (nextActive != m_particleActive[i])
        activityChanged = true;
      if (nextHasCell != this->m_particleHasCell[i])
        cellMembershipChanged = true;
      m_particleActive[i] = nextActive;
      this->m_particleHasCell[i] = nextHasCell;
    }
    if (activityChanged || cellMembershipChanged) {
      std::vector<Cell<real_t> > packedCells;
      packedCells.reserve(desiredActiveIds.size() + 8u);
      bool repackFailed = false;
      for (size_t particleId = 0; particleId < p.size(); ++particleId) {
        if (m_particleActive[particleId] == 0u || this->m_particleHasCell[particleId] == 0u)
          continue;
        Cell<real_t> cell;
        if (insertedState[particleId].load(std::memory_order_acquire) != 0u) {
          cell = insertedCells[particleId];
        } else if (particleId < oldHasCell.size() && oldHasCell[particleId] != 0u &&
                   oldCellIndexByParticle[particleId] != noNbr) {
          m_cellArena.materializeCell(oldCellIndexByParticle[particleId], cell);
        } else {
          repackFailed = true;
          break;
        }
        bool invalid = false;
        for (uint1 facet = 0; facet < cell.numFacets(); ++facet) {
          const uint2 nbrId = cell.getNbr(facet);
          if (nbrId == noNbr ||
              (nbrId < this->m_particleHasCell.size() && this->m_particleHasCell[nbrId] == 0u)) {
            invalid = true;
            break;
          }
        }
        if (invalid) {
          repackFailed = true;
          break;
        }
        packedCells.push_back(cell);
      }
      if (repackFailed) {
        m_lastUpdateStats.rebuilt_from_scratch = true;
        buildWeighted(p, NULL, true);
        return;
      }
      std::sort(packedCells.begin(), packedCells.end(),
                [](const Cell<real_t>& a, const Cell<real_t>& b) { return a.getID() < b.getID(); });
      m_cellArena.rebuildFromCells(packedCells);
      m_connectivity.resize(static_cast<uint2>(packedCells.size()));
      for (size_t i = 0; i < packedCells.size(); ++i) {
        std::vector<uint2> directNbrs;
        directNbrs.reserve(packedCells[i].numFacets());
        for (uint1 facet = 0; facet < packedCells[i].numFacets(); ++facet) {
          const uint2 nbrId = packedCells[i].getNbr(facet);
          if (nbrId != noNbr && nbrId < this->m_particleHasCell.size() &&
              this->m_particleHasCell[nbrId] != 0u)
            directNbrs.push_back(nbrId);
        }
        m_connectivity.overwrite(static_cast<uint2>(i), directNbrs, std::vector<uint2>());
      }
      buildGeometry(p);
      rebuildBuiltParticleMaps(p.size());
    } else {
      rebuildBuiltParticleMaps(p.size());
    }
    m_lastUpdateStats.num_cells = static_cast<uint2>(this->m_cellParticleIds.size());
    this->clearWeightDirty();
  }
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::update(const std::vector<std::array<real_t, 3> >& p,
                                 const std::vector<uint8_t>& active) {
  if (active.size() != p.size()) {
    std::fprintf(stderr, "CellComplex::update: activity mask has %zu entries, expected %zu\n",
                 active.size(), p.size());
    std::abort();
  }
  setParticleActivity(active);
  update(p);
}

template <typename real_t, bool Weighted>
void CellComplex<real_t, Weighted>::drawInterfaceGnuplot(uint0 iType, uint0 jType,
                                               const std::vector<std::array<real_t, 3> >& pos,
                                               FILE* fp) const {
#pragma omp parallel for
  for (size_t i = 0; i < m_cellArena.numCells(); ++i) {
    const uint2 particleId = m_cellArena.cellId(i);
    if (particleId < m_types.size() && m_types[particleId] == iType) {
      Cell<real_t> cell;
      cell = m_cellArena.getView(i);
      for (uint1 j = 0; j < cell.numFacets(); ++j) {
        const uint2 nbrId = cell.getNbr(j);
        if (nbrId < m_types.size() && m_types[nbrId] == jType) {
          cell.drawFacetGnuplot(j, pos[particleId], fp);
          fputs("\n\n", fp);
        }
      }
    }
  }
}

template <typename real_t>
void NbrsToFacets::init(const std::vector<Cell<real_t> >& cells) {
  m_numCells = cells.size();
  m_ptr.resize(m_numCells + 1);
  m_ptr[0] = 0;
  for (uint2 i = 0; i < m_numCells; ++i)
    m_ptr[i + 1] = m_ptr[i] + cells[i].numFacets();
  m_nbr.resize(m_ptr[m_numCells]);
  //    printf("total number of facets: %u\n", m_ptr[m_numCells]);
  m_facet.resize(m_ptr[m_numCells]);
#pragma omp parallel
  {
    std::vector<std::pair<uint2, uint1> > nbrLoc;
#pragma omp for
    for (uint2 i = 0; i < m_numCells; ++i) {
      uint2 numFacets(m_ptr[i + 1] - m_ptr[i]);
      nbrLoc.resize(numFacets);
      for (uint1 j = 0; j < numFacets; ++j) {
        nbrLoc[j].first = cells[i].getNbr(j);
        nbrLoc[j].second = j;
      }
      std::sort(nbrLoc.begin(), nbrLoc.end(), ComparePairFirst());
      for (uint1 j = 0; j < numFacets; ++j) {
        m_nbr[m_ptr[i] + j] = nbrLoc[j].first;
        m_facet[m_ptr[i] + j] = nbrLoc[j].second;
      }
    }
  }
  // for(uint2 i=0; i < m_numCells; ++i){
  //   printf("cell: %u, begin: %u, end: %u", i, m_ptr[i], m_ptr[i+1]);
  //   for(uint1 j(m_ptr[i]); j< m_ptr[i+1]; ++j)
  // 	printf(", nbr: %u, facet: %u", m_nbr[j], m_facet[j]);
  //   printf("\n");
  // }
}

template <typename real_t>
void NbrsToFacets::init(const CellArena<real_t>& arena) {
  m_numCells = static_cast<uint2>(arena.numCells());
  m_ptr.resize(m_numCells + 1);
  m_ptr[0] = 0;
  for (uint2 i = 0; i < m_numCells; ++i)
    m_ptr[i + 1] = m_ptr[i] + arena.cellNumFacets(i);
  m_nbr.resize(m_ptr[m_numCells]);
  m_facet.resize(m_ptr[m_numCells]);

#pragma omp parallel
  {
    std::vector<std::pair<uint2, uint1> > nbrLoc;
#pragma omp for
    for (uint2 i = 0; i < m_numCells; ++i) {
      uint2 numFacets(m_ptr[i + 1] - m_ptr[i]);
      nbrLoc.resize(numFacets);
      const uint2* nbrData = arena.cellNbrData(i);
      for (uint1 j = 0; j < numFacets; ++j) {
        nbrLoc[j].first = nbrData[j];
        nbrLoc[j].second = j;
      }
      std::sort(nbrLoc.begin(), nbrLoc.end(), ComparePairFirst());
      for (uint1 j = 0; j < numFacets; ++j) {
        m_nbr[m_ptr[i] + j] = nbrLoc[j].first;
        m_facet[m_ptr[i] + j] = nbrLoc[j].second;
      }
    }
  }
}

void NbrsToFacets::print() const {
  for (uint2 i = 0; i < m_numCells; ++i) {
    printf("cell: %u, begin: %u, end: %u", i, m_ptr[i], m_ptr[i + 1]);
    for (uint2 j = m_ptr[i]; j < m_ptr[i + 1]; ++j)
      printf(", nbr: %u, facet: %u", m_nbr[j], m_facet[j]);
    printf("\n");
  }
}

template <typename real_t>
NbrsToFacets NbrsToFacets::transposedV(const std::vector<real_t>& values,
                                       std::vector<real_t>& valuesTr) const {
  NbrsToFacets tr;
  tr.m_numCells = m_numCells;
  tr.m_ptr.resize(m_numCells + 1, 0);
#pragma omp parallel for
  for (uint2 i = 0; i < m_nbr.size(); ++i) {
    if (m_nbr[i] < m_numCells) {
      uint2 indx(m_nbr[i] + 1);
#pragma omp atomic
      ++(tr.m_ptr[indx]);
    }
  }
  for (uint2 i = 0; i < m_numCells; ++i) {
    tr.m_ptr[i + 1] += tr.m_ptr[i];
  }
  std::vector<uint2> ptrTmp(tr.m_ptr.size());
  std::vector<std::pair<uint2, std::pair<uint1, std::array<real_t, 3> > > > nbrTmp(m_nbr.size());
#pragma omp parallel for
  for (uint2 i = 0; i < tr.m_ptr.size(); ++i)
    ptrTmp[i] = tr.m_ptr[i];
#pragma omp parallel for
  for (uint2 i = 0; i < m_numCells; ++i)
    for (uint1 j(m_ptr[i]); j < m_ptr[i + 1]; ++j) {
      if (m_nbr[j] < m_numCells) {
        uint2 ptrLoc;
#pragma omp atomic capture
        ptrLoc = ptrTmp[m_nbr[j]]++;
        nbrTmp[ptrLoc].first = i;
        nbrTmp[ptrLoc].second.first = m_facet[j];
        nbrTmp[ptrLoc].second.second = values[j];
      }
    }
#pragma omp parallel for
  for (uint2 i = 0; i < tr.m_numCells; ++i)
    std::sort(nbrTmp.begin() + tr.m_ptr[i], nbrTmp.begin() + tr.m_ptr[i + 1], ComparePairFirst());
  tr.m_nbr.resize(nbrTmp.size());
  tr.m_facet.resize(nbrTmp.size());
  valuesTr.resize(nbrTmp.size());
#pragma omp parallel for
  for (size_t i = 0; i < nbrTmp.size(); ++i) {
    tr.m_nbr[i] = nbrTmp[i].first;
    tr.m_facet[i] = nbrTmp[i].second.first;
    valuesTr[i] = nbrTmp[i].second.second;
  }
  return tr;
}

//   template<typename real_t>
//   void NbrsToFacets::makeMatrixdVdV(const std::vector<CellGeometry<real_t> > & geoms, const
//   std::vector<real_t> & masses)
//   {
//     std::vector< std::array<real_t, 3> > values(m_nbr.size());
// #pragma omp parallel for
//     for(uint2 i=0; i< m_ptr.size()-1; ++i){
//       const std::vector< std::array<real_t, 3> > & dV(geoms[i].getdV());
//       for(uint2 j=m_ptr[i]; j < m_ptr[i+1]; ++j)
// 	for(uint0 k(0); k<3; ++k)
// 	  values[j][k] = dV[m_facet[j]][k];
//     }
//     std::vector< std::array<real_t, 3> > valuesTr();
//     NbrsToFacets tr(this->transposedV(values, valuesTr));

//     std::vector<CoordMatrix<real_t> > cMat;
//     cMat.reserve(m_numCells*300);
//     for(uint2 i(0); i< m_numCells; ++i){
//       const std::vector<Array, 3> & dV1(geom[i].getdV());
//       for(uint2 j(m_ptr[i]); j< m_ptr[i+1]; ++j){
// 	uint2 partIndx(m_nbr[j]);
// 	for(uint2 m(tr.m_ptr[partIndx]); m < tr.m_ptr[partIndx+1]; ++m){
// 	  coord.push_back();
// 	}

//       }

//     }
//   }

}  // namespace vor
