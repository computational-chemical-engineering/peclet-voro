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
 *  - CellUpdater   – topology-repair after particle movement
 *  - CellComplex   – collection of cells and high-level build/update interface
 *  - NbrsToFacets  – neighbour-to-facet mapping for sparse-matrix assembly
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <boost/container/flat_set.hpp>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

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
      : peak_vertices_seen(0),
        peak_facets_seen(0),
        peak_vertex_capacity(0),
        peak_facet_capacity(0),
        vertex_overflow_events(0),
        facet_overflow_events(0),
        vertex_growth_events(0),
        facet_growth_events(0) {}
};

inline CellMakerTelemetry &cellMakerTelemetry() {
  static CellMakerTelemetry telemetry;
  return telemetry;
}

inline void resetCellMakerTelemetry() {
  auto &t = cellMakerTelemetry();
  t.peak_vertices_seen.store(0, std::memory_order_relaxed);
  t.peak_facets_seen.store(0, std::memory_order_relaxed);
  t.peak_vertex_capacity.store(0, std::memory_order_relaxed);
  t.peak_facet_capacity.store(0, std::memory_order_relaxed);
  t.vertex_overflow_events.store(0, std::memory_order_relaxed);
  t.facet_overflow_events.store(0, std::memory_order_relaxed);
  t.vertex_growth_events.store(0, std::memory_order_relaxed);
  t.facet_growth_events.store(0, std::memory_order_relaxed);
}

inline void updatePeakCounter(std::atomic<uint64_t> &counter, uint64_t candidate) {
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

template <typename real_t>
class CellMaker;
template <typename real_t>
class CellUpdater;
template <typename real_t>
class CellGeometry;
template <typename real_t>
class CellArena;

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
  Cell(const Cell<real_t> &cell);
  //! destructor.
  ~Cell() = default;
  //! @brief copy operator
  //! @param rhs of type Cell
  //! @return reference to the copied cell
  Cell &operator=(const Cell<real_t> &rhs);
  //! @brief copy operator
  //! @param rhs of type CellMaker (\sa CellMaker)
  //! @return reference to the copied cell
  Cell &operator=(CellMaker<real_t> &rhs);
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
  void printNbrFacets(const std::vector<Cell<real_t> > &cells) const;
  //! @brief output the cell geometry in a Gnuplot format
  //! @param p coordinate of the center of the cell (Note that internal vertex coordinates are
  //! relative to the center.)
  void drawGnuplot(Array<real_t, 3> p, FILE *fp) const;
  //! @brief output a facet in a Gnuplot format
  //! @param iFacet index of the facet to be drawn
  //! @param p coordinate of the center of the cell (Note that internal vertex coordinates are
  //! relative to the center.)
  inline void drawFacetGnuplot(uint1 iFacet, Array<real_t, 3> p, FILE *fp) const;
  //! @brief facet information of a cell with all the verticies on it
  void printFacetInfo(Array<real_t, 3> p, uint facet_id) const;
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
  inline const uint2 *getNbrs() const { return m_nbr; }
  //! @brief check of the cell has a facet that does not correspond to a neighbor cell
  //! Not every facet need necesarrily have an neighbor cell associated to it.
  //! @return true, if there are 1 or more facets without neighbors in a cell. false otherwise
  inline bool hasNoNbr();
  friend class CellMaker<real_t>;
  friend class CellUpdater<real_t>;
  friend class CellGeometry<real_t>;
  friend class CellArena<real_t>;

 protected:
  uint2 m_id;
  uint0 m_numFacets;
  uint0 m_numVertices;
  Array<real_t, 3> m_vertexPos[maxNumVertices];
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
  Cuboid(const Array<real_t, 3> &L);
};

/**
 * @class CellMaker
 * @brief class for making a cell using planar cuts
 * @tparam real_t real type used for floating point numbers (e.g. real or double)
 */
template <typename real_t>
class CellMaker {
 public:
  //! @brief constructot
  CellMaker();
  //! @brief destructor
  ~CellMaker();
  //! @brief initialize a cellmaker by equating it to a cell
  //! @param rhs cell used to initialize the cellmaker
  CellMaker &operator=(const Cell<real_t> &rhs);
  //! @brief initialize a cellmaker by equating it to a cellUpdatet
  //! @param rhs cellupdater used to initialize the cellmaker
  CellMaker &operator=(const CellUpdater<real_t> &rhs);
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
  bool build(uint2 id, const std::vector<Array<real_t, 3> > &pos,
             const NbrList<uint2, real_t> &nbrList, const Cell<real_t> &initCell);
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
  bool rebuild(const std::vector<Array<real_t, 3> > &pos, const Box<real_t> &box,
               const Cell<real_t> &initCell);
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
                          const Array<real_t, 3> pos0, const Box<real_t> &box);
  //    inline real_t getRsqMax() const {return m_rSq[m_vRsqMax];}
  void getCloseNbrs(NbrInsert &nbrs);
  // void drawGnuplot(FILE *fp) const;
  // void testTopo() const;
  const uint2 *getNbrs() const { return m_nbr.data(); }
  friend class Cell<real_t>;
  friend class CellUpdater<real_t>;
  void renumber();

 protected:
  //! @brief initialize the cell to be cut
  //! @param cell used for the initialization
  void init(const Cell<real_t> &cell);
  inline bool cutCell(const Array<real_t, 3> p, real_t rSqHalf, uint2 nbr);
  inline bool cutCell2(const Array<real_t, 3> p, real_t rSqHalf, uint2 nbr);
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
  inline real_t computeDist(uint1 i, const Array<real_t, 3> p, const real_t rSqHalf);
  //! @brief compute distance between the vertex i and a plane
  //! @param i index of the vertex
  //! @param p the poisition of a point relative to the cell center. The plane is the perpendicular
  //! plane halfway between the center and p.
  inline real_t computeDist(uint1 i, const Array<real_t, 3> p);
  //! @brief compute distance between a plane and all vertices
  //! @param p the poisition of a point relative to the cell center. The plane is the perpendicular
  //! plane halfway between the center and p.
  //! @param rSqHalf = 0.5|p|^2
  void computeAllDist(const Array<real_t, 3> p, const real_t rSqHalf);
  //! @brief reset internal variables that indicate distances of vertices to a plane are computed
  inline void resetDist();

  inline void computeGCOrig(uint2 indx, const Array<real_t, 3> &pos);
  real_t computeRsqMinGC() const;
  Array<real_t, 3> getClosestPointGC(uint1 indx) const;
  inline real_t computeDistGC(uint1 i);
  inline real_t computeMaxDistGC();
  real_t computeMaxDistGCVerb();
  void getAllDistGCVerb();
  void computeAllDistGC();
  uint1 m_numVertices, m_numFacets;  // only to be used after renumber()!
 private:
  CellMaker(const CellMaker<real_t> &rhs);
  CellMaker &operator=(const CellMaker<real_t> &rhs);
  inline uint1 getNextLabelCCW(uint1 label) const;
  inline uint1 getReverseLabel(uint1 label) const;
  inline bool applyCut(const Array<real_t, 3> &p, real_t rSqHalf, uint2 nbr);
  inline void ensureVertexBuffers(uint1 minSize);
  inline void ensureFacetBuffers(uint1 minSize);
  inline uint1 allocVertexChecked(const char *caller);
  inline uint1 allocFacetChecked(const char *caller);
  [[noreturn]] inline void failCapacity(const char *caller, const char *kind, uint1 capacity) const;
  static constexpr uint1 kMaxV = maxNumVertices - 1;
  static constexpr uint1 kMaxF = maxNumFacets - 1;
#ifndef VOR_CELLMAKER_USE_CUTCELL2
#define VOR_CELLMAKER_USE_CUTCELL2 1
#endif
  static constexpr bool kUseCutCell2 = (VOR_CELLMAKER_USE_CUTCELL2 != 0);
  static constexpr uint1 kInitialV = 128;
  static constexpr uint1 kInitialF = 64;
  uint2 m_id;
  const NbrList<uint2, real_t> *p_nbrList;
  std::vector<Array<real_t, 3> > m_vertexPos;
  std::vector<real_t> m_rSq;
  std::vector<Vertex> m_vertices;
  std::vector<uint1> m_facets;
  std::vector<uint2> m_nbr;
  DenseSlots<uint1, kMaxV> m_slotsV;
  DenseSlots<uint1, kMaxF> m_slotsF;
  VisitedIndx<uint2> m_visited;
  std::vector<uint2> m_checkGridCell;  ///< BFS queue (use m_checkGCHead as front index)
  size_t m_checkGCHead;                ///< index of current BFS queue front
  real_t m_distMax, m_distGCMax;
  uint1 m_vRsqMax;
  bool m_isAllCut;
  uint1 m_vDistMax, m_vDistGCMax;
  std::vector<real_t> m_dist;
  uint2 m_distGen;              ///< generation counter for lazy distance reset
  std::vector<uint2> m_knownDistGen;  ///< per-vertex generation stamp
  Array<real_t, 3> m_relOrigGC, m_dLGC;
  std::vector<real_t> m_distGC;
  std::vector<uint1> m_renumVWrk;
  std::vector<uint1> m_renumFWrk;
  std::vector<uint1> m_newVerticesWrk;
  std::vector<uint1> m_facetPrevWrk;
  std::vector<PosAndId<uint2, real_t> > m_nbrsWrk;
  std::vector<NbrDist<real_t> > m_nbrDistWrk;
  std::vector<uint1> m_vStackWrk;  ///< DFS stack (push_back/back/pop_back)
};

template <typename real_t>
class CellGeometry {
 public:
  CellGeometry();
  CellGeometry(Cell<real_t> &cell);
  CellGeometry &operator=(Cell<real_t> &rhs);
  CellGeometry &operator=(const CellGeometry<real_t> &rhs);
  void computeConnectingVectors(const std::vector<Array<real_t, 3> > &pos, const Box<real_t> &box);
  void computeEdgeInv();
  void updateVertexPos();
  void computeAreas();
  void computeVolume();
  void diffVolume();
  void computeAll();
  real_t maxConvexViolation() const;
  Array<Array<real_t, 3>, 3> velocityGradient(const std::vector<Array<real_t, 3> > &velocity) const;
  void getDelaunayNbrs(uint1 iVertex, Array<uint2, 3> &nbrs) const;
  void computeDelaunayForces(uint1 iVertex, const Array<Array<real_t, 3>, 3> &stress,
                             Array<Array<real_t, 3>, 3> &forces);
  Array<Array<real_t, 3>, 3> velocityGradientDelaunay(
      uint1 iVertex, const Array<uint2, 3> &nbrs,
      const std::vector<Array<real_t, 3> > &velocities) const;
  Array<real_t, 3> force(const std::vector<Array<Array<real_t, 3>, 3> > &stresses) const;
  void gradFacetAreaSq(uint1 facetIndx, std::vector<uint2> &indx,
                       std::vector<Array<real_t, 3> > &grad) const;
  inline const std::vector<Array<real_t, 3> > &getdV() const { return m_dV; }
  inline const std::vector<Array<real_t, 3> > &getAreas() const { return m_areas; }
  real_t getVolume() const { return m_vol; }
  const std::vector<real_t> &getVolumeDelaunay() const { return m_volDelaunay; }
  const std::vector<Array<Array<Array<real_t, 3>, 3>, 3> > &getOmega() const { return m_omega; }
  bool isConvex() const;
  inline Cell<real_t> &getCell() { return *p_cell; }
  inline const std::vector<Array<real_t, 3> > &getConnVect() const { return m_connV; }
  inline const std::vector<real_t> &getConnVectSq() const { return m_rSq; }

 protected:
  Cell<real_t> *p_cell;
  std::vector<Array<real_t, 3> > m_connV;
  std::vector<real_t> m_rSq;
  std::vector<Array<Array<real_t, 3>, 3> > m_edgeInv;
  std::vector<real_t> m_volDelaunay;
  std::vector<Array<real_t, 3> > m_areas;
  real_t m_vol;
  // omega[i][j][l][k]
  // neighbor corresponding to facet i differentiated into j-direction
  // l: displacement direction, k: normal direction
  std::vector<Array<Array<Array<real_t, 3>, 3>, 3> > m_omega;
  std::vector<Array<real_t, 3> > m_dV;
};

template <typename real_t>
class CellUpdater {
 public:
  CellUpdater() : m_isSetup(false) {}
  CellUpdater(CellGeometry<real_t> &geom);
  CellUpdater &operator=(CellGeometry<real_t> &rhs);
  CellUpdater &operator=(const CellUpdater<real_t> &rhs);
  inline void reset();
  uint1 findFacet(uint2 nbr) const;
  void updateNbrInserts();
  inline const std::vector<NbrInsert> &getNbrInserts() { return m_nbrInserts; }
  inline void clearNbrInserts() { m_nbrInserts.clear(); }
  bool processNbrInserts(NbrInsertItr begin, NbrInsertItr end, CellMaker<real_t> &maker,
                         const std::vector<Array<real_t, 3> > &pos, const Box<real_t> &box,
                         CellComplexUpdateStats *stats = NULL);
  friend class CellMaker<real_t>;
  friend class CellGeometry<real_t>;
  // inline bool isConvex2() const;
 protected:
  void setupNbrSet();
  inline bool isSetupNbrSet() { return m_isSetup; }
  inline bool isInNbrs(uint2 nbr) const;
  // inline bool isConvex() const;
  CellGeometry<real_t> *p_geom;
  boost::container::flat_set<uint2> m_nbrs;
  std::vector<uint2> m_nbrsWrk;
  std::vector<NbrInsert> m_nbrInserts;
  bool m_isSetup;
  std::vector<PosAndId<uint2, real_t> > m_newNbrsWrk;
  std::vector<Array<bool, 3> > m_visitedWrk;
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
class CellArena {
 public:
  struct CellSpan {
    uint2 id;
    uint2 vertexOffset;
    uint1 vertexCount;
    uint2 facetOffset;
    uint1 facetCount;
  };

  #ifndef VOR_CELLARENA_RESERVE_VERTICES_PER_CELL
  #define VOR_CELLARENA_RESERVE_VERTICES_PER_CELL 28
  #endif
  #ifndef VOR_CELLARENA_RESERVE_FACETS_PER_CELL
  #define VOR_CELLARENA_RESERVE_FACETS_PER_CELL 20
  #endif
    static constexpr uint1 kReserveVerticesPerCell =
      static_cast<uint1>(VOR_CELLARENA_RESERVE_VERTICES_PER_CELL);
    static constexpr uint1 kReserveFacetsPerCell =
      static_cast<uint1>(VOR_CELLARENA_RESERVE_FACETS_PER_CELL);

  void clear() {
    m_spans.clear();
    m_vertexPos.clear();
    m_vertices.clear();
    m_facets.clear();
    m_nbr.clear();
  }

  void rebuildFromCells(const std::vector<Cell<real_t> > &cells) {
    clear();
    m_spans.reserve(cells.size());
    m_vertexPos.reserve(cells.size() * static_cast<size_t>(kReserveVerticesPerCell));
    m_vertices.reserve(cells.size() * static_cast<size_t>(kReserveVerticesPerCell));
    m_facets.reserve(cells.size() * static_cast<size_t>(kReserveFacetsPerCell));
    m_nbr.reserve(cells.size() * static_cast<size_t>(kReserveFacetsPerCell));

    for (size_t i = 0; i < cells.size(); ++i) {
      const Cell<real_t> &cell = cells[i];
      const uint1 nv = static_cast<uint1>(cell.m_numVertices);
      const uint1 nf = static_cast<uint1>(cell.m_numFacets);

      CellSpan span;
      span.id = cell.m_id;
      span.vertexOffset = static_cast<uint2>(m_vertexPos.size());
      span.vertexCount = nv;
      span.facetOffset = static_cast<uint2>(m_facets.size());
      span.facetCount = nf;
      m_spans.push_back(span);

      m_vertexPos.insert(m_vertexPos.end(), cell.m_vertexPos, cell.m_vertexPos + nv);
      m_vertices.insert(m_vertices.end(), cell.m_vertices, cell.m_vertices + nv);
      m_facets.insert(m_facets.end(), cell.m_facets, cell.m_facets + nf);
      m_nbr.insert(m_nbr.end(), cell.m_nbr, cell.m_nbr + nf);
    }
  }

  size_t numCells() const { return m_spans.size(); }
  const CellSpan &span(size_t i) const { return m_spans[i]; }
  uint2 cellId(size_t i) const { return m_spans[i].id; }
  uint1 cellNumVertices(size_t i) const { return m_spans[i].vertexCount; }
  uint1 cellNumFacets(size_t i) const { return m_spans[i].facetCount; }
  const Array<real_t, 3> *cellVertexPosData(size_t i) const {
    return m_vertexPos.data() + m_spans[i].vertexOffset;
  }
  const Vertex *cellVertexLabelData(size_t i) const {
    return m_vertices.data() + m_spans[i].vertexOffset;
  }
  const uint1 *cellFacetLabelData(size_t i) const { return m_facets.data() + m_spans[i].facetOffset; }
  const uint2 *cellNbrData(size_t i) const { return m_nbr.data() + m_spans[i].facetOffset; }

  void materializeCell(size_t i, Cell<real_t> &cell) const {
    const CellSpan &s = m_spans[i];
    cell.m_id = s.id;
    cell.m_numVertices = s.vertexCount;
    cell.m_numFacets = s.facetCount;
    std::memcpy(cell.m_vertexPos, m_vertexPos.data() + s.vertexOffset,
                s.vertexCount * sizeof(cell.m_vertexPos[0]));
    std::memcpy(cell.m_vertices, m_vertices.data() + s.vertexOffset,
                s.vertexCount * sizeof(cell.m_vertices[0]));
    std::memcpy(cell.m_facets, m_facets.data() + s.facetOffset,
                s.facetCount * sizeof(cell.m_facets[0]));
    std::memcpy(cell.m_nbr, m_nbr.data() + s.facetOffset, s.facetCount * sizeof(cell.m_nbr[0]));
  }

  const std::vector<CellSpan> &spans() const { return m_spans; }
  const std::vector<Array<real_t, 3> > &vertexPos() const { return m_vertexPos; }
  const std::vector<Vertex> &vertices() const { return m_vertices; }
  const std::vector<uint1> &facets() const { return m_facets; }
  const std::vector<uint2> &nbrs() const { return m_nbr; }

 private:
  std::vector<CellSpan> m_spans;
  std::vector<Array<real_t, 3> > m_vertexPos;
  std::vector<Vertex> m_vertices;
  std::vector<uint1> m_facets;
  std::vector<uint2> m_nbr;
};

template <typename real_t>
class CellComplex {
 public:
  CellComplex(Box<real_t> *box) : m_nbrList(box), m_isBuild(false) {}
  void build(const std::vector<Array<real_t, 3> > &p);
  /// Build geometry data (connecting vectors, edge inverses, volume derivatives)
  /// for all cells. Called automatically by build(); call separately if needed.
  void buildGeometry(const std::vector<Array<real_t, 3> > &p);
  void update(const std::vector<Array<real_t, 3> > &p);
  size_t numCells() const { return m_cellArena.numCells(); }
  void materializeCell(size_t i, Cell<real_t> &cell) const { m_cellArena.materializeCell(i, cell); }
  void materializeCells(std::vector<Cell<real_t> > &cells) const {
    cells.resize(m_cellArena.numCells());
    for (size_t i = 0; i < cells.size(); ++i)
      m_cellArena.materializeCell(i, cells[i]);
  }
  std::vector<uint0> &getTypes() { return m_types; }
  const std::vector<uint0> &getTypes() const { return m_types; }
  std::vector<CellGeometry<real_t> > &getGeoms() { return m_geom; }
  const std::vector<CellGeometry<real_t> > &getGeoms() const { return m_geom; }
  CellArena<real_t> &getCellArena() { return m_cellArena; }
  const CellArena<real_t> &getCellArena() const { return m_cellArena; }
  const NbrList<uint2, real_t> &getNbrList() const { return m_nbrList; }
  const CellComplexUpdateStats &getLastUpdateStats() const { return m_lastUpdateStats; }
  void drawInterfaceGnuplot(uint0 iType, uint0 jType, const std::vector<Array<real_t, 3> > &p,
                            FILE *fp) const;

 private:
  void repair(const std::vector<Array<real_t, 3> > &p);
  void initNbrList(const std::vector<Array<real_t, 3> > &p);
  NbrList<uint2, real_t> m_nbrList;
  std::vector<uint0> m_types;
  std::vector<Cell<real_t> > m_cells;
  CellArena<real_t> m_cellArena;
  std::vector<Cell<real_t> > m_geomCells;
  std::vector<CellGeometry<real_t> > m_geom;
  std::vector<CellUpdater<real_t> > m_updaters;
  std::vector<bool> m_hasChanged;
  CellComplexUpdateStats m_lastUpdateStats;
  bool m_isBuild;
};

class NbrsToFacets {
 public:
  NbrsToFacets() {}
  template <typename real_t>
  void init(const std::vector<Cell<real_t> > &cells);
  template <typename real_t>
  void init(const CellArena<real_t> &arena);
  void print() const;
  //    void makeMatrixdVdV(const std::vector<real_t> & dV);
 protected:
  template <typename real_t>
  NbrsToFacets transposedV(const std::vector<real_t> &values, std::vector<real_t> &valuesTr) const;
  uint2 m_numCells;
  std::vector<uint2> m_ptr;
  std::vector<uint2> m_nbr;
  std::vector<uint1> m_facet;
};

template <typename real_t>
Cell<real_t>::Cell(const Cell<real_t> &rhs)
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
Cell<real_t> &Cell<real_t>::operator=(const Cell<real_t> &rhs) {
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
Cell<real_t> &Cell<real_t>::operator=(CellMaker<real_t> &rhs) {
  m_id = rhs.m_id;
  rhs.renumber();
  m_numVertices = rhs.m_numVertices;
  m_numFacets = rhs.m_numFacets;
  std::memcpy(m_vertexPos, rhs.m_vertexPos.data(), m_numVertices * sizeof(m_vertexPos[0]));
  std::memcpy(m_vertices, rhs.m_vertices.data(), m_numVertices * sizeof(m_vertices[0]));
  std::memcpy(m_facets, rhs.m_facets.data(), m_numFacets * sizeof(m_facets[0]));
  std::memcpy(m_nbr, rhs.m_nbr.data(), m_numFacets * sizeof(m_nbr[0]));
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
void Cell<real_t>::printNbrFacets(const std::vector<Cell<real_t> > &cells) const {
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
void Cell<real_t>::drawGnuplot(Array<real_t, 3> p, FILE *fp) const {
  for (uint1 i(0); i < m_numFacets; ++i) {
    drawFacetGnuplot(i, p, fp);
    fputs("\n\n", fp);
  }
}

template <typename real_t>
void Cell<real_t>::drawFacetGnuplot(uint1 iFacet, Array<real_t, 3> p, FILE *fp) const {
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
void Cell<real_t>::printFacetInfo(Array<real_t, 3> p, uint facet_id) const {
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
Cuboid<real_t>::Cuboid(const Array<real_t, 3> &L) {
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

template <typename real_t>
CellMaker<real_t>::CellMaker()
  : p_nbrList(NULL), m_checkGCHead(0u), m_isAllCut(false), m_distGen(0) {
  m_slotsV.setActiveCapacity(kInitialV);
  m_slotsF.setActiveCapacity(kInitialF);
  ensureVertexBuffers(m_slotsV.activeCapacity());
  ensureFacetBuffers(m_slotsF.activeCapacity());
  updatePeakCounter(cellMakerTelemetry().peak_vertex_capacity,
                    static_cast<uint64_t>(m_slotsV.activeCapacity()));
  updatePeakCounter(cellMakerTelemetry().peak_facet_capacity,
                    static_cast<uint64_t>(m_slotsF.activeCapacity()));
  std::fill(m_knownDistGen.begin(), m_knownDistGen.end(), 0u);
  m_newVerticesWrk.reserve(20);
  m_facetPrevWrk.reserve(20);
  m_nbrsWrk.reserve(40);
  m_checkGridCell.reserve(64);
  m_vStackWrk.reserve(32);
}

template <typename real_t>
void CellMaker<real_t>::ensureVertexBuffers(uint1 minSize) {
  if (m_vertexPos.size() < minSize)
    m_vertexPos.resize(minSize);
  if (m_vertices.size() < minSize)
    m_vertices.resize(minSize);
  if (m_rSq.size() < minSize)
    m_rSq.resize(minSize);
  if (m_dist.size() < minSize)
    m_dist.resize(minSize);
  if (m_distGC.size() < minSize)
    m_distGC.resize(minSize);
  if (m_knownDistGen.size() < minSize)
    m_knownDistGen.resize(minSize, 0u);
  if (m_renumVWrk.size() < minSize)
    m_renumVWrk.resize(minSize);
}

template <typename real_t>
void CellMaker<real_t>::ensureFacetBuffers(uint1 minSize) {
  if (m_facets.size() < minSize)
    m_facets.resize(minSize);
  if (m_nbr.size() < minSize)
    m_nbr.resize(minSize);
  if (m_renumFWrk.size() < minSize)
    m_renumFWrk.resize(minSize);
}

template <typename real_t>
CellMaker<real_t>::~CellMaker() {}

template <typename real_t>
CellMaker<real_t> &CellMaker<real_t>::operator=(const Cell<real_t> &rhs) {
  m_id = rhs.m_id;
  init(rhs);
  return *this;
}

template <typename real_t>
CellMaker<real_t> &CellMaker<real_t>::operator=(const CellUpdater<real_t> &rhs) {
  m_id = rhs.p_cell->m_id;
  init(*(rhs.p_cell));
  return *this;
}

template <typename real_t>
uint1 CellMaker<real_t>::getNextLabelCCW(uint1 label) const {
  uint1 facetMasked(label & maskFacet);
  uint1 revLabel(m_vertices[getVertex(label)][getEdge(label)]);
  uint1 vertexMasked(revLabel & maskVertex);
  uint1 edge(getEdge(revLabel));
  (edge == 0 ? edge = 2 : --edge);
  return (facetMasked | vertexMasked | edge);
}

template <typename real_t>
uint1 CellMaker<real_t>::getReverseLabel(uint1 label) const {
  return m_vertices[getVertex(label)][getEdge(label)];
}

template <typename real_t>
bool CellMaker<real_t>::applyCut(const Array<real_t, 3> &p, real_t rSqHalf, uint2 nbr) {
  if (kUseCutCell2)
    return cutCell2(p, rSqHalf, nbr);
  return cutCell(p, rSqHalf, nbr);
}

template <typename real_t>
uint1 CellMaker<real_t>::allocVertexChecked(const char *caller) {
  uint1 v_new = m_slotsV.getFree();
  if (v_new == kMaxV && m_slotsV.growActiveCapacity()) {
    ensureVertexBuffers(m_slotsV.activeCapacity());
    cellMakerTelemetry().vertex_growth_events.fetch_add(1, std::memory_order_relaxed);
    updatePeakCounter(cellMakerTelemetry().peak_vertex_capacity,
                      static_cast<uint64_t>(m_slotsV.activeCapacity()));
    v_new = m_slotsV.getFree();
  }
  if (v_new == kMaxV)
    failCapacity(caller, "vertex", kMaxV);
  return v_new;
}

template <typename real_t>
uint1 CellMaker<real_t>::allocFacetChecked(const char *caller) {
  uint1 f_new = m_slotsF.getFree();
  if (f_new == kMaxF && m_slotsF.growActiveCapacity()) {
    ensureFacetBuffers(m_slotsF.activeCapacity());
    cellMakerTelemetry().facet_growth_events.fetch_add(1, std::memory_order_relaxed);
    updatePeakCounter(cellMakerTelemetry().peak_facet_capacity,
                      static_cast<uint64_t>(m_slotsF.activeCapacity()));
    f_new = m_slotsF.getFree();
  }
  if (f_new == kMaxF)
    failCapacity(caller, "facet", kMaxF);
  return f_new;
}

template <typename real_t>
[[noreturn]] void CellMaker<real_t>::failCapacity(const char *caller, const char *kind,
                                                  uint1 capacity) const {
  if (kind[0] == 'v') {
    cellMakerTelemetry().vertex_overflow_events.fetch_add(1, std::memory_order_relaxed);
  } else {
    cellMakerTelemetry().facet_overflow_events.fetch_add(1, std::memory_order_relaxed);
  }
  std::fprintf(stderr,
               "Fatal: CellMaker %s capacity exceeded while processing cell %u in %s. "
               "Configured %s capacity is %u.\n",
               kind, static_cast<unsigned>(m_id), caller, kind,
               static_cast<unsigned>(capacity));
  std::abort();
}

template <typename real_t>
void CellMaker<real_t>::init(const Cell<real_t> &cell) {
  m_slotsV.reset(cell.m_numVertices);
  ensureVertexBuffers(m_slotsV.activeCapacity());
  std::memcpy(m_vertexPos.data(), cell.m_vertexPos, cell.m_numVertices * sizeof(m_vertexPos[0]));
  std::memcpy(m_vertices.data(), cell.m_vertices, cell.m_numVertices * sizeof(m_vertices[0]));
  m_slotsF.reset(cell.m_numFacets);
  ensureFacetBuffers(m_slotsF.activeCapacity());
  std::memcpy(m_facets.data(), cell.m_facets, cell.m_numFacets * sizeof(m_facets[0]));
  std::memcpy(m_nbr.data(), cell.m_nbr, cell.m_numFacets * sizeof(m_nbr[0]));
  computeAllRsq();
  resetDist();
  const std::numeric_limits<real_t> lim;
  m_distGCMax = -lim.max();
  m_vDistGCMax = maxNumVertices;
}

template <typename real_t>
void CellMaker<real_t>::computeAllRsq() {
  m_vRsqMax = m_slotsV.firstAlive();
  for (uint1 i = 0; i < m_slotsV.numAllocated(); ++i) {
    if (m_slotsV.isFree(i))
      continue;
    computeRsq(i);
  }
}

template <typename real_t>
void CellMaker<real_t>::computeRsq(uint1 i) {
  m_rSq[i] = m_vertexPos[i][0] * m_vertexPos[i][0];
  for (uint0 k(1); k < 3; ++k)
    m_rSq[i] += m_vertexPos[i][k] * m_vertexPos[i][k];
  if (m_rSq[i] > m_rSq[m_vRsqMax]) {
    m_vRsqMax = i;
  }
}

template <typename real_t>
void CellMaker<real_t>::findRsqMax() {
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

template <typename real_t>
void CellMaker<real_t>::resetDist() {
  const std::numeric_limits<real_t> lim;
  m_distMax = -lim.max();
  m_vDistMax = maxNumVertices;
  ++m_distGen;
}

template <typename real_t>
real_t CellMaker<real_t>::computeDist(uint1 i, const Array<real_t, 3> p, const real_t rSqHalf) {
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

template <typename real_t>
real_t CellMaker<real_t>::computeDist(uint1 i, const Array<real_t, 3> p) {
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

template <typename real_t>
void CellMaker<real_t>::computeAllDist(const Array<real_t, 3> p, const real_t rSqHalf) {
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

template <typename real_t>
void CellMaker<real_t>::renumber() {
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

template <typename real_t>
void CellMaker<real_t>::computeGCOrig(uint2 indx, const Array<real_t, 3> &pos) {
  Indx indcs(p_nbrList->getGrid().expand(indx));
  const Array<real_t, 3> &L(p_nbrList->getBox().getL());
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

template <typename real_t>
real_t CellMaker<real_t>::computeRsqMinGC() const {
  Array<real_t, 3> dx(m_relOrigGC);
  real_t rSq(0);
  for (uint0 k(0); k < 3; ++k) {
    (dx[k] < 0 ? (dx[k] < -m_dLGC[k] ? dx[k] += m_dLGC[k] : dx[k] = 0) : dx[k]);
    rSq += dx[k] * dx[k];
  }
  return rSq;
}

template <typename real_t>
real_t CellMaker<real_t>::computeDistGC(uint1 indx) {
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

template <typename real_t>
Array<real_t, 3> CellMaker<real_t>::getClosestPointGC(uint1 indx) const {
  Array<real_t, 3> pos;
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

template <typename real_t>
real_t CellMaker<real_t>::computeMaxDistGC() {
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

template <typename real_t>
real_t CellMaker<real_t>::computeMaxDistGCVerb() {
  if (m_slotsV.empty())
    return 0;
  uint1 v1 = m_slotsV.firstAlive();
  Array<real_t, 3> posGC(getClosestPointGC(v1));
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

template <typename real_t>
void CellMaker<real_t>::computeAllDistGC() {
  const std::numeric_limits<real_t> lim;
  m_distGCMax = -lim.max();
  m_vDistGCMax = maxNumVertices;
  for (uint1 indx = 0; indx < m_slotsV.numAllocated(); ++indx) {
    if (m_slotsV.isFree(indx))
      continue;
    computeDistGC(indx);
  }
}

template <typename real_t>
void CellMaker<real_t>::getAllDistGCVerb() {
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

template <typename real_t>
bool CellMaker<real_t>::cutCell2(const Array<real_t, 3> p, real_t rSqHalf, uint2 nbr) {
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

template <typename real_t>
bool CellMaker<real_t>::cutCell(const Array<real_t, 3> p, real_t rSqHalf, uint2 nbr) {
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

template <typename real_t>
bool CellMaker<real_t>::build(uint2 id, const std::vector<Array<real_t, 3> > &pos,
                              const NbrList<uint2, real_t> &nbrList, const Cell<real_t> &initCell) {
  p_nbrList = &nbrList;
  {
    const Array<real_t, 3> &L(p_nbrList->getBox().getL());
    const Indx &N(p_nbrList->getGrid().getN());
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
      (processNbrs(begin, end, pos[m_id], p_nbrList->getBox()) == true ? isUpdated = true
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
    computeGCOrig(indx, pos[m_id]);
    real_t rSqMin = computeRsqMinGC();
    if (rSqMin > 4.0 * m_rSq[m_vRsqMax])
      continue;
    p_nbrList->getCellContent(indx, begin, end);
    if (end != begin)
      (processNbrs(begin, end, pos[m_id], p_nbrList->getBox()) == true ? isUpdated = true
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
    computeGCOrig(indx, pos[m_id]);
    real_t rSqMin = computeRsqMinGC();
    if (rSqMin > 4.0 * m_rSq[m_vRsqMax])
      continue;
    computeAllDistGC();
    if (m_distGCMax < 0)
      continue;
    p_nbrList->getCellContent(indx, begin, end);
    if (end != begin)
      (processNbrs(begin, end, pos[m_id], p_nbrList->getBox()) == true ? isUpdated = true
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

template <typename real_t>
bool CellMaker<real_t>::rebuild(const std::vector<Array<real_t, 3> > &pos, const Box<real_t> &box,
                                const Cell<real_t> &initCell) {
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

template <typename real_t>
bool CellMaker<real_t>::processNbrs(
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator begin,
    typename std::vector<PosAndId<uint2, real_t> >::const_iterator end, const Array<real_t, 3> pos0,
    const Box<real_t> &box) {
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
    Array<real_t, 3> relPos;
    for (uint0 k(0); k < 3; ++k)
      relPos[k] = (itr->pos)[k] - pos0[k];
    box.makeShortestDistance(relPos);
    for (uint0 k(0); k < 3; ++k)
      nbrDist[k] = relPos[k];
    nbrDist.rSqHalf = 0.5 * (relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2]);
    if (nbrDist.rSqHalf < 2.0 * m_rSq[m_vRsqMax])
      m_nbrDistWrk.push_back(nbrDist);
  }
  std::sort(m_nbrDistWrk.begin(), m_nbrDistWrk.end(), CompareNbrDist<real_t>());
  for (size_t i(0); (i < m_nbrDistWrk.size()) && (m_nbrDistWrk[i].rSqHalf < 2.0 * m_rSq[m_vRsqMax]);
       ++i) {
    const NbrDist<real_t> &p(m_nbrDistWrk[i]);
    (applyCut(p, p.rSqHalf, p.id) ? isCut = true : m_isAllCut = false);
  }
  return isCut;
}

template <typename real_t>
void CellMaker<real_t>::getCloseNbrs(NbrInsert &nbrs) {
  for (uint k(0); k < 3; ++k) {
    uint1 facet(getFacet(m_vertices[m_vDistMax][k]));
    nbrs[k] = m_nbr[facet];
  }
}

template <typename real_t>
CellUpdater<real_t>::CellUpdater(CellGeometry<real_t> &geom) : m_isSetup(false) {
  p_geom = &geom;
}

template <typename real_t>
CellUpdater<real_t> &CellUpdater<real_t>::operator=(CellGeometry<real_t> &geom) {
  p_geom = &geom;
  reset();
  return *this;
}

template <typename real_t>
CellUpdater<real_t> &CellUpdater<real_t>::operator=(const CellUpdater<real_t> &rhs) {
  if (&rhs == this)
    return *this;
  p_geom = this->p_geom;
  m_nbrs = this->m_nbrs;
  m_nbrsWrk = this->m_nbrsWrk;
  m_nbrInserts = this->m_nbrInserts;
  m_isSetup = this->m_isSetup;
  m_newNbrsWrk = this->m_newNbrsWrk;
  m_visitedWrk = this->m_visitedWrk;
  return *this;
}

template <typename real_t>
void CellUpdater<real_t>::reset() {
  m_nbrs.clear();
  clearNbrInserts();
  m_isSetup = false;
}

template <typename real_t>
void CellUpdater<real_t>::setupNbrSet() {
  const uint1 numFacets(p_geom->getCell().m_numFacets);
  const uint2 *const nbr(p_geom->getCell().m_nbr);
  m_nbrsWrk.clear();
  m_nbrsWrk.reserve(numFacets);
  for (uint0 i(0); i < (numFacets); ++i)
    if (nbr[i] != noNbr)
      m_nbrsWrk.push_back(nbr[i]);
  std::sort(m_nbrsWrk.begin(), m_nbrsWrk.end());
  m_nbrs.clear();
  m_nbrs.insert(m_nbrsWrk.begin(), m_nbrsWrk.end());
  m_isSetup = true;
}

template <typename real_t>
bool CellUpdater<real_t>::isInNbrs(uint2 nbr) const {
  return (m_nbrs.find(nbr) != m_nbrs.end());
}

template <typename real_t>
uint1 CellUpdater<real_t>::findFacet(uint2 nbr) const {
  const uint1 numFacets(p_geom->getCell().m_numFacets);
  const uint2 *nbrs = p_geom->getCell().m_nbr;
  uint1 indx(~0);
  for (uint0 i(0); i < numFacets; ++i)
    (nbrs[i] == nbr ? indx = i : indx);
  return indx;
}

template <typename real_t>
void CellUpdater<real_t>::updateNbrInserts() {
  const Cell<real_t> &cell(p_geom->getCell());
  const uint1 numFacets(cell.m_numFacets);
  const uint2 *const nbr(cell.m_nbr);
  const uint1 *const facets(cell.m_facets);
  NbrInsert nbrIns;
  nbrIns[1] = cell.m_id;
  for (uint1 i(0); i < numFacets; ++i) {
    if (nbr[i] != noNbr) {
      nbrIns[0] = nbr[i];
      nbrIns[2] = nbrIns[1];
      m_nbrInserts.push_back(nbrIns);
      uint1 labelStart(facets[i]);
      uint1 label(labelStart);
      do {
        if (nbr[getFacet(cell.getReverseLabel(label))] != noNbr) {
          nbrIns[2] = nbr[getFacet(cell.getReverseLabel(label))];
          m_nbrInserts.push_back(nbrIns);
        }
        label = cell.getNextLabelCCW(label);
      } while (label != labelStart);
      // Emit self-proposals for every other neighbor of this cell to
      // neighbor i.  Self-proposals route through the cutCell path
      // (direct geometric intersection) rather than the processNbrs
      // batch path, ensuring non-edge-adjacent neighbor pairs are
      // tested individually.
      for (uint1 j(0); j < numFacets; ++j) {
        if (j != i && nbr[j] != noNbr) {
          nbrIns[0] = nbr[i];
          nbrIns[1] = nbr[j];
          nbrIns[2] = nbr[j];
          m_nbrInserts.push_back(nbrIns);
        }
      }
    }
  }
}

template <typename real_t>
bool CellUpdater<real_t>::processNbrInserts(NbrInsertItr begin, NbrInsertItr end,
                                            CellMaker<real_t> &maker,
                                            const std::vector<Array<real_t, 3> > &pos,
                                            const Box<real_t> &box,
                                            CellComplexUpdateStats *stats) {
  //    printf("begin %u %u %u\n", (*begin)[0], (*begin)[1], (*begin)[2]);
  bool hasChanged(false);
  if (begin == end)
    return hasChanged;
  NbrInsert nbrClose, nbrIns;
  if (!m_isSetup)
    setupNbrSet();
  m_newNbrsWrk.clear();
  boost::container::flat_set<uint2> attemptedNbrs;
  PosAndId<uint2, real_t> newNbr;
  Cell<real_t> &cell(p_geom->getCell());
  uint2 id(cell.getID());
  bool hasSetMaker(false);
  for (NbrInsertItr itr(begin); itr != end; ++itr) {
    // printf("trying %u %u %u\n", (*itr)[0], (*itr)[1], (*itr)[2]);
    if (isInNbrs((*itr)[2]))
      continue;
    if ((*itr)[0] != id)
      continue;
    if ((*itr)[1] == (*itr)[2]) {
      if (attemptedNbrs.find((*itr)[1]) != attemptedNbrs.end())
        continue;
      attemptedNbrs.insert((*itr)[1]);
      if (stats != NULL)
        ++stats->num_repair_direct_attempts;
      // printf("testing %u %u %u\n", (*itr)[0], (*itr)[1], (*itr)[2]);
      if (!hasSetMaker) {
        maker = cell;
        hasSetMaker = true;
      }
      Array<real_t, 3> relPos;
      real_t rSqHalf;
      for (uint0 k(0); k < 3; ++k)
        relPos[k] = pos[(*itr)[1]][k] - pos[id][k];
      box.makeShortestDistance(relPos);
      rSqHalf = 0.5 * (relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2]);
      bool isInserted = maker.applyCut(relPos, rSqHalf, (*itr)[1]);
      if (isInserted) {
        hasChanged = true;
        m_nbrs.insert((*itr)[1]);
        if (stats != NULL)
          ++stats->num_repair_direct_successes;
      }
      if (!isInserted) {
        //	  printf("not inserted: %u %u %u\n",(*itr)[0],(*itr)[1],(*itr)[2]);
        // const std::vector<uint2> & nbrs(maker.getNbrs());
        // for(size_t i(0); i<nbrs.size(); ++i)
        //   if (nbrs[i] != noNbr){
        //     nbrIns[0] = (*itr)[1];
        //     nbrIns[1] = nbrs[i];
        //     nbrIns[2] = nbrs[i];
        // 	m_nbrInserts.push_back(nbrIns);
        //   }
        //   //If (*itr)[1] is not a neighbor of (*itr)[0] then (*itr)[0] is not a neighbor of
        //   (*itr)[1]
        //   //Propose other neighbors to cell (*itr)[1] that might be closerby
        maker.getCloseNbrs(nbrClose);
        nbrIns[0] = (*itr)[1];
        for (uint k(0); k < 3; ++k) {
          if (nbrClose[k] != noNbr) {
            nbrIns[1] = nbrClose[k];
            nbrIns[2] = nbrClose[k];
            m_nbrInserts.push_back(nbrIns);
          }
        }
        // skip other neighbor suggestions proposed by (*itr)[1]
        uint2 nbr((*itr)[1]);
        for (; (itr + 1) != end && (*itr)[1] == nbr && (*itr)[0] == cell.m_id; ++itr) {
        }
      }
    } else {
      //	printf("inserting %u %u %u\n", (*itr)[0], (*itr)[1], (*itr)[2]);
      if (attemptedNbrs.find((*itr)[2]) != attemptedNbrs.end())
        continue;
      attemptedNbrs.insert((*itr)[2]);
      if (stats != NULL)
        ++stats->num_repair_indirect_candidates;
      newNbr.id = (*itr)[2];
      newNbr.pos = pos[newNbr.id];
      m_newNbrsWrk.push_back(newNbr);
    }
  }
  if (!m_newNbrsWrk.empty()) {
    if (!hasSetMaker)
      maker = cell;
    if (stats != NULL)
      ++stats->num_repair_batch_calls;
    if (maker.processNbrs(m_newNbrsWrk.begin(), m_newNbrsWrk.end(), pos[id], box)) {
      hasChanged = true;
      if (stats != NULL)
        ++stats->num_repair_batch_changes;
    }
  }
  if (hasChanged) {
    cell = maker;
    // Reset the neighbor set so it is rebuilt from the actual cell
    // topology on the next call.  Without this, candidates that were
    // staged for processNbrs but rejected by the distance filter
    // remain in m_nbrs and are permanently skipped, even though a
    // subsequent self-proposal could succeed via cutCell.
    m_isSetup = false;
  }
  return hasChanged;
}

template <typename real_t>
CellGeometry<real_t>::CellGeometry() : p_cell(NULL) {}

template <typename real_t>
CellGeometry<real_t>::CellGeometry(Cell<real_t> &cell) {
  p_cell = &cell;
}

template <typename real_t>
CellGeometry<real_t> &CellGeometry<real_t>::operator=(Cell<real_t> &rhs) {
  p_cell = &rhs;
  m_connV.clear();
  m_rSq.clear();
  m_edgeInv.clear();
  m_areas.clear();
  m_vol = 0.0;
  m_omega.clear();
  m_dV.clear();
  return *this;
}

template <typename real_t>
CellGeometry<real_t> &CellGeometry<real_t>::operator=(const CellGeometry<real_t> &rhs) {
  if (&rhs == this)
    return *this;
  this->p_cell = rhs.p_cell;
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
void CellGeometry<real_t>::computeConnectingVectors(const std::vector<Array<real_t, 3> > &pos,
                                                    const Box<real_t> &box) {
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
    Array<uint1, 3> indxF;
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
  Array<real_t, 3> dA;
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
  std::vector<Array<Array<real_t, 3>, 3> > dA(numVertices);
  std::vector<Array<Array<real_t, 3>, 3> > dv(numVertices);
  std::vector<Array<real_t, 3> > xcm(numFacets);
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
// Array<Array<real_t, 3>, 3> CellGeometry<real_t>::velocityGradient(const std::vector<Array<real_t,
// 3> > & velocities) const
// {
//   Array<Array<real_t, 3>, 3> gradV; //gradV[i][j] = dv[i]/dx[j]
//   Array<real_t, 3> vCenter = velocities[p_cell->m_id];
//   // omega[i][j][l][k]
//   // neighbor corresponding to facet i differentiated into j-direction
//   // l: displacement direction, k: normal direction
//   for(int l(0); l<3; ++l)
//     for(int k(0); k<3; ++k)
// 	gradV[l][k] = 0.0;
//   for(int i(0); i< m_omega.size(); ++i){
//     Array<real_t, 3> v = velocities[p_cell->m_nbr[i]];
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
Array<Array<real_t, 3>, 3> CellGeometry<real_t>::velocityGradient(
    const std::vector<Array<real_t, 3> > &velocities) const {
  Array<Array<real_t, 3>, 3> gradV;  // gradV[i][j] = dv[i]/dx[j]
  for (int l(0); l < 3; ++l)
    for (int k(0); k < 3; ++k)
      gradV[l][k] = 0.0;
  for (int i(0); i < m_areas.size(); ++i) {
    Array<real_t, 3> v = velocities[p_cell->m_nbr[i]];
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
void CellGeometry<real_t>::getDelaunayNbrs(uint1 iVertex, Array<uint2, 3> &nbrs) const {
  Array<uint1, 3> indxF;
  indxF[0] = getFacet(p_cell->m_vertices[iVertex][2]);
  indxF[1] = getFacet(p_cell->m_vertices[iVertex][0]);
  indxF[2] = getFacet(p_cell->m_vertices[iVertex][1]);
  for (uint0 m(0); m < 3; ++m)
    nbrs[m] = p_cell->getNbrs()[indxF[m]];
}

template <typename real_t>
Array<Array<real_t, 3>, 3> CellGeometry<real_t>::velocityGradientDelaunay(
    uint1 iVertex, const Array<uint2, 3> &nbrs,
    const std::vector<Array<real_t, 3> > &velocities) const {
  Array<Array<real_t, 3>, 3> gradV;  // gradV[i][j] = dv[i]/dx[j]
  for (int l(0); l < 3; ++l)
    for (int k(0); k < 3; ++k)
      gradV[l][k] = 0.0;
  Array<real_t, 3> v0 = velocities[p_cell->getID()];
  for (uint0 m(0); m < 3; ++m) {
    Array<real_t, 3> v = velocities[nbrs[m]];
    Array<real_t, 3> dv;
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
                                                 const Array<Array<real_t, 3>, 3> &stress,
                                                 Array<Array<real_t, 3>, 3> &forces) {
  for (uint0 m(0); m < 3; ++m)
    for (uint0 k(0); k < 3; ++k) {
      forces[m][k] = 0;
      for (uint0 l(0); l < 3; ++l)
        forces[m][k] += stress[k][l] * m_edgeInv[iVertex][m][l];
      forces[m][k] *= -m_volDelaunay[iVertex];
    }
}

template <typename real_t>
Array<real_t, 3> CellGeometry<real_t>::force(
    const std::vector<Array<Array<real_t, 3>, 3> > &stresses) const {
  Array<real_t, 3> f;  // gradV[i][j] = dv[i]/dx[j]
  Array<Array<real_t, 3>, 3> stressCenter = stresses[p_cell->id];
  // omega[i][j][l][k]
  // neighbor corresponding to facet i differentiated into j-direction
  // l: displacement direction, k: normal direction
  for (int j(0); j < 3; ++j)
    f[j] = 0.0;
  for (int i(0); i < m_omega.size(); ++i) {
    Array<Array<real_t, 3>, 3> stress = stresses[p_cell->m_nbr[i]];
    Array<Array<real_t, 3>, 3> dStress;
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
void CellGeometry<real_t>::gradFacetAreaSq(uint1 indxFacet, std::vector<uint2> &indxFacets,
                                           std::vector<Array<real_t, 3> > &grad) const {
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
void CellComplex<real_t>::initNbrList(const std::vector<Array<real_t, 3> > &p) {
  const Array<real_t, 3> &L(m_nbrList.getBox().getL());
  real_t density = real_t(p.size()) / (L[0] * L[1] * L[2]);
  real_t rcut = 1.75 * (std::pow(density, -1.0 / 3.0));
  m_nbrList.setup(p, rcut);
  //    printf("number grid cells: %u\n", m_nbrList.getGrid().getN()[0]);
}

template <typename real_t>
void CellComplex<real_t>::build(const std::vector<Array<real_t, 3> > &p) {
  initNbrList(p);
  const Array<real_t, 3> &L(m_nbrList.getBox().getL());
  Cuboid<real_t> cub(L);
  m_cells.resize(p.size());
#pragma omp parallel
  {
    CellMaker<real_t> maker;
#pragma omp for
    for (uint2 i = 0; i < p.size(); ++i) {
      maker.build(i, p, m_nbrList, cub);
      m_cells[i] = maker;
    }
  }
  m_nbrList.clear();
  m_cellArena.rebuildFromCells(m_cells);
  buildGeometry(p);
  m_isBuild = true;
}

template <typename real_t>
void CellComplex<real_t>::buildGeometry(const std::vector<Array<real_t, 3> > &p) {
  if (m_cellArena.numCells() != m_cells.size())
    m_cellArena.rebuildFromCells(m_cells);
  m_geom.resize(p.size());
  m_geomCells.resize(p.size());
  m_updaters.resize(p.size());
#pragma omp parallel for
  for (size_t i = 0; i < m_updaters.size(); ++i) {
    m_cellArena.materializeCell(i, m_geomCells[i]);
    m_geom[i] = m_geomCells[i];
    m_geom[i].computeConnectingVectors(p, m_nbrList.getBox());
    m_geom[i].computeEdgeInv();
    m_geom[i].diffVolume();
    m_updaters[i] = m_geom[i];
  }
  m_hasChanged.resize(p.size());
}

template <typename real_t>
void CellComplex<real_t>::update(const std::vector<Array<real_t, 3> > &p) {
  m_lastUpdateStats = CellComplexUpdateStats();
  m_lastUpdateStats.num_cells = static_cast<uint2>(p.size());
  (p.size() == m_cells.size() ? m_isBuild : m_isBuild = false);
  if (!m_isBuild) {
    m_lastUpdateStats.rebuilt_from_scratch = true;
    build(p);
    return;
  }
  for (size_t i = 0; i < m_updaters.size(); ++i) {
    m_updaters[i].reset();
    m_hasChanged[i] = false;
  }
  uint2 numChanged(0);
  const Box<real_t> &box(m_nbrList.getBox());
  for (size_t i = 0; i < m_geom.size(); ++i) {
    m_geom[i].computeConnectingVectors(p, box);
    m_geom[i].computeEdgeInv();
    m_geom[i].updateVertexPos();
    bool isConvex = m_geom[i].isConvex();
    if (!isConvex) {
      m_hasChanged[i] = true;
      ++numChanged;
    }
  }
  m_lastUpdateStats.num_non_convex_before = numChanged;
  m_lastUpdateStats.num_rebuild_candidates = numChanged;
  //    printf("number changed cells: %u\n",numChanged);
  if (numChanged == 0)
    return;
  const Array<real_t, 3> &L(box.getL());
  Cuboid<real_t> cub(L);
  std::vector<uint2> emptyCells;
  {
    CellMaker<real_t> maker;
    for (size_t i = 0; i < m_updaters.size(); ++i) {
      if (!m_hasChanged[i])
        continue;
      maker = m_cells[i];
      maker.rebuild(p, box, cub);
      m_cells[i] = maker;
      ++m_lastUpdateStats.num_local_rebuild_cells;
      if (m_cells[i].numVertices() == 0 || m_cells[i].hasNoNbr())
        emptyCells.push_back(i);
    }
  }
  //    printf("number empty cells: %lu\n",emptyCells.size());
  m_lastUpdateStats.num_empty_after_local_rebuild = static_cast<uint2>(emptyCells.size());
  if (!emptyCells.empty()) {
    initNbrList(p);
    {
      CellMaker<real_t> maker;
      for (size_t i = 0; i < emptyCells.size(); ++i) {
        maker.build(emptyCells[i], p, m_nbrList, cub);
        m_cells[emptyCells[i]] = maker;
        ++m_lastUpdateStats.num_full_rebuild_cells;
      }
    }
  }
  for (size_t i = 0; i < m_updaters.size(); ++i) {
    if (!m_hasChanged[i])
      continue;
    m_geomCells[i] = m_cells[i];
    m_geom[i] = m_geomCells[i];
    m_updaters[i] = m_geom[i];
  }
  repair(p);
  m_cellArena.rebuildFromCells(m_cells);
  m_geomCells.resize(m_cells.size());
  for (size_t i = 0; i < m_geom.size(); ++i) {
    m_cellArena.materializeCell(i, m_geomCells[i]);
    m_geom[i] = m_geomCells[i];
    m_geom[i].computeConnectingVectors(p, box);
    m_geom[i].computeEdgeInv();
    m_geom[i].diffVolume();
  }
}

template <typename real_t>
void CellComplex<real_t>::repair(const std::vector<Array<real_t, 3> > &p) {
  //    ProfilerStart("test.prof");
  const bool debugRepair = (std::getenv("VOR_DEBUG_REPAIR") != NULL);
  for (size_t i = 0; i < m_updaters.size(); ++i) {
    if (!m_hasChanged[i])
      continue;
    m_updaters[i].updateNbrInserts();
  }
  bool needsUpdate(true);
  while (needsUpdate) {
    ++m_lastUpdateStats.num_repair_iterations;
    const uint2 iterIndex = m_lastUpdateStats.num_repair_iterations;
    const uint2 proposalsBefore = m_lastUpdateStats.num_repair_proposals_total;
    const uint2 groupsBefore = m_lastUpdateStats.num_repair_target_groups_total;
    const uint2 changedBefore = m_lastUpdateStats.num_repair_cells_changed_total;
    const uint2 directAttemptsBefore = m_lastUpdateStats.num_repair_direct_attempts;
    const uint2 directSuccessesBefore = m_lastUpdateStats.num_repair_direct_successes;
    const uint2 indirectBefore = m_lastUpdateStats.num_repair_indirect_candidates;
    const uint2 batchCallsBefore = m_lastUpdateStats.num_repair_batch_calls;
    const uint2 batchChangesBefore = m_lastUpdateStats.num_repair_batch_changes;
    needsUpdate = false;
    std::vector<NbrInsert> nbrInserts, nbrInsTmp;
    {
      std::vector<NbrInsert> nbrInsPriv;
      for (size_t i = 0; i < m_updaters.size(); ++i) {
        const std::vector<NbrInsert> &inserts(m_updaters[i].getNbrInserts());
        if (!inserts.empty()) {
          nbrInsPriv.insert(nbrInsPriv.end(), inserts.begin(), inserts.end());
          m_updaters[i].clearNbrInserts();
          needsUpdate = true;
        }
      }
      std::sort(nbrInsPriv.begin(), nbrInsPriv.end(), CompareNbrInsert());
      nbrInsTmp.resize(nbrInserts.size() + nbrInsPriv.size());
      std::merge(nbrInserts.begin(), nbrInserts.end(), nbrInsPriv.begin(), nbrInsPriv.end(),
                 nbrInsTmp.begin(), CompareNbrInsert());
      nbrInserts.swap(nbrInsTmp);
    }
    m_lastUpdateStats.num_repair_proposals_total += static_cast<uint2>(nbrInserts.size());
    //      std::__parallel::std::sort(nbrInserts.begin(), nbrInserts.end(), CompareNbrInsert());
    // std::sort(nbrInserts.begin(), nbrInserts.end(), CompareNbrInsert());
    CellMaker<real_t> maker;
    size_t nextGroupBegin = 0;
    while (nextGroupBegin < nbrInserts.size()) {
      size_t groupEnd = nextGroupBegin;
      const uint2 cellId = nbrInserts[nextGroupBegin][0];
      while (groupEnd < nbrInserts.size() && nbrInserts[groupEnd][0] == cellId) {
        ++groupEnd;
      }
      ++m_lastUpdateStats.num_repair_target_groups_total;
      NbrInsertItr beginPriv = nbrInserts.begin() + static_cast<std::ptrdiff_t>(nextGroupBegin);
      NbrInsertItr endPriv = nbrInserts.begin() + static_cast<std::ptrdiff_t>(groupEnd);
      bool hasChanged = m_updaters[cellId].processNbrInserts(beginPriv, endPriv, maker, p,
                                                             m_nbrList.getBox(),
                                                             &m_lastUpdateStats);
      m_hasChanged[cellId] = true;
      if (hasChanged) {
        m_cells[cellId] = maker;
        m_updaters[cellId].updateNbrInserts();
        ++m_lastUpdateStats.num_repair_cells_changed_total;
      }
      nextGroupBegin = groupEnd;
    }
    if (debugRepair) {
      std::fprintf(stderr,
                   "DEBUG repair iter=%u proposals=%u target_groups=%u changed_cells=%u "
                   "direct_attempts=%u direct_successes=%u indirect_candidates=%u "
                   "batch_calls=%u batch_changes=%u next_round=%d\n",
                   static_cast<unsigned>(iterIndex),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_proposals_total - proposalsBefore),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_target_groups_total - groupsBefore),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_cells_changed_total - changedBefore),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_direct_attempts - directAttemptsBefore),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_direct_successes - directSuccessesBefore),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_indirect_candidates - indirectBefore),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_batch_calls - batchCallsBefore),
                   static_cast<unsigned>(m_lastUpdateStats.num_repair_batch_changes - batchChangesBefore),
                   needsUpdate ? 1 : 0);
    }
  }
}

template <typename real_t>
void CellComplex<real_t>::drawInterfaceGnuplot(uint0 iType, uint0 jType,
                                               const std::vector<Array<real_t, 3> > &pos,
                                               FILE *fp) const {
#pragma omp parallel for
  for (size_t i = 0; i < m_types.size(); ++i) {
    if (m_types[i] == iType) {
      Cell<real_t> cell;
      m_cellArena.materializeCell(i, cell);
      for (uint1 j = 0; j < cell.numFacets(); ++j) {
        if (m_types[cell.getNbr(j)] == jType) {
          cell.drawFacetGnuplot(j, pos[i], fp);
          fputs("\n\n", fp);
        }
      }
    }
  }
}

template <typename real_t>
void NbrsToFacets::init(const std::vector<Cell<real_t> > &cells) {
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
void NbrsToFacets::init(const CellArena<real_t> &arena) {
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
      const uint2 *nbrData = arena.cellNbrData(i);
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
NbrsToFacets NbrsToFacets::transposedV(const std::vector<real_t> &values,
                                       std::vector<real_t> &valuesTr) const {
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
  std::vector<std::pair<uint2, std::pair<uint1, Array<real_t, 3> > > > nbrTmp(m_nbr.size());
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
//     std::vector< Array<real_t, 3> > values(m_nbr.size());
// #pragma omp parallel for
//     for(uint2 i=0; i< m_ptr.size()-1; ++i){
//       const std::vector< Array<real_t, 3> > & dV(geoms[i].getdV());
//       for(uint2 j=m_ptr[i]; j < m_ptr[i+1]; ++j)
// 	for(uint0 k(0); k<3; ++k)
// 	  values[j][k] = dV[m_facet[j]][k];
//     }
//     std::vector< Array<real_t, 3> > valuesTr();
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
