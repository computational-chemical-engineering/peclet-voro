/**
 * @file vor_types.hpp
 * @brief General-purpose type definitions and utility classes for the Voronoi
 *        dynamics library.
 *
 * Defines integer type aliases, bit-mask constants, static array helpers,
 * comparison functors, and index-management utilities used throughout the
 * voronoi_dynamics library.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace vor {

typedef uint8_t uint0;
typedef uint16_t uint1;
typedef uint32_t uint2;

static const uint1 shiftFacet(9);
static const uint1 maskFacet(static_cast<uint1>(~0u << shiftFacet));
static const uint1 maskNoFacet((uint1)~maskFacet);
static const uint1 maskEdge(3);
static const uint1 maskVertex((uint1) ~(maskFacet | maskEdge));
static const uint2 noNbr(~0);
static const uint1 maxNumVertices(1 << 7);
static const uint1 maxNumFacets(1 << 7);

/**
 * @class Array
 * @brief class simple class for static arrays. The purpose is to provide the use of [][]-notation
 * when using e.g. vectors of arrays
 * @tparam T type of the elements of the array
 * @tparam n size of the array
 */
template <typename T, unsigned int n>
class Array {
 public:
  inline T& operator[](unsigned int i) { return m[i]; }
  inline const T& operator[](unsigned int i) const { return m[i]; }

 private:
  T m[n];
};

typedef Array<uint1, 3> Vertex;
typedef Array<uint2, 3> NbrInsert;
typedef std::vector<NbrInsert>::const_iterator NbrInsertItr;

template <typename real_t>
class NbrDist : public Array<real_t, 3> {
 public:
  uint2 id;
  real_t rSqHalf;
};

template <typename real_t>
class MatrixEntry {
 public:
  inline size_t& col() { return m_col; }
  inline size_t col() const { return m_col; }
  inline size_t& row() { return m_row; }
  inline size_t row() const { return m_row; }
  inline real_t& value() { return m_value; }
  inline real_t value() const { return m_value; }

 private:
  size_t m_col, m_row;
  real_t m_value;
};

template <typename real_t>
class CompareMatrixEntryRow {
 public:
  inline bool operator()(const MatrixEntry<real_t>& a, const MatrixEntry<real_t>& b) const {
    return (a.row() == b.row() ? a.col() < b.col() : a.row() < b.row());
  }
};

template <typename real_t>
class CompressedRowStorage {
 public:
  CompressedRowStorage(size_t numRows, size_t numNZ) : m_rowPtr(numRows + 1, 0) {
    m_values.reserve(numNZ);
    m_rowPtr.reserve(numNZ);
  }
  std::vector<real_t>& getValues() { return m_values; }
  std::vector<size_t>& getColInd() { return m_colInd; }
  std::vector<size_t>& getRowPtr() { return m_rowPtr; }

 private:
  std::vector<real_t> m_values;
  std::vector<size_t> m_colInd;
  std::vector<size_t> m_rowPtr;
};

class ComparePairFirst {
 public:
  template <typename T1, typename T2>
  inline bool operator()(std::pair<T1, T2> a, std::pair<T1, T2> b) const {
    return a.first < b.first;
  }
};

template <typename real_t>
class CompareNbrDist {
 public:
  inline bool operator()(const NbrDist<real_t>& a, const NbrDist<real_t>& b) const {
    return a.rSqHalf < b.rSqHalf;
  }
};

class CompareNbrInsert {
 public:
  inline bool operator()(const NbrInsert& a, const NbrInsert& b) const { return a[0] < b[0]; }
};

/**
 * @class IndxList
 * @brief class for handing out and managing indices and iterate through used indices
 * @tparam Uint unsigned integer type used as type for the indices
 */
template <typename UInt>
class IndxList {
 public:
  //! @brief constructor
  //! @param endIndx the valid indices given out run from 0 to endIndx-1
  IndxList(UInt endIndx) : m_endIndx(endIndx), m_next(endIndx), m_free(endIndx) { reset(0); }
  //! @brief get the end index to end an iteration
  //! @return value of endIndx
  inline UInt endIndx() const { return m_endIndx; }
  //! @brief get the beginning index to start an iteration
  //! @return value of the first index in use
  inline UInt beginIndx() const { return m_firstUsed; }
  //! @brief get the next iteration from i in an iteration
  //! @param i value of the initial index index
  //! @return value of the next index in use
  inline UInt nextIndx(UInt i) const { return m_next[i]; }
  //! @brief test if an index is free
  //! @param i index to be tested
  //! @return true if the index is free, false if it is in use
  inline bool isFree(UInt i) const { return m_free[i]; }
  //! @brief release an index (change it from in-use to free)
  //! @param i index to be release
  //! @return true if the index is release, false if the index was already free
  bool release(UInt i);
  //! @brief get a free index (after which it is in use)
  //! @return index. If there are not free indices anymore endIndx is returned
  UInt getFree();
  //! @brief reset the index list
  //! @param N first free index (indices 0...N-1 are in-use)
  void reset(UInt N = 0);

 private:
  std::vector<bool> m_free;
  std::vector<UInt> m_next;
  const UInt m_endIndx;
  UInt m_firstFree, m_firstUsed;
};

/**
 * @class VisitedIndx
 * @brief class for registring of indices are visited. This is useful for e.g. vertex tranversal.
 * @tparam Uint unsigned integer type used as type for the indices
 */
template <typename UInt>
class VisitedIndx {
 public:
  //! @brief constructor
  VisitedIndx();
  //! @brief constructor
  //! @param endIndx maximum number of indices
  VisitedIndx(UInt endIndx);
  //! @brief destructor
  ~VisitedIndx();
  //! @brief (re)initiallize
  //! @param endIndx: maximum number of indices
  void init(UInt endIndx);
  //! @brief set index i to visited
  //! @param i index that is set to visited
  inline void set(UInt i);
  //! @brief check if index has been visited
  //! @param i index to be checked
  //! @return true if index has been visited
  inline bool isVisited(UInt i) { return m_visited[i]; }
  //! @brief reset all indices to visited==false
  void reset();
  //! @return maximum number of indices allowed
  UInt size() { return m_endIndx; }

 private:
  UInt m_endIndx;
  bool* m_visited;
  std::vector<UInt> m_visitedIndx;
};

template <typename UInt>
bool IndxList<UInt>::release(UInt i) {
  if (i >= m_endIndx)
    return false;
  if (m_free[i])
    return false;
  UInt nextUsed(m_next[i]);
  UInt prev;
  for (prev = i; !m_free[prev] && prev != 0; --prev) {
  }
  if (m_free[prev]) {
    m_next[i] = m_next[prev];
    m_next[prev] = i;
  } else {
    m_next[i] = m_firstFree;
    m_firstFree = i;
  }
  m_free[i] = true;
  for (prev = i; m_free[prev] && prev != 0; --prev) {
  }
  if (!m_free[prev])
    m_next[prev] = nextUsed;
  else
    m_firstUsed = nextUsed;
  return true;
}

template <typename UInt>
UInt IndxList<UInt>::getFree() {
  if (m_firstFree == m_endIndx)
    return m_endIndx;
  UInt i = m_firstFree;
  m_firstFree = m_next[i];
  if (i == 0) {
    m_next[i] = m_firstUsed;
    m_firstUsed = i;
  } else {
    m_next[i] = m_next[i - 1];
    m_next[i - 1] = i;
  }
  m_free[i] = false;
  return i;
}

template <typename UInt>
void IndxList<UInt>::reset(UInt N) {
  for (UInt i(0); i < m_endIndx; ++i)
    m_next[i] = i + 1;
  {
    if (N > m_endIndx) {
      N = m_endIndx;
    }
    UInt j(0);
    for (; j < N; ++j)
      m_free[j] = false;
    for (; j < m_endIndx; ++j)
      m_free[j] = true;
  }
  if (N == 0) {
    m_firstFree = 0;
    m_firstUsed = m_endIndx;
  } else {
    m_firstFree = N;
    m_firstUsed = 0;
    m_next[N - 1] = m_endIndx;
  }
}

template <typename UInt>
VisitedIndx<UInt>::VisitedIndx() : m_endIndx(0), m_visited(NULL) {}

template <typename UInt>
VisitedIndx<UInt>::VisitedIndx(UInt endIndx) {
  init(endIndx);
}

template <typename UInt>
void VisitedIndx<UInt>::init(UInt endIndx) {
  m_endIndx = endIndx;
  if (m_visited != NULL)
    delete[] m_visited;
  m_visited = new bool[m_endIndx];
  for (UInt i(0); i < m_endIndx; ++i)
    m_visited[i] = false;
  m_visitedIndx.reserve(endIndx);
  m_visitedIndx.clear();
}

template <typename UInt>
VisitedIndx<UInt>::~VisitedIndx() {
  if (m_visited != NULL)
    delete[] m_visited;
}

template <typename UInt>
void VisitedIndx<UInt>::set(UInt i) {
  if (i >= m_endIndx || m_visited[i])
    return;
  m_visited[i] = true;
  m_visitedIndx.push_back(i);
}

template <typename UInt>
void VisitedIndx<UInt>::reset() {
  if (m_visitedIndx.size() < (m_endIndx / 5)) {
    for (UInt i(0); i < m_visitedIndx.size(); ++i)
      m_visited[m_visitedIndx[i]] = false;
  } else {
    for (UInt i(0); i < m_endIndx; ++i)
      m_visited[i] = false;
  }
  m_visitedIndx.clear();
}

/// Make a half-edge label encoding facet, vertex and edge indices.
inline uint1 makeLabel(uint1 facet, uint1 vertex, uint1 edge) {
  return static_cast<uint1>((facet << shiftFacet) | (vertex << 2) | edge);
}
/// Extract the facet index from a half-edge label.
inline uint1 getFacet(uint1 label) { return static_cast<uint1>((label & maskFacet) >> shiftFacet); }
/// Extract the vertex index from a half-edge label.
inline uint1 getVertex(uint1 label) { return static_cast<uint1>((label & maskVertex) >> 2); }
/// Extract the edge index (0-2) from a half-edge label.
inline uint1 getEdge(uint1 label) { return static_cast<uint1>(label & maskEdge); }

/**
 * @brief Dense slot allocator replacing IndxList for CellMaker vertex/facet tracking.
 * Stores active slot indices in a dense prefix of a fixed-size array.
 * - alloc(): O(1) – returns slots[numUsed++]
 * - free(i): O(numUsed) scan then swap-to-back, O(1) amortized
 * - Iteration: sequential scan of slots[0..numUsed-1]
 * - sort(): sorts active prefix for renumber() which requires ascending order
 *
 * @tparam maxSlots Maximum number of slots (should equal maxNumVertices-1 or maxNumFacets-1).
 */
template <uint8_t maxSlots>
class SlotAllocator {
 public:
  SlotAllocator() : m_numUsed(0) {
    for (uint8_t i = 0; i < maxSlots; ++i) m_slots[i] = i;
  }

  /// Reset: first n slots are in use (0..n-1), rest are free.
  void reset(uint8_t n) {
    m_numUsed = n;
    for (uint8_t i = 0; i < maxSlots; ++i) m_slots[i] = i;
  }

  /// Allocate a free slot (returns the slot index).
  inline uint8_t alloc() { return m_slots[m_numUsed++]; }

  /// Free slot at index idx (O(numUsed) linear scan + swap-to-back).
  inline void free(uint8_t idx) {
    for (uint8_t i = 0; i < m_numUsed; ++i) {
      if (m_slots[i] == idx) {
        --m_numUsed;
        m_slots[i] = m_slots[m_numUsed];
        m_slots[m_numUsed] = idx;
        return;
      }
    }
  }

  /// Check if slot idx is free (not in active prefix).
  inline bool isFree(uint8_t idx) const {
    for (uint8_t i = 0; i < m_numUsed; ++i)
      if (m_slots[i] == idx) return false;
    return true;
  }

  /// Number of active slots.
  inline uint8_t numUsed() const { return m_numUsed; }

  /// Access i-th active slot (0-based).
  inline uint8_t operator[](uint8_t i) const { return m_slots[i]; }

  /// Sentinel for end-of-iteration (one past last active slot).
  inline uint8_t endIndx() const { return maxSlots; }

  /// First active slot (for API compatibility with IndxList).
  inline uint8_t beginIndx() const { return (m_numUsed > 0 ? m_slots[0] : maxSlots); }

  /// Sort active slots in ascending order (required before renumber()).
  inline void sort() { std::sort(m_slots, m_slots + m_numUsed); }

 private:
  uint8_t m_slots[maxSlots];
  uint8_t m_numUsed;
};

}  // namespace vor
