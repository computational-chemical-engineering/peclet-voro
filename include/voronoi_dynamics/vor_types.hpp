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

#include <cstdint>
#include <cstring>
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
 * @class DenseSlots
 * @brief Cache-friendly slot allocator replacing linked-list-based IndxList.
 *
 * Manages a fixed-capacity pool of slot indices.  Alive slots are tracked via
 * a byte array (no bit-packing).  A small stack caches recently freed indices
 * for O(1) re-allocation.  Iteration is a simple sequential scan over the
 * [0, numAllocated) range, checking the alive flag — ideal for the hardware
 * prefetcher.
 *
 * @tparam UInt unsigned integer type for slot indices
 * @tparam Capacity maximum number of slots
 */
template <typename UInt, UInt Capacity>
class DenseSlots {
 public:
  DenseSlots() : m_numAllocated(0), m_numAlive(0), m_freeTop(0), m_activeCapacity(Capacity) {
    std::memset(m_alive, 0, Capacity);
  }
  /// Reset: slots 0..n-1 are alive, rest are dead.
  void reset(UInt n) {
    if (n > m_activeCapacity)
      setActiveCapacity(n);
    m_numAllocated = n;
    m_numAlive = n;
    m_freeTop = 0;
    std::memset(m_alive, 0, Capacity);
    for (UInt i = 0; i < n; ++i)
      m_alive[i] = 1;
  }
  /// Allocate a free slot (returns Capacity if none available).
  UInt getFree() {
    UInt idx;
    if (m_freeTop > 0) {
      idx = m_freeStack[--m_freeTop];
    } else if (m_numAllocated < m_activeCapacity) {
      idx = m_numAllocated++;
    } else {
      return Capacity;
    }
    m_alive[idx] = 1;
    ++m_numAlive;
    return idx;
  }
  /// Release slot i. Returns true if it was alive.
  bool release(UInt i) {
    if (i >= Capacity || !m_alive[i])
      return false;
    m_alive[i] = 0;
    if (m_freeTop < Capacity)
      m_freeStack[m_freeTop++] = i;
    --m_numAlive;
    return true;
  }
  /// Check if slot i is free (dead).
  inline bool isFree(UInt i) const { return !m_alive[i]; }
  /// Number of slots ever allocated (high-water mark for iteration).
  inline UInt numAllocated() const { return m_numAllocated; }
  /// Number of currently alive slots.
  inline UInt numAlive() const { return m_numAlive; }
  /// Current active capacity (<= Capacity).
  inline UInt activeCapacity() const { return m_activeCapacity; }
  /// Set active capacity (clamped to [numAllocated, Capacity]).
  void setActiveCapacity(UInt cap) {
    if (cap < m_numAllocated)
      cap = m_numAllocated;
    if (cap > Capacity)
      cap = Capacity;
    m_activeCapacity = cap;
  }
  /// Grow active capacity by doubling (or to target if provided).
  /// Returns true if capacity increased.
  bool growActiveCapacity(UInt target = 0) {
    if (m_activeCapacity >= Capacity)
      return false;
    UInt new_cap = target;
    if (new_cap == 0) {
      new_cap = static_cast<UInt>(m_activeCapacity * 2);
      if (new_cap <= m_activeCapacity)
        new_cap = static_cast<UInt>(m_activeCapacity + 1);
    }
    if (new_cap > Capacity)
      new_cap = Capacity;
    if (new_cap <= m_activeCapacity)
      return false;
    m_activeCapacity = new_cap;
    return true;
  }
  /// True if no alive slots.
  inline bool empty() const { return m_numAlive == 0; }
  /// Find first alive slot index (returns Capacity if empty).
  inline UInt firstAlive() const {
    for (UInt i = 0; i < m_numAllocated; ++i)
      if (m_alive[i])
        return i;
    return Capacity;
  }

 private:
  uint8_t m_alive[Capacity];
  UInt m_freeStack[Capacity];
  UInt m_freeTop;
  UInt m_numAllocated;
  UInt m_numAlive;
  UInt m_activeCapacity;
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
  IndxList(UInt endIndx) : m_endIndx(endIndx), m_next(endIndx), m_free(endIndx, 1) { reset(0); }
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
  inline bool isFree(UInt i) const { return m_free[i] != 0; }
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
  std::vector<uint8_t> m_free;  ///< byte array (not bit-packed) for fast byte access
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
  m_free[i] = 1;
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
  m_free[i] = 0;
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
      m_free[j] = 0;
    for (; j < m_endIndx; ++j)
      m_free[j] = 1;
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

}  // namespace vor
