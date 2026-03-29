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
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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

typedef std::array<uint1, 3> Vertex;
typedef std::array<uint2, 3> NbrInsert;
typedef std::vector<NbrInsert>::const_iterator NbrInsertItr;

template <typename real_t>
class NbrDist : public std::array<real_t, 3> {
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
  inline bool operator()(const NbrInsert& a, const NbrInsert& b) const {
    if (a[0] != b[0])
      return a[0] < b[0];
    if (a[1] != b[1])
      return a[1] < b[1];
    return a[2] < b[2];
  }
};

/**
 * @class DenseSlotsView
 * @brief Logic-only slot allocator view backed by externally owned storage.
 *
 * Alive flags and the free stack are supplied by the caller via `setStorage()`.
 * This keeps slot-management logic portable while leaving allocation policy to
 * the owner.
 *
 * @tparam UInt unsigned integer type for slot indices
 */
template <typename UInt>
class DenseSlotsView {
 public:
  static constexpr UInt InvalidIdx = static_cast<UInt>(~0);

  DenseSlotsView()
      : m_alive(NULL), m_freeStack(NULL), m_capacity(0), m_numAllocated(0), m_numAlive(0),
        m_freeTop(0) {}

  void setStorage(uint8_t *alivePtr, UInt *stackPtr, UInt capacity) {
    m_alive = alivePtr;
    m_freeStack = stackPtr;
    m_capacity = capacity;
  }

  void reset(UInt n) {
    m_numAllocated = n;
    m_numAlive = n;
    m_freeTop = 0;
  }

  UInt getFree() {
    UInt idx;
    if (m_freeTop > 0) {
      idx = m_freeStack[--m_freeTop];
    } else if (m_numAllocated < m_capacity) {
      idx = m_numAllocated++;
    } else {
      return InvalidIdx;
    }
    m_alive[idx] = 1;
    ++m_numAlive;
    return idx;
  }

  bool release(UInt i) {
    if (i >= m_capacity || !m_alive[i])
      return false;
    m_alive[i] = 0;
    m_freeStack[m_freeTop++] = i;
    --m_numAlive;
    return true;
  }

  inline bool isFree(UInt i) const { return !m_alive[i]; }
  inline UInt numAllocated() const { return m_numAllocated; }
  inline UInt numAlive() const { return m_numAlive; }
  inline UInt activeCapacity() const { return m_capacity; }
  inline bool empty() const { return m_numAlive == 0; }
  inline UInt firstAlive() const {
    for (UInt i = 0; i < m_numAllocated; ++i)
      if (m_alive[i])
        return i;
    return InvalidIdx;
  }

 private:
  uint8_t *m_alive;
  UInt *m_freeStack;
  UInt m_capacity;
  UInt m_numAllocated;
  UInt m_numAlive;
  UInt m_freeTop;
};

template <typename T>
class ChunkedPool {
 public:
  static constexpr uint2 InvalidIdx = static_cast<uint2>(~0u);

  explicit ChunkedPool(size_t chunkSize = 1024)
      : m_nextChunk(0), m_chunkSize(std::max<size_t>(chunkSize, 1)) {}

  void clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_chunks.clear();
    m_freeList.clear();
    m_nextChunk.store(0, std::memory_order_relaxed);
  }

  void resetReuse() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_freeList.clear();
    m_nextChunk.store(0, std::memory_order_relaxed);
  }

  T *allocate(size_t count, uint2 &outOverflowIdx) {
    if (count == 0) {
      outOverflowIdx = InvalidIdx;
      return NULL;
    }
    if (count > m_chunkSize) {
      std::fprintf(stderr,
                   "Fatal: ChunkedPool allocation of %zu items exceeds chunk size %zu\n", count,
                   m_chunkSize);
      std::abort();
    }

    uint2 chunkIdx = InvalidIdx;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      if (!m_freeList.empty()) {
        chunkIdx = m_freeList.back();
        m_freeList.pop_back();
      }
    }

    if (chunkIdx == InvalidIdx)
      chunkIdx = static_cast<uint2>(m_nextChunk.fetch_add(1, std::memory_order_relaxed));

    ensureChunk(static_cast<size_t>(chunkIdx) + 1);
    outOverflowIdx = static_cast<uint2>(static_cast<size_t>(chunkIdx) * m_chunkSize);
    return m_chunks[chunkIdx].get();
  }

  void releaseChunk(uint2 overflowIdx) {
    if (overflowIdx == InvalidIdx)
      return;
    const uint2 chunkIdx = static_cast<uint2>(overflowIdx / m_chunkSize);
    std::lock_guard<std::mutex> lock(m_mutex);
    m_freeList.push_back(chunkIdx);
  }

  T &get(size_t idx) { return m_chunks[idx / m_chunkSize][idx % m_chunkSize]; }
  const T &get(size_t idx) const { return m_chunks[idx / m_chunkSize][idx % m_chunkSize]; }
  size_t chunkSize() const { return m_chunkSize; }

 private:
  void ensureChunk(size_t requiredChunks) {
    std::lock_guard<std::mutex> lock(m_mutex);
    while (m_chunks.size() < requiredChunks)
      m_chunks.emplace_back(new T[m_chunkSize]());
  }

  std::vector<std::unique_ptr<T[]> > m_chunks;
  std::vector<uint2> m_freeList;
  std::atomic<size_t> m_nextChunk;
  size_t m_chunkSize;
  std::mutex m_mutex;
};

template <typename T, uint1 PrimaryCap>
class PrimaryOverflowArray {
 public:
  static constexpr uint2 InvalidOverflow = ChunkedPool<T>::InvalidIdx;

  explicit PrimaryOverflowArray(size_t overflowChunkSize = 1024) : m_overflow(overflowChunkSize) {}

  void clear() {
    m_primary.clear();
    m_counts.clear();
    m_overflowIdx.clear();
    m_overflow.clear();
  }

  void resize(uint2 numCells) {
    m_primary.assign(static_cast<size_t>(numCells) * static_cast<size_t>(PrimaryCap), T());
    m_counts.assign(numCells, 0);
    m_overflowIdx.assign(numCells, InvalidOverflow);
    m_overflow.clear();
  }

  void prepare(uint2 numCells) {
    const size_t primarySize = static_cast<size_t>(numCells) * static_cast<size_t>(PrimaryCap);
    if (m_primary.size() != primarySize)
      m_primary.resize(primarySize);
    if (m_counts.size() != numCells)
      m_counts.resize(numCells);
    if (m_overflowIdx.size() != numCells)
      m_overflowIdx.resize(numCells);
    std::fill(m_counts.begin(), m_counts.end(), 0);
    std::fill(m_overflowIdx.begin(), m_overflowIdx.end(), InvalidOverflow);
    m_overflow.resetReuse();
  }

  void insert(uint2 cellId, const T *data, uint1 count) {
    m_counts[cellId] = count;
    T *primary = primaryData(cellId);
    const uint1 primaryCount = std::min<uint1>(count, PrimaryCap);
    for (uint1 i = 0; i < primaryCount; ++i)
      primary[i] = data[i];
    for (uint1 i = primaryCount; i < PrimaryCap; ++i)
      primary[i] = T();

    if (count > PrimaryCap) {
      uint2 overflowIdx = InvalidOverflow;
      T *overflow = m_overflow.allocate(static_cast<size_t>(count - PrimaryCap), overflowIdx);
      for (uint1 i = PrimaryCap; i < count; ++i)
        overflow[i - PrimaryCap] = data[i];
      m_overflowIdx[cellId] = overflowIdx;
    } else {
      m_overflowIdx[cellId] = InvalidOverflow;
    }
  }

  void overwrite(uint2 cellId, const T *data, uint1 count) {
    releaseCellOverflow(cellId);
    insert(cellId, data, count);
  }

  void clearCell(uint2 cellId) {
    releaseCellOverflow(cellId);
    m_counts[cellId] = 0;
    T *primary = primaryData(cellId);
    for (uint1 i = 0; i < PrimaryCap; ++i)
      primary[i] = T();
  }

  uint1 count(uint2 cellId) const { return m_counts[cellId]; }
  bool hasOverflow(uint2 cellId) const { return m_overflowIdx[cellId] != InvalidOverflow; }
  const T &get(uint2 cellId, uint1 idx) const {
    if (idx < PrimaryCap)
      return m_primary[(static_cast<size_t>(cellId) * static_cast<size_t>(PrimaryCap)) + idx];
    return m_overflow.get(static_cast<size_t>(m_overflowIdx[cellId]) + (idx - PrimaryCap));
  }
  T *primaryData(uint2 cellId) {
    return m_primary.data() + (static_cast<size_t>(cellId) * static_cast<size_t>(PrimaryCap));
  }
  const T *primaryData(uint2 cellId) const {
    return m_primary.data() + (static_cast<size_t>(cellId) * static_cast<size_t>(PrimaryCap));
  }

  const std::vector<T> &primary() const { return m_primary; }
  const std::vector<uint1> &counts() const { return m_counts; }
  const std::vector<uint2> &overflowIdx() const { return m_overflowIdx; }

 private:
  void releaseCellOverflow(uint2 cellId) {
    if (m_overflowIdx[cellId] != InvalidOverflow) {
      m_overflow.releaseChunk(m_overflowIdx[cellId]);
      m_overflowIdx[cellId] = InvalidOverflow;
    }
  }

  std::vector<T> m_primary;
  ChunkedPool<T> m_overflow;
  std::vector<uint1> m_counts;
  std::vector<uint2> m_overflowIdx;
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
