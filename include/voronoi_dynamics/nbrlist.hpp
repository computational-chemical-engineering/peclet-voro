/**
 * @file nbrlist.hpp
 * @brief Neighbor-list and spatial grid data structures for efficient
 *        particle lookups.
 *
 * Provides Grid, Box, BoxLE (Lees-Edwards), PosAndId, and NbrList template
 * classes used to locate particle neighbours in O(1) average time via
 * cell-linked lists.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#ifdef VORONOI_USE_OPENMP
#include <omp.h>
#endif

#include "vor_types.hpp"

namespace vor {
typedef std::array<uint1, 3> Indx;

/// Branchless floor: truncation rounds towards zero; for negative non-integer
/// values we subtract 1 to correctly round down (e.g. int(-0.3)=0, need -1).
template <typename real_t>
inline real_t fastFloor(real_t x) {
  int n = static_cast<int>(x);
  n -= (x < n);
  return static_cast<real_t>(n);
}

template <typename UInt>
class Grid {
 public:
  void init(const Indx& n);
  void init(uint1 n);
  inline const Indx& getN() const { return m_n; }
  inline UInt numCells() const { return m_numCells; }
  inline Indx expand(UInt indxCell) const;
  inline void expand(UInt indxCell, Indx& indx) const;
  inline UInt compress(const Indx& indx) const;
  inline void getNbrs(UInt indxCell, std::vector<UInt>& ngbrs) const;
  inline void getDirectNbrs(UInt indxCell, std::vector<UInt>& ngbrs) const;

 private:
  Indx m_n;
  Indx m_kMax;
  UInt m_numCells;
  UInt m_numNbrs;
};

template <typename real_t = float>
class Box {
 public:
  Box() {}
  Box(std::array<real_t, 3> L) : m_L(L) { computeInvL(); }
  Box(real_t L) { setL(L); }
  inline void setL(std::array<real_t, 3> L);
  inline void setL(real_t L);
  inline const std::array<real_t, 3>& getL() const { return m_L; }
  /// Return precomputed 1/L[k] for each dimension.
  inline const std::array<real_t, 3>& getInvL() const { return m_invL; }
  virtual void makeShortestDistance(std::array<real_t, 3>& pos) const;
  //    inline void makeShortestDistance(std::array<real_t, 3> & pos, real_t shear) const;
  inline void putInBox(std::vector<std::array<real_t, 3> >& pos) const;

 protected:
  std::array<real_t, 3> m_L;
  std::array<real_t, 3> m_invL;  ///< precomputed 1/L[k] — avoids division in hot paths

 private:
  inline void computeInvL() {
    for (uint0 k = 0; k < 3; ++k)
      m_invL[k] = real_t(1) / m_L[k];
  }
};

template <typename real_t = float>
class BoxLE : public Box<real_t> {
 public:
  BoxLE() : m_shear(0) {}
  BoxLE(std::array<real_t, 3> L) : Box<real_t>(L), m_shear(0) {}
  BoxLE(real_t L) : Box<real_t>(L), m_shear(0) {}
  virtual void makeShortestDistance(std::array<real_t, 3>& pos) const;
  void setShear(real_t shear) { m_shear = shear; }
  void addShear(real_t dShear) { m_shear += dShear; }
  real_t getXShift() { return m_shear * this->m_L[1] / this->m_L[0]; }

 private:
  real_t m_shear;
};

/**
 * @class PosAndId
 * @brief struct for storing position and id of a particle
 * @tparam UInt integer type used to store id
 * @tparam real_t floating point type used to store pos
 */
template <typename UInt = uint2, typename real_t = float>
struct PosAndId {
 public:
  //! @brief position stored as a 3D array
  std::array<real_t, 3> pos;
  //! @brief id of particle
  UInt id;
};

template <typename UInt = uint2, typename real_t = float>
class NbrList {
 public:
  NbrList(Box<real_t>* box) : p_box(box) {}
  inline const Grid<UInt>& getGrid() const { return m_grid; }
  inline const Box<real_t>& getBox() const { return *p_box; }
  void setup(const std::vector<std::array<real_t, 3> >& pos, real_t rcut);
  void setupCurrentTeam(const std::vector<std::array<real_t, 3> >& pos, real_t rcut);
  inline UInt computeCellIndex(const std::array<real_t, 3>& pos) const;
  inline void getGridNbrs(const std::array<real_t, 3>& pos, std::vector<UInt>& nbrs) const;
  void getNbrs(UInt posIndx, const std::vector<std::array<real_t, 3> >& pos,
               std::vector<UInt>& indcs) const;
  inline bool empty(const UInt indxCell) const;
  inline void getCellContent(
      const UInt indxCell, typename std::vector<PosAndId<UInt, real_t> >::const_iterator& begin,
      typename std::vector<PosAndId<UInt, real_t> >::const_iterator& end) const;
  void clear() {
    m_headCell.clear();
    m_cell2Pos.clear();
  }

 private:
 Grid<UInt> m_grid;
  Box<real_t>* p_box;
  std::vector<UInt> m_headCell;
  std::vector<PosAndId<UInt, real_t> > m_cell2Pos;
  std::vector<UInt> m_teamScratchIndx;
  std::vector<UInt> m_teamScratchCounts;
};

template <typename UInt>
void Grid<UInt>::init(const Indx& n) {
  m_n = n;
  for (uint0 i(0); i < 3; ++i) {
    (m_n[i] == 0 ? m_n[i] = 1 : m_n[i]);
    m_kMax[i] = (m_n[i] > 2 ? 3 : (m_n[i] == 2 ? 2 : 1));
  }
  m_numCells = static_cast<UInt>(m_n[0]) * static_cast<UInt>(m_n[1]) * static_cast<UInt>(m_n[2]);
  m_numNbrs =
      static_cast<UInt>(m_kMax[0]) * static_cast<UInt>(m_kMax[1]) * static_cast<UInt>(m_kMax[2]);
}

template <typename UInt>
void Grid<UInt>::init(uint1 n) {
  Indx nTemp;
  for (unsigned i(0); i < 3; ++i)
    nTemp[i] = n;
  init(nTemp);
}

template <typename UInt>
void Grid<UInt>::expand(UInt indxCell, Indx& indx) const {
  std::array<UInt, 3> indx2;
  indx2[2] = indxCell;
  indx2[1] = indx2[2] / m_n[2];
  indx2[2] -= m_n[2] * indx2[1];
  indx2[0] = indx2[1] / m_n[1];
  indx2[1] -= m_n[1] * indx2[0];
  indx[0] = static_cast<uint1>(indx2[0]);
  indx[1] = static_cast<uint1>(indx2[1]);
  indx[2] = static_cast<uint1>(indx2[2]);
}

template <typename UInt>
Indx Grid<UInt>::expand(UInt indxCell) const {
  Indx indx;
  expand(indxCell, indx);
  return indx;
}

template <typename UInt>
UInt Grid<UInt>::compress(const Indx& indx) const {
  UInt indxCell = indx[0];
  indxCell *= m_n[1];
  indxCell += indx[1];
  indxCell *= m_n[2];
  indxCell += indx[2];
  return indxCell;
}

template <typename UInt>
void Grid<UInt>::getNbrs(UInt indxCell, std::vector<UInt>& nbrs) const {
  Indx indx;
  expand(indxCell, indx);
  std::array<Indx, 2> indxNbr;
  // k=0
  for (uint0 k(0); k < 3; ++k) {
    indxNbr[0][k] = ((indx[k] + 1) == m_n[k] ? 0 : indx[k] + 1);
    indxNbr[1][k] = (indx[k] == 0 ? m_n[k] - 1 : indx[k] - 1);
  }
  nbrs.clear();
  nbrs.reserve(26);
  Indx indx2;
  UInt indxCell2;

  indx2 = indx;
  indx2[0] = indxNbr[0][0];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[1][0];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);

  indx2 = indx;
  indx2[1] = indxNbr[0][1];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[1] = indxNbr[1][1];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);

  indx2 = indx;
  indx2[2] = indxNbr[0][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[2] = indxNbr[1][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);

  indx2 = indx;
  indx2[0] = indxNbr[0][0];
  indx2[1] = indxNbr[0][1];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[0][0];
  indx2[1] = indxNbr[1][1];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[1][0];
  indx2[1] = indxNbr[0][1];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[1][0];
  indx2[1] = indxNbr[1][1];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);

  indx2 = indx;
  indx2[1] = indxNbr[0][1];
  indx2[2] = indxNbr[0][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[1] = indxNbr[0][1];
  indx2[2] = indxNbr[1][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[1] = indxNbr[1][1];
  indx2[2] = indxNbr[0][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[1] = indxNbr[1][1];
  indx2[2] = indxNbr[1][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);

  indx2 = indx;
  indx2[2] = indxNbr[0][2];
  indx2[0] = indxNbr[0][0];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[2] = indxNbr[0][2];
  indx2[0] = indxNbr[1][0];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[2] = indxNbr[1][2];
  indx2[0] = indxNbr[0][0];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[2] = indxNbr[1][2];
  indx2[0] = indxNbr[1][0];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);

  indx2[0] = indxNbr[0][0];
  indx2[1] = indxNbr[0][1];
  indx2[2] = indxNbr[0][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[1][0];
  indx2[1] = indxNbr[0][1];
  indx2[2] = indxNbr[0][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[0][0];
  indx2[1] = indxNbr[1][1];
  indx2[2] = indxNbr[0][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[0][0];
  indx2[1] = indxNbr[0][1];
  indx2[2] = indxNbr[1][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[1][0];
  indx2[1] = indxNbr[1][1];
  indx2[2] = indxNbr[1][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[0][0];
  indx2[1] = indxNbr[1][1];
  indx2[2] = indxNbr[1][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[1][0];
  indx2[1] = indxNbr[0][1];
  indx2[2] = indxNbr[1][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
  indx2[0] = indxNbr[1][0];
  indx2[1] = indxNbr[1][1];
  indx2[2] = indxNbr[0][2];
  indxCell2 = compress(indx2);
  nbrs.push_back(indxCell2);
}

template <typename UInt>
void Grid<UInt>::getDirectNbrs(UInt indxCell, std::vector<UInt>& nbrs) const {
  Indx indx;
  expand(indxCell, indx);
  // k=0
  nbrs.clear();
  nbrs.reserve(6);
  Indx indxNbr;
  for (uint0 k(0); k < 3; ++k) {
    indxNbr = indx;
    switch (m_n[k]) {
      case 1:
        break;
      case 2:
        indxNbr[k] = (indx[k] == 0 ? 1 : 0);
        nbrs.push_back(compress(indxNbr));
        break;
      default:
        indxNbr[k] = ((indx[k] + 1) == m_n[k] ? 0 : indx[k] + 1);
        nbrs.push_back(compress(indxNbr));
        indxNbr[k] = (indx[k] == 0 ? m_n[k] - 1 : indx[k] - 1);
        nbrs.push_back(compress(indxNbr));
    };
  }
}

template <typename real_t>
void Box<real_t>::setL(std::array<real_t, 3> L) {
  m_L = L;
  for (uint0 k = 0; k < 3; ++k)
    m_invL[k] = real_t(1) / m_L[k];
}

template <typename real_t>
void Box<real_t>::setL(real_t L) {
  for (uint0 i(0); i < 3; ++i) {
    m_L[i] = L;
    m_invL[i] = real_t(1) / L;
  }
}

template <typename real_t>
void Box<real_t>::makeShortestDistance(std::array<real_t, 3>& pos) const {
  static constexpr real_t half = real_t(0.5);
  for (uint1 k(0); k < 3; ++k) {
    real_t r = pos[k] * m_invL[k];
    real_t s = r + half;
    pos[k] = (r - fastFloor(s)) * m_L[k];
  }
}

template <typename real_t>
void Box<real_t>::putInBox(std::vector<std::array<real_t, 3> >& pos) const {
#pragma omp parallel for
  for (size_t i = 0; i < pos.size(); ++i)
    for (uint1 k(0); k < 3; ++k) {
      real_t r(pos[i][k] * m_invL[k]);
      r -= floor(r);
      pos[i][k] = r * m_L[k];
    }
}

template <typename real_t>
void BoxLE<real_t>::makeShortestDistance(std::array<real_t, 3>& pos) const {
  static constexpr real_t half = real_t(0.5);
  real_t fl1 = fastFloor(pos[1] * this->m_invL[1] + half);
  real_t pos0(pos[0] - fl1 * (m_shear * this->m_L[1]));
  pos[0] = this->m_L[0] * (pos0 * this->m_invL[0] - fastFloor(pos0 * this->m_invL[0] + half));
  pos[1] = this->m_L[1] * (pos[1] * this->m_invL[1] - fl1);
  pos[2] = this->m_L[2] * (pos[2] * this->m_invL[2] - fastFloor(pos[2] * this->m_invL[2] + half));
}

template <typename UInt, typename real_t>
UInt NbrList<UInt, real_t>::computeCellIndex(const std::array<real_t, 3>& pos) const {
  UInt indx(0);
  const std::array<real_t, 3>& L(p_box->getL());
  const std::array<real_t, 3>& invL(p_box->getInvL());
  for (uint0 k(0); k < 3; ++k) {
    real_t r(pos[k] * invL[k]);
    r -= floor(r);
    indx *= static_cast<UInt>(m_grid.getN()[k]);
    indx += static_cast<UInt>(floor(r * m_grid.getN()[k]));
  }
  return indx;
}

template <typename UInt, typename real_t>
void NbrList<UInt, real_t>::getGridNbrs(const std::array<real_t, 3>& pos,
                                        std::vector<UInt>& nbrs) const {
  std::array<Indx, 2> indcs;
  const std::array<real_t, 3>& L(p_box->getL());
  const std::array<real_t, 3>& invL(p_box->getInvL());
  for (uint0 k(0); k < 3; ++k) {
    real_t r(pos[k] * invL[k]);
    r -= floor(r);
    real_t indxR(r * m_grid.getN()[k]);
    real_t indxRFl = floor(indxR);
    indcs[0][k] = static_cast<UInt>(indxRFl);
    if (indxR < indxRFl + real_t(0.5))
      indcs[1][k] = (indcs[0][k] == 0 ? m_grid.getN()[k] - 1 : indcs[0][k] - 1);
    else
      indcs[1][k] = ((indcs[0][k] + 1) == m_grid.getN()[k] ? 0 : indcs[0][k] + 1);
  }
  nbrs.clear();
  nbrs.reserve(8);
  Indx indx;
  UInt indxCell;
  indxCell = m_grid.compress(indcs[0]);
  nbrs.push_back(indxCell);
  indx = indcs[0];
  indx[0] = indcs[1][0];
  indxCell = m_grid.compress(indx);
  nbrs.push_back(indxCell);
  indx = indcs[0];
  indx[1] = indcs[1][1];
  indxCell = m_grid.compress(indx);
  nbrs.push_back(indxCell);
  indx = indcs[0];
  indx[2] = indcs[1][2];
  indxCell = m_grid.compress(indx);
  nbrs.push_back(indxCell);
  indx = indcs[1];
  indx[0] = indcs[0][0];
  indxCell = m_grid.compress(indx);
  nbrs.push_back(indxCell);
  indx = indcs[1];
  indx[1] = indcs[0][1];
  indxCell = m_grid.compress(indx);
  nbrs.push_back(indxCell);
  indx = indcs[1];
  indx[2] = indcs[0][2];
  indxCell = m_grid.compress(indx);
  nbrs.push_back(indxCell);
  indxCell = m_grid.compress(indcs[1]);
  nbrs.push_back(indxCell);
}

template <typename UInt, typename real_t>
void NbrList<UInt, real_t>::setup(const std::vector<std::array<real_t, 3> >& pos, real_t rcut) {
  Indx n;
  const std::array<real_t, 3>& L(p_box->getL());
  const std::array<real_t, 3>& invL(p_box->getInvL());
  for (uint0 i(0); i < 3; ++i) {
    n[i] = static_cast<uint2>(floor(L[i] / rcut));
  }
  m_grid.init(n);
  m_headCell.clear();
  std::vector<UInt> indx(pos.size());
  const size_t numPos = pos.size();
  const size_t numCells = static_cast<size_t>(m_grid.numCells());
#ifdef VORONOI_USE_OPENMP
  const int numThreads = omp_get_max_threads();
#else
  const int numThreads = 1;
#endif

  if (numThreads <= 1) {
    for (UInt i = 0; i < pos.size(); ++i)
      indx[i] = computeCellIndex(pos[i]);

    std::vector<UInt> counts(numCells, 0);
    for (UInt i = 0; i < pos.size(); ++i)
      ++counts[indx[i]];

    m_headCell.resize(counts.size() + 1);
    m_headCell[0] = 0;
    std::partial_sum(counts.begin(), counts.end(), m_headCell.begin() + 1);

    counts = m_headCell;
    m_cell2Pos.resize(indx.size());
    for (UInt i = 0; i < indx.size(); ++i) {
      const UInt head = counts[indx[i]]++;
      m_cell2Pos[head].id = i;
      m_cell2Pos[head].pos = pos[i];
      for (uint0 k(0); k < 3; ++k)
        m_cell2Pos[head].pos[k] -= L[k] * floor(m_cell2Pos[head].pos[k] * invL[k]);
    }
    return;
  }

#ifdef VORONOI_USE_OPENMP
#pragma omp parallel for
  for (UInt i = 0; i < pos.size(); ++i)
    indx[i] = computeCellIndex(pos[i]);

  std::vector<UInt> localCounts(static_cast<size_t>(numThreads) * numCells, 0);
#pragma omp parallel num_threads(numThreads)
  {
    const int tid = omp_get_thread_num();
    UInt* const counts = localCounts.data() + static_cast<size_t>(tid) * numCells;
    const size_t begin = (numPos * static_cast<size_t>(tid)) / static_cast<size_t>(numThreads);
    const size_t end = (numPos * static_cast<size_t>(tid + 1)) / static_cast<size_t>(numThreads);
    for (size_t i = begin; i < end; ++i)
      ++counts[indx[i]];
  }

  std::vector<UInt> counts(numCells, 0);
  for (size_t cell = 0; cell < numCells; ++cell) {
    UInt total = 0;
    for (int tid = 0; tid < numThreads; ++tid)
      total += localCounts[static_cast<size_t>(tid) * numCells + cell];
    counts[cell] = total;
  }

  m_headCell.resize(counts.size() + 1);
  m_headCell[0] = 0;
  std::partial_sum(counts.begin(), counts.end(), m_headCell.begin() + 1);

  for (size_t cell = 0; cell < numCells; ++cell) {
    UInt head = m_headCell[cell];
    for (int tid = 0; tid < numThreads; ++tid) {
      UInt& offset = localCounts[static_cast<size_t>(tid) * numCells + cell];
      const UInt count = offset;
      offset = head;
      head += count;
    }
  }

  m_cell2Pos.resize(indx.size());
#pragma omp parallel num_threads(numThreads)
  {
    const int tid = omp_get_thread_num();
    UInt* const offsets = localCounts.data() + static_cast<size_t>(tid) * numCells;
    const size_t begin = (numPos * static_cast<size_t>(tid)) / static_cast<size_t>(numThreads);
    const size_t end = (numPos * static_cast<size_t>(tid + 1)) / static_cast<size_t>(numThreads);
    for (size_t i = begin; i < end; ++i) {
      const UInt head = offsets[indx[i]]++;
      m_cell2Pos[head].id = static_cast<UInt>(i);
      m_cell2Pos[head].pos = pos[i];
      for (uint0 k(0); k < 3; ++k)
        m_cell2Pos[head].pos[k] -= L[k] * floor(m_cell2Pos[head].pos[k] * invL[k]);
    }
  }
#else
  for (UInt i = 0; i < pos.size(); ++i)
    indx[i] = computeCellIndex(pos[i]);
#endif
  // #pragma omp parallel for
  //     for(UInt i=0; i < m_headCell.size()-1; ++i){
  //       std::sort(m_cell2Pos.begin() + m_headCell[i], m_cell2Pos.begin() + m_headCell[i+1]);
  //     }
  //     std::array<real_t, 3> dL;
  //     for(uint0 k(0); k<3; ++k)
  //       dL[k] = L[k]/(static_cast<real_t >(n[k]));
  // #pragma omp parallel for
  //     for(UInt i=0; i < m_grid.numCells(); ++i){
  //       n = m_grid.expand(i);
  //       std::array<real_t, 3> orig;
  //       for(uint0 k(0); k<3; ++k)
  // 	orig[k] = static_cast<real_t >(n[k]) * dL[k];
  //       for(UInt j= m_headCell[i]; j < m_headCell[i+1]; ++j)
  // 	for(uint0 k(0); k<3; ++k){
  // 	  m_cell2Pos[j].pos[k] -= L[k]*floor(m_cell2Pos[j].pos[k]/L[k]);
  // 	  m_cell2Pos[j].pos[k] - =orig[k];
  // 	}
  //     }
}

template <typename UInt, typename real_t>
void NbrList<UInt, real_t>::setupCurrentTeam(const std::vector<std::array<real_t, 3> >& pos,
                                             real_t rcut) {
#ifndef VORONOI_USE_OPENMP
  setup(pos, rcut);
#else
  if (!omp_in_parallel()) {
    setup(pos, rcut);
    return;
  }

  const int numThreads = omp_get_num_threads();
  const int tid = omp_get_thread_num();
  const size_t numPos = pos.size();
  const std::array<real_t, 3>& L(p_box->getL());
  const std::array<real_t, 3>& invL(p_box->getInvL());
  size_t numCells = 0;

#pragma omp single
  {
    Indx n;
    for (uint0 i(0); i < 3; ++i)
      n[i] = static_cast<uint2>(floor(L[i] / rcut));
    m_grid.init(n);
    m_headCell.clear();
    m_cell2Pos.clear();
    numCells = static_cast<size_t>(m_grid.numCells());
    m_teamScratchIndx.resize(numPos);
    m_teamScratchCounts.assign(static_cast<size_t>(numThreads) * numCells, 0);
    m_headCell.resize(numCells + 1);
    m_cell2Pos.resize(numPos);
  }

#pragma omp barrier

  numCells = static_cast<size_t>(m_grid.numCells());
  std::vector<UInt>& indx = m_teamScratchIndx;
  std::vector<UInt>& localCounts = m_teamScratchCounts;

#pragma omp for nowait
  for (UInt i = 0; i < pos.size(); ++i)
    indx[i] = computeCellIndex(pos[i]);

  UInt* const counts = localCounts.data() + static_cast<size_t>(tid) * numCells;
  const size_t begin = (numPos * static_cast<size_t>(tid)) / static_cast<size_t>(numThreads);
  const size_t end = (numPos * static_cast<size_t>(tid + 1)) / static_cast<size_t>(numThreads);
  for (size_t i = begin; i < end; ++i)
    ++counts[indx[i]];

#pragma omp barrier

#pragma omp single
  {
    std::vector<UInt> countsPerCell(numCells, 0);
    for (size_t cell = 0; cell < numCells; ++cell) {
      UInt total = 0;
      for (int t = 0; t < numThreads; ++t)
        total += localCounts[static_cast<size_t>(t) * numCells + cell];
      countsPerCell[cell] = total;
    }

    m_headCell[0] = 0;
    std::partial_sum(countsPerCell.begin(), countsPerCell.end(), m_headCell.begin() + 1);

    for (size_t cell = 0; cell < numCells; ++cell) {
      UInt head = m_headCell[cell];
      for (int t = 0; t < numThreads; ++t) {
        UInt& offset = localCounts[static_cast<size_t>(t) * numCells + cell];
        const UInt count = offset;
        offset = head;
        head += count;
      }
    }
  }

#pragma omp barrier

  UInt* const offsets = localCounts.data() + static_cast<size_t>(tid) * numCells;
  for (size_t i = begin; i < end; ++i) {
    const UInt head = offsets[indx[i]]++;
    m_cell2Pos[head].id = static_cast<UInt>(i);
    m_cell2Pos[head].pos = pos[i];
    for (uint0 k(0); k < 3; ++k)
      m_cell2Pos[head].pos[k] -= L[k] * floor(m_cell2Pos[head].pos[k] * invL[k]);
  }

#pragma omp barrier
#endif
}

template <typename UInt, typename real_t>
bool NbrList<UInt, real_t>::empty(const UInt indxCell) const {
  return (m_headCell[indxCell] == m_headCell[indxCell + 1]);
}

template <typename UInt, typename real_t>
void NbrList<UInt, real_t>::getCellContent(
    const UInt indxCell, typename std::vector<PosAndId<UInt, real_t> >::const_iterator& begin,
    typename std::vector<PosAndId<UInt, real_t> >::const_iterator& end) const {
  begin = m_cell2Pos.begin() + m_headCell[indxCell];
  end = m_cell2Pos.begin() + m_headCell[indxCell + 1];
}

template <typename UInt, typename real_t>
void NbrList<UInt, real_t>::getNbrs(const UInt indxPos, const std::vector<std::array<real_t, 3> >& pos,
                                    std::vector<UInt>& indcs) const {
  UInt indxCell(computeCellIndex(pos[indxPos]));
  std::vector<UInt> nbrCells;
  m_grid.getNbrs(indxCell, nbrCells);
  UInt numNbrs(m_headCell[indxCell + 1] - m_headCell[indxCell] - 1);
  for (UInt k(0); k < nbrCells.size(); ++k)
    numNbrs += m_headCell[nbrCells[k] + 1] - m_headCell[nbrCells[k]];
  indcs.clear();
  indcs.reserve(numNbrs);
  for (UInt i(m_headCell[indxCell]); i < m_headCell[indxCell + 1]; ++i)
    if (m_cell2Pos[i].id != indxPos)
      indcs.push_back(m_cell2Pos[i].id);
  for (UInt k(0); k < nbrCells.size(); ++k)
    for (UInt i(m_headCell[nbrCells[k]]); i < m_headCell[nbrCells[k] + 1]; ++i)
      indcs.push_back(m_cell2Pos[i].id);
}
}  // namespace vor
