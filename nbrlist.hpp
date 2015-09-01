#ifndef VOR_NBRLIST_H
#define VOR_NBRLIST_H

#include<vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <math.h>
#include "vor_types.hpp"

using std::vector;
using std::string;
using std::sort;
using vor::Array;
using vor::uint0;
using vor::uint1;
using vor::uint2;
using std::partial_sum;

namespace vor
{
  typedef Array<uint1, 3> Indx;

  template< typename UInt>
  class Grid
  {
  public:
    void init(const Indx & n);
    void init(uint1 n);
    inline const Indx & getN() const {return m_n;}
    inline UInt numCells() const {return m_numCells;}
    inline Indx expand(UInt indxCell) const;
    inline void expand(UInt indxCell, Indx & indx) const;
    inline UInt compress(const Indx & indx) const;
    inline void getNbrs(UInt indxCell, vector<UInt> & ngbrs) const;
    inline void getDirectNbrs(UInt indxCell, vector<UInt> & ngbrs) const;
  private:
    Indx m_n;
    Indx m_kMax;
    UInt m_numCells;
    UInt m_numNbrs;
  };

  template<typename real_t = float>
  class Box
  {
  public:
    Box(Array<real_t, 3> L);
    Box(real_t L);
    inline Array<real_t, 3> & getL() {return m_L;}
    inline const Array<real_t, 3> & getL() const {return m_L;}
    virtual void makeShortestDistance(Array<real_t, 3> & pos) const;
    //    inline void makeShortestDistance(Array<real_t, 3> & pos, real_t shear) const;
    inline void putInBox(vector<Array<real_t, 3> > & pos) const;
  protected:
    Array<real_t, 3> m_L;
  };

  template<typename real_t = float>
  class BoxLE: public Box<real_t>
  {
  public:
    BoxLE(Array<real_t, 3> L): Box<real_t>(L), m_shear(0) {}
    BoxLE(real_t L): Box<real_t>(L), m_shear(0) {}
    virtual void makeShortestDistance(Array<real_t, 3> & pos) const;
    void setShear(real_t shear) {m_shear = shear;}
    void addShear(real_t dShear) {m_shear += dShear;}
    real_t getXShift() {return m_shear*this->m_L[1]/this->m_L[0];}
  private:
    real_t m_shear;
  };
  
  template<typename UInt = uint2, typename real_t = float>
  struct PosAndId
  {
  public:
    Array<real_t, 3> pos;
    UInt id;
  };

  template<typename UInt = uint2, typename real_t = float>
  class NbrList
  {
  public:
    NbrList(Box<real_t> * box):p_box(box) {}
    inline const Grid<UInt> & getGrid() const {return m_grid;}
    inline const Box<real_t> & getBox() const {return *p_box;}
    void setup(const vector<Array<real_t, 3> > & pos, real_t rcut);
    inline UInt computeCellIndex(const Array<real_t, 3> & pos) const;
    inline void getGridNbrs(const Array<real_t, 3> & pos, vector<UInt> & nbrs) const;
    void getNbrs(UInt posIndx, const vector<Array<real_t, 3> > & pos, vector<UInt> & indcs) const;
    inline bool empty(const UInt indxCell) const;
    inline void getCellContent(const UInt indxCell, typename vector<PosAndId<UInt, real_t> >::const_iterator & begin, typename vector<PosAndId<UInt, real_t> >::const_iterator & end) const;
    void clear() {m_headCell.clear(); m_cell2Pos.clear();}
  private:
    Grid<UInt> m_grid;
    Box<real_t> * p_box;
    vector<UInt> m_headCell;
    vector<PosAndId<UInt, real_t> > m_cell2Pos;
  };

  template< typename UInt>
  void Grid<UInt>::init(const Indx & n)
  {
    m_n = n;
    for(uint0 i(0); i<3; ++i){
      (m_n[i] == 0? m_n[i]=1 : m_n[i]);
      m_kMax[i] = ( m_n[i] > 2 ? 3 : ( m_n[i]==2 ? 2 : 1));
    }
    m_numCells = static_cast<UInt>(m_n[0])*static_cast<UInt>(m_n[1])*static_cast<UInt>(m_n[2]);
    m_numNbrs = static_cast<UInt>(m_kMax[0])*static_cast<UInt>(m_kMax[1])*static_cast<UInt>(m_kMax[2]);
  }

  template< typename UInt>
  void Grid<UInt>::init(uint1 n)
  {
    Indx nTemp;
    for(unsigned i(0); i<3; ++i) nTemp[i] = n;
    init(nTemp);
  }

  template< typename UInt>
  void Grid<UInt>::expand(UInt indxCell, Indx & indx) const
  {
    Array<UInt, 3> indx2;
    indx2[2] = indxCell;
    indx2[1] = indx2[2]/m_n[2];
    indx2[2] -=  m_n[2]*indx2[1];
    indx2[0] = indx2[1]/m_n[1];
    indx2[1] -=  m_n[1]*indx2[0];
    indx[0] = static_cast<uint1>(indx2[0]);
    indx[1] = static_cast<uint1>(indx2[1]);
    indx[2] = static_cast<uint1>(indx2[2]);
  }

  template< typename UInt>
  Indx Grid<UInt>::expand(UInt indxCell) const
  {
    Indx indx;
    expand(indxCell, indx);
    return indx;
  }

  template< typename UInt>
  UInt Grid<UInt>::compress(const Indx & indx) const
  {
    UInt indxCell = indx[0];
    indxCell *= m_n[1];
    indxCell += indx[1];
    indxCell *= m_n[2];
    indxCell += indx[2];
    return indxCell;
  }

  template< typename UInt>
  void Grid<UInt>::getNbrs(UInt indxCell, vector<UInt> & nbrs) const
  {
    Indx indx;
    expand(indxCell, indx);
    Array<Indx, 2> indxNbr;
    // k=0
    for(uint0 k(0); k<3; ++k){
      indxNbr[0][k] = ((indx[k]+1) == m_n[k] ? 0 : indx[k]+1);
      indxNbr[1][k] = (indx[k] == 0? m_n[k]-1 : indx[k]-1);
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

  template< typename UInt>
  void Grid<UInt>::getDirectNbrs(UInt indxCell, vector<UInt> & nbrs) const
  {
    Indx indx;
    expand(indxCell, indx);
    // k=0
    nbrs.clear();
    nbrs.reserve(6);
    Indx indxNbr;
    for(uint0 k(0); k<3; ++k){
      indxNbr = indx;
      switch (m_n[k]){
      case 1:
	break;
      case 2:
	indxNbr[k] = (indx[k] == 0 ? 1 : 0);
	nbrs.push_back(compress(indxNbr));
	break;
      default:
	indxNbr[k] = ((indx[k]+1) == m_n[k] ? 0 : indx[k]+1);
	nbrs.push_back(compress(indxNbr));
	indxNbr[k]=(indx[k] == 0? m_n[k]-1 : indx[k]-1);
	nbrs.push_back(compress(indxNbr));
      };
    }
  }

  template<typename real_t>
  Box<real_t>::Box(const Array<real_t, 3> L) : m_L(L){}
  
  template<typename real_t>
  Box<real_t>::Box(real_t L)
  {
    for(uint0 i(0); i<3; ++i)
      m_L[i] = L;
  }

  template<typename real_t>
  void Box<real_t>::makeShortestDistance(Array<real_t, 3> & pos) const
  {
    for(uint1 k(0); k<3; ++k){
      real_t r(pos[k]/m_L[k]);
      r -= floor(r+0.5);
      pos[k] = r*m_L[k];
    }    
  }

  template<typename real_t>
  void Box<real_t>::putInBox(vector<Array<real_t, 3> > & pos) const
  {
#pragma omp parallel for
    for(size_t i=0; i< pos.size(); ++i)
      for(uint1 k(0); k<3; ++k){	
	real_t r(pos[i][k]/m_L[k]);
	r -= floor(r);
	pos[i][k] = r*m_L[k];
      }    
  }

  template<typename real_t>
  void BoxLE<real_t>::makeShortestDistance(Array<real_t, 3> & pos) const
  {
    real_t fl1(floor(pos[1]/this->m_L[1] + 0.5 ));
    real_t pos0(pos[0] - fl1*(m_shear*this->m_L[1]));
    pos[0] = this->m_L[0]*(pos0 - floor(pos0/this->m_L[0] + 0.5 ));
    pos[1] = this->m_L[1]*(pos[1] - fl1) ;
    pos[2] = this->m_L[2]*(pos[2] - floor(pos[2]/this->m_L[2] + 0.5 )) ;
  }

  template< typename UInt, typename real_t>
  UInt NbrList<UInt, real_t>::computeCellIndex(const Array<real_t, 3> & pos) const
  {
    UInt indx(0);
    const Array<real_t, 3> & L(p_box->getL());
    for(uint0 k(0); k<3; ++k){
      real_t r(pos[k]/L[k]);
      r -= floor(r);
      indx *= static_cast<UInt>(m_grid.getN()[k]);
      indx += static_cast<UInt>(floor(r*m_grid.getN()[k]));
    }
    return indx;
  }

  template< typename UInt, typename real_t>
  void NbrList<UInt, real_t>::getGridNbrs(const Array<real_t, 3> & pos, vector<UInt> & nbrs) const
  {
    Array<Indx, 2> indcs;
    const Array<real_t, 3> & L(p_box->getL());
    for(uint0 k(0); k<3; ++k){
      real_t r(pos[k]/L[k]);
      r -= floor(r);
      real_t indxR(r*m_grid.getN()[k]);
      real_t indxRFl = floor(indxR);
      indcs[0][k]= static_cast<UInt>(indxRFl);
      if (indxR < indxRFl + 0.5)
	indcs[1][k]= (indcs[0][k] == 0? m_grid.getN()[k]-1 : indcs[0][k]-1);
      else
	indcs[1][k]= ((indcs[0][k]+1) == m_grid.getN()[k] ? 0 : indcs[0][k]+1);
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

  template< typename UInt, typename real_t>
  void NbrList<UInt, real_t>::setup(const vector<Array<real_t, 3> > & pos, real_t rcut)
  {
    Indx n;
    const Array<real_t, 3> & L(p_box->getL());
    for(uint0 i(0); i<3; ++i){
      n[i] = static_cast<uint2>(floor(L[i]/rcut));
    }
    m_grid.init(n);
    m_headCell.clear();
    vector<UInt> indx(pos.size());
#pragma omp parallel for
    for(UInt i=0; i < pos.size(); ++i)
      indx[i] = computeCellIndex(pos[i]);
    vector<UInt> temp(m_grid.numCells(),0);
#pragma omp parallel for
    for(UInt i=0; i < pos.size(); ++i){
#pragma omp atomic
      ++temp[indx[i]];
    }
    m_headCell.resize(temp.size()+1);
    m_headCell[0] = 0;
    partial_sum(temp.begin(), temp.end(), m_headCell.begin()+1);
    temp = m_headCell;
    m_cell2Pos.resize(indx.size());
#pragma omp parallel for
    for(UInt i=0; i < indx.size(); ++i){
      UInt head;
#pragma omp atomic capture
      head = temp[indx[i]]++;
      m_cell2Pos[ head ].id = i;
      m_cell2Pos[ head ].pos = pos[i];
      for(uint0 k(0); k<3; ++k)
	m_cell2Pos[ head ].pos[k] -= L[k]*floor(m_cell2Pos[head].pos[k]/L[k]);
    }
// #pragma omp parallel for
//     for(UInt i=0; i < m_headCell.size()-1; ++i){
//       sort(m_cell2Pos.begin() + m_headCell[i], m_cell2Pos.begin() + m_headCell[i+1]);
//     }
//     Array<real_t, 3> dL;
//     for(uint0 k(0); k<3; ++k)
//       dL[k] = L[k]/(static_cast<real_t >(n[k]));
// #pragma omp parallel for
//     for(UInt i=0; i < m_grid.numCells(); ++i){
//       n = m_grid.expand(i);
//       Array<real_t, 3> orig;
//       for(uint0 k(0); k<3; ++k)
// 	orig[k] = static_cast<real_t >(n[k]) * dL[k];
//       for(UInt j= m_headCell[i]; j < m_headCell[i+1]; ++j) 
// 	for(uint0 k(0); k<3; ++k){
// 	  m_cell2Pos[j].pos[k] -= L[k]*floor(m_cell2Pos[j].pos[k]/L[k]);
// 	  m_cell2Pos[j].pos[k] - =orig[k];
// 	}
//     }
  }

  template< typename UInt, typename real_t>
  bool NbrList<UInt, real_t>::empty(const UInt indxCell) const
  {
    return (m_headCell[indxCell] == m_headCell[indxCell+1]);
  }

  template< typename UInt, typename real_t>
  void NbrList<UInt, real_t>::getCellContent(const UInt indxCell, typename vector<PosAndId<UInt, real_t> >::const_iterator & begin, typename vector<PosAndId<UInt, real_t> >::const_iterator & end) const
  {
    begin = m_cell2Pos.begin() + m_headCell[indxCell];
    end = m_cell2Pos.begin() + m_headCell[indxCell+1];
  }

  template< typename UInt, typename real_t>
  void NbrList<UInt, real_t>::getNbrs(const UInt indxPos, const vector<Array<real_t, 3> > & pos, vector<UInt> & indcs) const
  {
    UInt indxCell(computeCellIndex(pos[indxPos]));
    vector<UInt> nbrCells;
    m_grid.getNbrs(indxCell, nbrCells);
    UInt numNbrs(m_headCell[indxCell+1]-m_headCell[indxCell]-1);
    for(UInt k(0); k < nbrCells.size(); ++k)
      numNbrs += m_headCell[nbrCells[k]+1]-m_headCell[nbrCells[k]];
    indcs.clear();
    indcs.reserve(numNbrs);
    for(UInt i(m_headCell[indxCell]); i < m_headCell[indxCell+1]; ++i)
      if (m_cell2Pos[i] != indxPos)
	indcs.push_back(m_cell2Pos[i]);
    for(UInt k(0); k < nbrCells.size(); ++k)
      for(UInt i(m_headCell[nbrCells[k]]); i < m_headCell[nbrCells[k]+1]; ++i)
	indcs.push_back(m_cell2Pos[i]);
  }  
}
#endif
