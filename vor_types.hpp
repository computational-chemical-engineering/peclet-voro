#ifndef VOR_TYPES_H
#define VOR_TYPES_H

#include <inttypes.h>
#include <vector>
#include <utility>

using std::vector;
using std::pair;

namespace vor {

  typedef uint8_t uint0;
  typedef uint16_t uint1;
  typedef uint32_t uint2;

  static const uint1 shiftFacet(9);
  static const uint1 maskFacet( (~0) << shiftFacet);
  static const uint1 maskNoFacet((uint1) ~maskFacet);
  static const uint1 maskEdge(3);
  static const uint1 maskVertex((uint1) ~(maskFacet | maskEdge));
  static const uint2 noNbr(~0);
  static const uint1 maxNumVertices(1<<7);
  static const uint1 maxNumFacets(1<<7);

  template<typename T, unsigned int n>
  class Array
  {
  public:
    inline T & operator[](unsigned int i)
    {
      return m[i];
    }
    inline const T & operator[](unsigned int i) const
    {
      return m[i];
    }
  private:
    T m[n];
  };

  typedef Array<uint1, 3> Vertex;
  typedef Array<uint2, 3> NbrInsert;

  typedef vector<NbrInsert>::const_iterator NbrInsertItr;

  template<typename real_t>
  class NbrDist: public Array<real_t, 3>
  {
  public:
    uint2 id;
    real_t rSq;
  };

  template<typename real_t>
  class CoordMatrix
  {
    uint2 i,j;
    real_t value;
  };

  template<typename real_t>
  class CompareCoordMatrix
  {
  public:
    inline bool operator()(const CoordMatrix<real_t> & a, const CoordMatrix<real_t> & b) const
    {
      return (a.i==b.i ? a.j < b.j : a.i < b.i);
    }
  };


  class ComparePairFirst
  {
  public:
    template<typename T1, typename T2>
    inline bool operator()(pair<T1, T2> a, pair<T1, T2> b) const
    {
      return a.first < b.first;
    }
  };


  template<typename real_t>
  class CompareNbrDist
  {
  public:
    inline bool operator()(const NbrDist<real_t> & a, const NbrDist<real_t> & b) const
    {
      return a.rSq < b.rSq;
    }
  };

  class CompareNbrInsert
  {
  public:
    inline bool operator()(const NbrInsert & a, const NbrInsert & b) const
    {
      return a[0] < b[0];
    }
  };

}

#endif
