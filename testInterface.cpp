/**
 * \file testSimulation.cpp
 * \brief test program: test a simple simulation of compressible Euler equations
 *
 */
//#include <omp.h>
#include <cstdio>
#include <vector>
#include <boost/random.hpp>

#include "voronoi.hpp"
#include "simulation.hpp"

using vor::Array;
using vor::ExplicitEuler;

namespace vor {
  
  template<typename real_t = float>
  class IntfDyn: public ExplicitEuler<real_t>
  {
  public:
    IntfDyn(): ExplicitEuler<real_t>(), m_intfTension(0) {}
    void setIntfTension(real_t intfTension) {m_intfTension = intfTension;}
    real_t getIntfEnergy() const;
  private:
    virtual void computeForces();
    void computeIntfForces();
    real_t m_intfTension;  
  };

    
  template<typename real_t>
  void IntfDyn<real_t>::computeForces()
  {
    this->ExplicitEuler<real_t>::computeForces();
    computeIntfForces();
  }

  template<typename real_t>
  void IntfDyn<real_t>::computeIntfForces()
  {
    const uint0 iType=0, jType=1;
    const vector<uint0> & types(this->m_complex.getTypes());
    vector<uint2> indxFacets;
    vector<Array<real_t, 3> > grad;
#pragma omp parallel for
    for(size_t i=0; i< types.size(); ++i){
      if(types[i] == iType){
	const Cell<real_t> & cell(this->m_complex.getCells()[i]);
	const CellGeometry<real_t> & geom(this->m_complex.getGeoms()[i]);
	for(uint1 j=0; j< cell.numFacets(); ++j){
	  uint2 nbr = cell.getNbr(j);
	  if(types[nbr] == jType){
	    const Array<real_t, 3> & areaV(geom.getAreas()[j]);
	    real_t area= sqrt(areaV[0]*areaV[0]+areaV[1]*areaV[1]+areaV[2]*areaV[2]);
	    geom.gradFacetAreaSq(j, indxFacets, grad);
	    for(int m=0; m< indxFacets.size(); ++m){
	      uint2 nbr2 = cell.getNbr(indxFacets[m]);
	      for(uint0 k=0; k<3; ++k){
		real_t df = -m_intfTension*grad[m][k]/(2.0*area);
#pragma omp atomic
		this->m_forces[nbr2][k] += df;
#pragma omp atomic
		this->m_forces[i][k] -= df;
	      }
	    }
	  }
	}
      }
    }
  }

  template<typename real_t>
  real_t IntfDyn<real_t>::getIntfEnergy() const
  {
    real_t E = 0;
    const uint0 iType=0, jType=1;
    const vector<uint0> & types(this->m_complex.getTypes());
    vector<uint2> indxFacets;
    vector<Array<real_t, 3> > grad;
#pragma omp parallel for reduction(+:E)
    for(size_t i=0; i< types.size(); ++i){
      if(types[i] == iType){
	const Cell<real_t> & cell(this->m_complex.getCells()[i]);
	const CellGeometry<real_t> & geom(this->m_complex.getGeoms()[i]);
	for(uint1 j=0; j< cell.numFacets(); ++j){
	  uint2 nbr = cell.getNbr(j);
	  if(types[nbr] == jType){
	    //printf("%d: type %d, %d: type %d\n", i, types[i], nbr,types[nbr]);
	    const Array<real_t, 3> & areaV(geom.getAreas()[j]);
	    real_t area= sqrt(areaV[0]*areaV[0]+areaV[1]*areaV[1]+areaV[2]*areaV[2]);
	    E += m_intfTension*area;
	  }
	}
      }
    }
    return E;
  }
}

using vor::IntfDyn;

int main()
{
  typedef double real_t;
  //generate random particle positions
  Array<real_t, 3> L;
  L[0] = 1;
  L[1] = 1;
  L[2] = 1;
  IntfDyn<real_t> sim;
  sim.setL(L);
  sim.setPressure(1);
  sim.setMassDensity(1);
  sim.setIntfTension(1.e-2);
  real_t dt(0.01);
  {
    int numPart=10000;
    vector<Array<real_t, 3> > pos(numPart);
    vector<Array<real_t, 3> > vel(numPart);
    FILE *pFile;
    char filename[255];
    sprintf(filename, "pos_eq_%d.dat",numPart);
    pFile=fopen(filename,"r");
    int j;
    for(int i=0; i<pos.size(); ++i){
      int j, ierr;
      ierr = fscanf(pFile, "%d",&j);
      for(int k=0; k<3; ++k){
	double a;
    	ierr  =fscanf(pFile, "%lf", &a);
	if (ierr == EOF) break; 
	pos[j][k]=a;
      }
    }
    fclose(pFile);
    sim.getCellComplex().getNbrList().getBox().putInBox(pos);
    vector<uint0> types(numPart);
    vector<Array<real_t, 3> > pos2(numPart);
    size_t iBegin=0, iLast=numPart-1;
    real_t rSq = 0.09*L[1]*L[1];
    for(int i=0; i<pos.size(); ++i){
      bool inSphere = ((pos[i][0]-0.5)*(pos[i][0]-0.5)+(pos[i][1]-0.5)*(pos[i][1]-0.5)+(pos[i][2]-0.5)*(pos[i][2]-0.5) < rSq);
      if (inSphere){
	for(int k=0; k<3; ++k)
	  pos2[iBegin][k] = pos[i][k];
	types[iBegin] = 0;
	vel[iBegin][0] = 0;
	vel[iBegin][1] = 0;
	vel[iBegin][2] = 0;
	++iBegin;
      } else {
	for(int k=0; k<3; ++k)
	  pos2[iLast][k] = pos[i][k];
	types[iLast] = 1;
	vel[iLast][0] = 0;
	vel[iLast][1] = 0;
	vel[iLast][2] = 0;
	--iLast;
      }
    }
    //    printf("last particle of type0: %lu, first particle of type 1: %lu\n", iLast, iBegin);
    sim.setPositions(pos2);
    sim.setVelocities(vel);
    sim.setTypes(types);
  }
  if (!sim.init()){
    fprintf(stderr, "initialization simulation failed\n");
    return 1;
  }
  
  for(int m=0; m< 100; ++m){
    for(int i=0; i< 50; ++i){
      sim.step(5,dt);
      printf("%16.8g %16.8g %16.8g %16.8g\n", sim.getTime(), sim.getKineticEnergy(), sim.getInternalEnergy(), sim.getIntfEnergy());
      vector<Array<real_t,3> > & vel(sim.getVelocities());
#pragma omp parallel for
      for(int i=0; i<vel.size(); ++i)
      	for(int k=0; k<3; ++k)
      	  vel[i][k] *= 0.995;
    }
    const vector<Array<real_t,3> > & pos(sim.getPositions());
    //    sim.getCellComplex().getNbrList().getBox().putInBox(pos);
    sim.putInBox();
    
    const vector<uint0> & types(sim.getCellComplex().getTypes());
    char filename[255];
    FILE *pFile;
    sprintf (filename, "pos_%03d.dat", m);
    pFile=fopen (filename,"w");
    for(int i=0; i<pos.size(); ++i){
      fprintf(pFile, "%d ",i);
      for(int k=0; k<3; ++k)
	fprintf(pFile, "%16.8f ", pos[i][k]);
      fprintf(pFile, "%u3", types[i]);
      fprintf(pFile, "\n");
    }
    fclose(pFile);

    sprintf (filename, "intf_%03d.dat", m);
    pFile = fopen (filename,"w");
    sim.getCellComplex().drawInterfaceGnuplot(0, 1, pos, pFile);
    fclose(pFile);      
  }
  
  return 0;
}
