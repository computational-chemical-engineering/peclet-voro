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

int main()
{
  typedef double real_t;
  ExplicitEuler<real_t> sim;
  Array<real_t, 3> L;
  L[0] = 1;
  L[1] = 1;
  L[2] = 1;
  sim.setL(L);
  sim.setPressure(1);
  sim.setMassDensity(1);
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
    vector<Array<real_t, 3> > pos2(numPart);
    vector<uint0> types(numPart);
    types.resize(numPart);
    size_t iBegin=0, iLast=numPart-1;
    real_t halfL = 0.5*L[1];
    for(int i=0; i<pos.size(); ++i){
      if (pos[i][1]<halfL){
	for(int k=0; k<3; ++k)
	  pos2[iBegin][k] = pos[i][k];
	types[iBegin] = 0;
	vel[iBegin][0] = -2e-2;
	vel[iBegin][1] = 0;
	vel[iBegin][2] = 0;
	++iBegin;
      } else {
	for(int k=0; k<3; ++k)
	  pos2[iLast][k] = pos[i][k];
	types[iLast] = 1;
	vel[iLast][0] = 2e-2;
	vel[iLast][1] = 0;
	vel[iLast][2] = 0;
	--iLast;
      }
    }
    // for(size_t i=0; i< pos2.size(); ++i)
    //   printf("%f %f %f\n", pos2[i][0], this->pos2[i][1], pos2[i][2]);
    sim.setPositions(pos2);
    sim.setVelocities(vel);
    sim.setTypes(types);
    if (!sim.init()){
      fprintf(stderr, "initialization simulation failed\n");
      return 1;
    }
  }

  for(int m=0; m<100; ++m){
    for(int i=0; i< 1; ++i){
      sim.step(2,dt);
      printf("%16.8g %16.8g %16.8g\n", sim.getTime(), sim.getKineticEnergy(), sim.getInternalEnergy());
      //       vector<Array<real_t,3> > & vel(sim.getVelocities());
      // #pragma omp parallel for
      //       for(int i=0; i<vel.size(); ++i)
      // 	for(int k=0; k<3; ++k)
      // 	  vel[i][k] *= 0.995;
    }
    const vector<Array<real_t,3> > & pos(sim.getPositions());
    //    sim.getCellComplex().getNbrList().getBox().putInBox(pos);
    sim.putInBox();
    char filename[50];
    FILE *pFile;
    sprintf (filename, "pos_%03d.dat", m);
    pFile=fopen (filename,"w");
    for(int i=0; i<pos.size(); ++i){
      fprintf(pFile, "%d ",i);
      for(int k=0; k<3; ++k)
	fprintf(pFile, "%16.8f ", pos[i][k]);
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
