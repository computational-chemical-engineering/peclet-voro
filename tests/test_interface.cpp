/**
 * @file test_interface.cpp
 * \brief test program: test a simple simulation of compressible Euler equations
 *
 */
// #include <omp.h>
#include <boost/random.hpp>
#include <cstdio>
#include <vector>
#include <voronoi_dynamics/simulation.hpp>
#include <voronoi_dynamics/voronoi.hpp>
using std::vector;
using vor::uint0;
using vor::uint1;
using vor::uint2;

using vor::Array;
using vor::IntfDyn;

int main() {
  typedef double real_t;
  // generate random particle positions
  Array<real_t, 3> L;
  L[0] = 1;
  L[1] = 1;
  L[2] = 1;
  IntfDyn<real_t> sim;
  sim.setL(L);
  sim.setPressure(100);
  sim.setMassDensity(1);
  real_t dt(0.001);
  {
    int numPart = 10000;
    vector<Array<real_t, 3> > pos(numPart);
    FILE *pFile;
    char filename[255];
    sprintf(filename, "pos_eq_%d.dat", numPart);
    pFile = fopen(filename, "r");
    if (!pFile) {
      fprintf(stderr, "Error: cannot open %s\n", filename);
      return 1;
    }
    int j;
    for (int i = 0; i < pos.size(); ++i) {
      int j, ierr;
      ierr = fscanf(pFile, "%d", &j);
      for (int k = 0; k < 3; ++k) {
        double a;
        ierr = fscanf(pFile, "%lf", &a);
        if (ierr == EOF)
          break;
        pos[j][k] = a;
      }
    }
    fclose(pFile);
    sim.getCellComplex().getNbrList().getBox().putInBox(pos);
    sim.setPositions(pos);

    vector<uint0> types(numPart);
    vector<Array<real_t, 3> > pos2(numPart);
    size_t iBegin = 0, iLast = numPart - 1;
    real_t rSq = 0.018 * L[1] * L[1];
    for (int i = 0; i < pos.size(); ++i) {
      bool inSphere =
          ((pos[i][0] - 0.5) * (pos[i][0] - 0.5) + (pos[i][1] - 0.5) * (pos[i][1] - 0.5) +
               (pos[i][2] - 0.5) * (pos[i][2] - 0.5) <
           rSq);
      if (inSphere) {
        types[i] = 1;
      } else {
        types[i] = 0;
      }
    }
    //    printf("last particle of type0: %lu, first particle of type 1: %lu\n", iLast, iBegin);
    sim.setTypes(types);

    vector<real_t> masses(types.size());
    vector<real_t> visc(types.size());
    vector<real_t> bulkVisc(types.size());
    vector<Array<real_t, 3> > vel(types.size());
    real_t mass = 1.0 / types.size();
    sim.setIntfTension(1.e-2, 0, 1);
    for (size_t i = 0; i < types.size(); ++i) {
      if (types[i] == 0) {
        masses[i] = mass;
        visc[i] = 0.01;
        bulkVisc[i] = 0.01;
      } else {
        masses[i] = 10 * mass;
        visc[i] = 0.1;
        bulkVisc[i] = 0.1;
      }
      for (uint0 k = 0; k < 3; ++k)
        vel[i][k] = 0.0;
    }
    sim.setMasses(masses);
    sim.setVelocities(vel);
    sim.setViscosities(visc);
    sim.setBulkViscosities(bulkVisc);
  }

  if (!sim.init()) {
    fprintf(stderr, "initialization simulation failed\n");
    return 1;
  }

  for (int m = 0; m < 100; ++m) {
    for (int i = 0; i < 50; ++i) {
      sim.step(50, dt);
      printf("%16.8g %16.8g %16.8g %16.8g\n", sim.getTime(), sim.getKineticEnergy(),
             sim.getInternalEnergy(), sim.getIntfEnergy());
    }
    const vector<Array<real_t, 3> > &pos(sim.getPositions());
    //    sim.getCellComplex().getNbrList().getBox().putInBox(pos);
    sim.putInBox();

    const vector<uint0> &types(sim.getCellComplex().getTypes());
    char filename[255];
    FILE *pFile;
    sprintf(filename, "pos_%03d.dat", m);
    pFile = fopen(filename, "w");
    for (int i = 0; i < pos.size(); ++i) {
      fprintf(pFile, "%d ", i);
      fprintf(pFile, "%3u ", types[i]);
      for (uint0 k = 0; k < 3; ++k)
        fprintf(pFile, "%16.8f ", pos[i][k]);
      fprintf(pFile, "\n");
    }
    fclose(pFile);

    sprintf(filename, "intf_%03d.dat", m);
    pFile = fopen(filename, "w");
    sim.getCellComplex().drawInterfaceGnuplot(1, 0, pos, pFile);
    //    sim.getCellComplex().drawInterfaceGnuplot(1, 2, pos, pFile);
    fclose(pFile);
  }

  return 0;
}
