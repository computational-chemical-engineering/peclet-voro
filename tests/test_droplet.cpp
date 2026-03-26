/**
 * @file test_droplet.cpp
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
  L[0] = 2;
  L[1] = 1;
  L[2] = 1;
  IntfDyn<real_t> sim;
  sim.setL(L);
  sim.setPressure(100);
  sim.setMassDensity(1);
  real_t dt(0.0005);
  {
    vector<Array<real_t, 3> > pos;
    vector<uint0> types;
    FILE *pFile;
    char filename[255];
    sprintf(filename, "pos_init.dat");
    pFile = fopen(filename, "r");
    if (!pFile) {
      fprintf(stderr, "Error: cannot open %s\n", filename);
      return 1;
    }
    while (true) {
      int j, ierr, type;
      ierr = fscanf(pFile, "%d", &j);
      if (ierr == EOF)
        break;
      ierr = fscanf(pFile, "%d", &type);
      if (ierr == EOF)
        break;
      Array<real_t, 3> coord;
      for (int k = 0; k < 3; ++k) {
        double a;
        ierr = fscanf(pFile, "%lf", &a);
        if (ierr == EOF)
          break;
        coord[k] = a;
      }
      if (ierr == EOF)
        break;
      types.push_back((uint0)type);
      pos.push_back(coord);
    }
    fclose(pFile);

    size_t n = pos.size();
    pos.resize(2 * n);
    types.resize(2 * n);
    for (size_t i = 0; i < n; ++i) {
      pos[n + i][0] = 1.0 + pos[i][0];
      pos[n + i][1] = pos[i][1];
      pos[n + i][2] = pos[i][2];
      types[n + i] = types[i];
    }

    sim.getCellComplex().getNbrList().getBox().putInBox(pos);
    sim.setPositions(pos);

    vector<real_t> masses(pos.size());
    vector<real_t> bulkVisc(pos.size());
    vector<real_t> visc(pos.size());
    vector<Array<real_t, 3> > vel(pos.size());
    real_t mass = 1.0 / pos.size();

    sim.setIntfTension(1.e-1, 0, 1);
    sim.setIntfTension(1.e-1, 0, 2);
    sim.setIntfTension(2.e-1, 1, 2);

    for (size_t i = 0; i < pos.size(); ++i) {
      for (uint0 k = 0; k < 3; ++k)
        vel[i][k] = 0.0;
      if (types[i] == 0) {
        masses[i] = 0.1 * mass;
        visc[i] = 0.001;
        bulkVisc[i] = 0.001;
      } else {
        masses[i] = 10 * mass;
        visc[i] = 0.01;
        bulkVisc[i] = 0.01;
        if (pos[i][0] > 0.5 * L[0]) {
          vel[i][0] = -0.2;
          types[i] = 2;
        } else
          vel[i][0] = 0.2;
      }
    }

    sim.setTypes(types);
    sim.setMasses(masses);
    sim.setVelocities(vel);
    sim.setViscosities(visc);
    sim.setBulkViscosities(bulkVisc);
  }
  if (!sim.init()) {
    fprintf(stderr, "initialization simulation failed\n");
    return 1;
  }

  for (int m = 0; m < 999; ++m) {
    for (int i = 0; i < 5; ++i) {
      sim.step(20, dt);
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

    sprintf(filename, "intf1_%03d.dat", m);
    pFile = fopen(filename, "w");
    sim.getCellComplex().drawInterfaceGnuplot(1, 0, pos, pFile);
    sim.getCellComplex().drawInterfaceGnuplot(1, 2, pos, pFile);
    fclose(pFile);

    sprintf(filename, "intf2_%03d.dat", m);
    pFile = fopen(filename, "w");
    sim.getCellComplex().drawInterfaceGnuplot(2, 0, pos, pFile);
    sim.getCellComplex().drawInterfaceGnuplot(2, 1, pos, pFile);
    fclose(pFile);
  }

  return 0;
}
