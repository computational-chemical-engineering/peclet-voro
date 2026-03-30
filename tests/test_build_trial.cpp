/**
 * @file test_build_trial.cpp
 * \brief test program: build a Voronoi tesselation and rebuild it after deformation
 *
 */
// #include <omp.h>
#include <stdio.h>

#include <algorithm>
#include <boost/random.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <cstdio>
#include <ctime>
#include <vector>
#include <voronoi_dynamics/voronoi.hpp>

using std::clock;
using std::clock_t;
using std::sort;
using std::vector;
using vor::Box;
using vor::BoxLE;
using vor::Cell;
using vor::CellComplex;
using vor::CellGeometry;
using vor::NbrsToFacets;
using vor::uint1;
using vor::uint2;
using namespace std;
using namespace vor;

// using voro::container;

int main() {
  typedef double real_t;
  typedef boost::mt11213b base_generator_type;
  typedef boost::uniform_01<real_t> distribution_type;
  typedef boost::variate_generator<base_generator_type&, distribution_type> gen_type;
  base_generator_type rng(1);
  gen_type pointGen(rng, distribution_type());

  // generate random particle positions
  std::array<real_t, 3> L;
  L[0] = 1;
  L[1] = 1;
  L[2] = 1;
  BoxLE<real_t> box(L);
  CellComplex<real_t> complex(&box);
  vector<CellGeometry<real_t> >& geoms(complex.getGeoms());
  vector<Cell<real_t> > cells;

  uint particle_type, no_particles, no_particles_x, no_particles_y, no_particles_z;
  printf("Type 1 for random and 2 for regular particle generation\n");
  scanf("%u", &particle_type);
  if (particle_type == 1) {
    printf("Enter total number of particles to generate at random locations\n");
    scanf("%u", &no_particles);
  } else if (particle_type == 2) {
    printf("Enter number of particles in x direction\n");
    scanf("%u", &no_particles_x);
    printf("Enter number of particles in y direction\n");
    scanf("%u", &no_particles_y);
    printf("Enter number of particles in z direction\n");
    scanf("%u", &no_particles_z);
    no_particles = no_particles_x * no_particles_y * no_particles_z;
  }

  vector<std::array<real_t, 3> > p(no_particles);

  if (particle_type == 1) {
    for (int i = 0; i < p.size(); ++i) {
      for (uint k(0); k < 3; ++k) {
        p[i][k] = L[k] * pointGen();
      }
    }
  } else if (particle_type == 2) {
    real_t spacing_x, spacing_y, spacing_z;
    spacing_x = L[0] / no_particles_x;
    spacing_y = L[1] / no_particles_y;
    spacing_z = L[2] / no_particles_z;

    uint m = 0;
    for (uint i = 0; i < no_particles_x; i++) {
      for (uint j = 0; j < no_particles_y; j++) {
        for (uint k = 0; k < no_particles_z; k++, m++) {
          p[m][0] = (spacing_x / 2) + (i * spacing_x);
          p[m][1] = (spacing_y / 2) + (j * spacing_y);
          p[m][2] = (spacing_z / 2) + (k * spacing_z);
        }
      }
    }
  }
  complex.build(p);
  complex.materializeCells(cells);

  real_t vol(0);
#pragma omp parallel for reduction(+ : vol)
  for (size_t i = 0; i < geoms.size(); ++i) {
    geoms[i].computeVolume();
    vol += geoms[i].getVolume();
  }
  printf("summed volume cells: %f\n", vol);

  // additional stuff

  uint cell_id, facet_id;

  printf("Please enter the cell number to investigate : ");
  scanf("%u", &cell_id);
  printf("Cell number %u information\n", cells[cell_id].getID());
  printf("Cell Volume : %f\n", geoms[cell_id].getVolume());
  printf("Total number of facets in cell : %u \n", cells[cell_id].numFacets());
  printf("Total number of verticies in cell : %u \n\n", cells[cell_id].numVertices());
  printf("Please enter the facet number to investigate : ");
  scanf("%u", &facet_id);
  // printf("Facet area : %f\n",geoms[cell_id].getVolume());
  printf("Neighbour cell id : %u\n", cells[cell_id].getNbr(facet_id));
  printf("Running CCW on facet : %u\n", facet_id);
  cells[cell_id].printFacetInfo(p[cell_id], facet_id);

  FILE* printFile;
  printFile = fopen("GNUPlotfile.txt", "w");
  cells[cell_id].drawGnuplot(p[cell_id], printFile);
  fclose(printFile);

  // start = clock();
  // container con(0,L[0],0,L[1],0,L[2],57,57,57,true,true,true,8);
  // for(size_t i=0;i<p.size();i++) {
  //   con.put(i,p[i][0],p[i][1],p[i][2]);
  // }

  // for (uint i=0; i<N; ++i)
  //   con.compute_all_cells();
  // duration = ( clock() - start ) / (real_t) CLOCKS_PER_SEC;
  // printf("Voronoi cells created by voro++ in %f second\n", duration/double(N));

  // vol = con.sum_cell_volumes();
  // printf("summed volume cells: %f\n", vol);

  std::array<real_t, 3> orig;
  orig[0] = 0;
  orig[1] = 0;
  orig[2] = 0;
  //  cells[0].drawGnuplot(orig, stdout);

  // vector<std::array<real_t,3> > areas;
  // printf("volume: %f\n", cells[0].computeAreas(areas));

  // CellGeometry<real_t> geom(cells[0]);
  // geom.computeConnectingVectors(p, box);
  // geom.computeEdgeInv();
  // geom.updateVertexPos();

  // vector<std::array<real_t, 3> > dp(p.size());
  // for(size_t i(0); i < dp.size(); ++i)
  //   for(uint0 k(0); k<3; ++k)
  //     dp[i][k] = 1e-9*pointGen();

  // // dp[cells[0].getNbr(0)][0]=0.113700e-7;
  // // dp[cells[0].getNbr(0)][1]=0.071085e-7;
  // // dp[cells[0].getNbr(0)][2]=0.045748e-7;

  // // dp[cells[0].getNbr(0)][0] = 1.0e-7;
  // // dp[cells[0].getNbr(0)][1] = 1.0e-7;
  // // dp[cells[0].getNbr(0)][2] = 0.0;

  // geom.diffVolume();
  // double vol = geom.getVol();
  // vector< std::array<real_t, 3> > dV(geom.getdV());
  // real_t volTot(0);
  // for(size_t i(0); i < dV.size(); ++i){
  //   uint2 j(cells[0].getNbr(i));
  //   volTot +=
  //   dV[i][0]*(dp[j][0]-dp[0][0])+dV[i][1]*(dp[j][1]-dp[0][1])+dV[i][2]*(dp[j][2]-dp[0][2]);
  // }
  // printf("dV: %g\n", volTot);

  // geom.computeAll();
  // areas = geom.getAreas();
  // dV = geom.getdV();
  // volTot = 0.0;
  // for(size_t i(0); i < dV.size(); ++i){
  //   uint2 j(cells[0].getNbr(i));
  //   volTot +=
  //   dV[i][0]*(dp[j][0]-dp[0][0])+dV[i][1]*(dp[j][1]-dp[0][1])+dV[i][2]*(dp[j][2]-dp[0][2]);
  // }
  // printf("dV: %g\n", volTot);

  // for(size_t i(0); i<dp.size(); ++i)
  //   for(uint0 k(0); k<3; ++k)
  //     p[i][k] += dp[i][k];

  // //  cells[0].printNbrFacets(cells);
  // //cells[0].drawGnuplot(orig, stdout);

  // Cell<real_t> cellOld(cells[0]);

  // geom.computeConnectingVectors(p, box);
  // geom.computeEdgeInv();
  // geom.updateVertexPos();
  // //  geom.computeVolume();
  // geom.computeAll();
  // vector< std::array<real_t, 3> > areasNew = geom.getAreas();

  // // for(uint1 i(0); i< cells[0].m_numVertices; ++i)
  // //   for(uint0 k(0); k< 3; ++k)
  // //     printf("v0: %u, dir: %u, diff: %g\n", i, k,
  // // 	     (cells[0].m_vertexPos[i][k]-cellOld.m_vertexPos[i][k])/1e-5);
  // printf("dV: %g\n", (geom.getVol()-vol));

  // for(int i(0); i< areas.size(); ++i){
  //   printf("%u: ", i);
  //   real_t dASq = (areasNew[i][0]*areasNew[i][0] + areasNew[i][1]*areasNew[i][1] +
  //   areasNew[i][2]*areasNew[i][2]); dASq -= (areas[i][0]*areas[i][0] + areas[i][1]*areas[i][1] +
  //   areas[i][2]*areas[i][2]); printf("dASq: %g\n", dASq);
  // }

  // printf("\n");
  // vector<uint2> indx;
  // vector<std::array<real_t, 3> > grad;
  // for(size_t i(0); i< areas.size(); ++i){
  //   printf("%lu: ", i);
  //   geom.gradFacetAreaSq(i, indx, grad);
  //   real_t dASq(0);
  //   for(size_t j(0); j< indx.size(); ++j)
  //     for(uint0 k(0); k<3; ++k){
  // 	uint2 m = cells[0].getNbr(indx[j]);
  // 	dASq += grad[j][k]*(dp[m][k]-dp[0][k]);
  //     }
  //   printf("dASq: %g\n", dASq);
  // }

  // NbrsToFacets nbr2F, nbr2F2;
  // nbr2F.init(cells);
  // nbr2F.print();
  //  nbr2F2 = nbr2F.transpose();

  //   vector<uint2> ptr(cells.size()+1);
  //   vector<uint2> nbr;
  //   indx.reserve(cells.size()*12);
  //   vector<uint1> facet;
  //   value.reserve(cells.size()*12);
  //   ptr[0] = 0;

  //   vector<uint2> ptrTrans(ptr.size());
  //   vector<uint2> nbrTrans(nbr.size());
  //   vector<uint2> facetTrans(facet.size());
  // #pragma omp parallel for
  //   for(size_t i(0); i< nbr.size(); ++i)
  // #pragma omp atomic
  //     ++ptrTrans[nbr[i]+1];
  //   for(size_t i(0); i< ptr.size()-1; ++i)
  //     ptrTrans[i+1] += ptrTrans[i];
  //   {
  //     vector<uint2> ptrTemp(ptrTrans.size());
  // #pragma omp parallel for
  //     for(size_t i(0); i< ptrTrans.size(); ++i)
  //       ptrTemp[i] = ptrTrans[i];
  //     for(size_t i(0); i< cells.size(); ++i)
  //       for(uint2 j(ptr[i]); j< ptr[i+1]; ++j){
  // 	uint2 indx = nbr[j];
  // #pragma omp atomic capture
  // 	uint2 indx2 = ptrTemp[indx]++;
  // 	nbrTrans[indx2] = i;
  // 	facetTrans[indx2] = facet[j];
  //       }

  //   }

  //   vector<std::array<real_t, 3> > dVself(cells.size(),0);
  // #pragma omp parallel
  //   {
  //     vector<IndxValue> indxValue;
  // #pragma omp for
  //     for(size_t i=0; i< geoms.size(); ++i){
  //       geoms[i].computeAll();
  //       //    printf("volume cell %lu: %f\n", i, dVol);
  //       vector< std::array<real_t, 3> > & dV(geoms[i].getdV());
  //       for(uint1 j(0); j< dV.size(); ++j)
  // 	for(uint0 k(0); k<3; ++k)
  // 	  dVself[i][k] -= dV[j][k];

  //       uint1 numVars(3*dV.size());
  //       ptr[i+1] = ptr[i] + numVars;
  //       indxValue.resize(numVars);
  //       for(uint1 j(0); j< dV.size(); ++j){
  // 	uint2 baseIndx = 3*cells[0].getNbr(j);
  // 	for(uint0 k(0); k<3; ++k){
  // 	  indxValue[3*j+k].indx = baseIndx+k;
  // 	  indxValue[3*j+k].value = dV[j][k];
  // 	}
  //       }
  //       sort(indxValue.begin(), indxValue.end(), CompareIndxValue<real_t>());
  // #pragma omp critical
  //       {
  // 	indx.resize(ptr[i+1]);
  // 	value.resize(ptr[i+1]);
  //       }
  //       for(uint1 j(0); j< indxValue.size(); ++j){
  // 	indx[ptr[i]+j] = indxValue[j].indx;
  // 	value[ptr[i]+j] = indxValue[j].value;
  //       }
  //     }
  //   }

  // make matrix dV/dx

  // t=0
  //   v0_i, r0_i

  // t=0.5
  //   v*_i = v0_i + 0.5*f0_i/m_i * dt
  //   vh_i = v*_i + 0.5 * Ph_j * dV_j/dr0_i 1/m_i * dt

  // t=1.0
  //   r*_i = r0_i + v*_i * dt
  //   r1_i = r0_i + vh_i *dt
  //        = r*_i + (0.5 * Ph_j *dt^2) * dV_j/dr0_i 1/m_i
  //   start with Ph_j = P0_j
  //   V1_i = V0_i

  // // Newton-Raphson:
  // G0_kj = sum_i dV_k/dr0_i*1/mi*dV_j/dr0_i

  // G0_kj * (0.5*dPh_j*dt^2) = (V0_k-V1_k)
  // Ph_j := Ph_j + dPh_j
  // r1_i = r*_i + (0.5 * Ph_j *dt^2) * dV_j/dr0_i 1/m_i
  // when converged compute also:
  // vh_i = v*_i + 0.5 * Ph_j * dV_j/dr0_i 1/m_i * dt

  // v*_i = vh_i + 0.5*f1_i/m_i * dt
  // v1_i = v*_i + 0.5 * P1_j * dV_j/dr1_i * 1/m_i * dt

  // dV_k/dt = 0 -> dV_k/dr1_i * v1_i = 0

  // G1_kj * (0.5 * P1_j * dt) = -dV_k/dr1_i * v*_i

  return 0;
}
