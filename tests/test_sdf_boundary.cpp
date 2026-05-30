#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <voronoi_dynamics/voronoi.hpp>

using std::array;
using std::sort;
using std::vector;
using vor::Box;
using vor::CellComplex;
using vor::PowerCellComplex;
using vor::SignedDistanceBoundary;
using vor::boundaryNbr;
using vor::uint1;
using vor::uint2;

namespace {

const double kPi = 3.141592653589793238462643383279502884;

class SlabBoundary : public SignedDistanceBoundary<double> {
 public:
  SlabBoundary(double xmin, double xmax) : m_xmin(xmin), m_xmax(xmax) {}

  double value(const array<double, 3>& x) const override {
    return std::min(x[0] - m_xmin, m_xmax - x[0]);
  }

  array<double, 3> gradient(const array<double, 3>& x) const override {
    if ((x[0] - m_xmin) <= (m_xmax - x[0]))
      return array<double, 3>{1.0, 0.0, 0.0};
    return array<double, 3>{-1.0, 0.0, 0.0};
  }

  bool closestPoint(const array<double, 3>& x, array<double, 3>& c,
                    array<double, 3>& normal) const override {
    c = x;
    if ((x[0] - m_xmin) <= (m_xmax - x[0])) {
      c[0] = m_xmin;
      normal = array<double, 3>{1.0, 0.0, 0.0};
    } else {
      c[0] = m_xmax;
      normal = array<double, 3>{-1.0, 0.0, 0.0};
    }
    return true;
  }

 private:
  double m_xmin;
  double m_xmax;
};

class SphereHoleBoundary : public SignedDistanceBoundary<double> {
 public:
  SphereHoleBoundary(const array<double, 3>& center, double radius)
      : m_center(center), m_radius(radius) {}

  double value(const array<double, 3>& x) const override {
    const double dx = x[0] - m_center[0];
    const double dy = x[1] - m_center[1];
    const double dz = x[2] - m_center[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - m_radius;
  }

  array<double, 3> gradient(const array<double, 3>& x) const override {
    array<double, 3> grad = {x[0] - m_center[0], x[1] - m_center[1], x[2] - m_center[2]};
    const double norm =
        std::sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
    if (norm <= 1.0e-14)
      return array<double, 3>{1.0, 0.0, 0.0};
    for (uint1 k = 0; k < 3; ++k)
      grad[k] /= norm;
    return grad;
  }

 private:
  array<double, 3> m_center;
  double m_radius;
};

class CylinderBoundary : public SignedDistanceBoundary<double> {
 public:
  CylinderBoundary(const array<double, 3>& center, double radius)
      : m_center(center), m_radius(radius) {}

  double value(const array<double, 3>& x) const override {
    const double dx = x[0] - m_center[0];
    const double dy = x[1] - m_center[1];
    return m_radius - std::sqrt(dx * dx + dy * dy);
  }

  array<double, 3> gradient(const array<double, 3>& x) const override {
    const double dx = x[0] - m_center[0];
    const double dy = x[1] - m_center[1];
    const double rho = std::sqrt(dx * dx + dy * dy);
    if (rho <= 1.0e-14)
      return array<double, 3>{1.0, 0.0, 0.0};
    return array<double, 3>{-dx / rho, -dy / rho, 0.0};
  }

  bool closestPoint(const array<double, 3>& x, array<double, 3>& c,
                    array<double, 3>& normal) const override {
    const double dx = x[0] - m_center[0];
    const double dy = x[1] - m_center[1];
    const double rho = std::sqrt(dx * dx + dy * dy);
    c = x;
    if (rho <= 1.0e-14) {
      c[0] = m_center[0] + m_radius;
      c[1] = m_center[1];
      normal = array<double, 3>{-1.0, 0.0, 0.0};
      return true;
    }
    const double scale = m_radius / rho;
    c[0] = m_center[0] + dx * scale;
    c[1] = m_center[1] + dy * scale;
    normal = array<double, 3>{-dx / rho, -dy / rho, 0.0};
    return true;
  }

 private:
  array<double, 3> m_center;
  double m_radius;
};

template <typename Complex>
double sumVolumes(const Complex& complex) {
  double sum = 0.0;
  for (size_t i = 0; i < complex.numCells(); ++i)
    sum += complex.getGeometryView(i).getVolume();
  return sum;
}

template <typename Complex>
bool allVerticesInsideBoundary(const Complex& complex, const vector<array<double, 3> >& pos,
                               const SignedDistanceBoundary<double>& boundary, double tol) {
  for (size_t cellId = 0; cellId < complex.numCells(); ++cellId) {
    const auto cell = complex.getCellView(cellId);
    const array<double, 3>& center = pos[cell.getID()];
    for (uint1 i = 0; i < cell.numVertices(); ++i) {
      array<double, 3> x = center;
      for (uint1 k = 0; k < 3; ++k)
        x[k] += cell.getVertexPos(i)[k];
      if (boundary.value(x) < -tol)
        return false;
    }
  }
  return true;
}

template <typename Complex>
double maxOutsideViolation(const Complex& complex, const vector<array<double, 3> >& pos,
                           const SignedDistanceBoundary<double>& boundary) {
  double minPhi = 0.0;
  for (size_t cellId = 0; cellId < complex.numCells(); ++cellId) {
    const auto cell = complex.getCellView(cellId);
    const array<double, 3>& center = pos[cell.getID()];
    for (uint1 i = 0; i < cell.numVertices(); ++i) {
      array<double, 3> x = center;
      for (uint1 k = 0; k < 3; ++k)
        x[k] += cell.getVertexPos(i)[k];
      const double phi = boundary.value(x);
      if (phi < minPhi)
        minPhi = phi;
    }
  }
  return -minPhi;
}

template <typename Complex>
size_t countBoundaryFacets(const Complex& complex) {
  size_t count = 0;
  for (size_t cellId = 0; cellId < complex.numCells(); ++cellId) {
    const auto cell = complex.getCellView(cellId);
    for (uint1 facet = 0; facet < cell.numFacets(); ++facet)
      if (cell.getNbr(facet) == boundaryNbr)
        ++count;
  }
  return count;
}

template <typename ComplexA, typename ComplexB>
bool compareByParticleId(const ComplexA& a, const ComplexB& b, const vector<uint2>& particleIds,
                         double tol, const char* label) {
  vector<uint2> nbrsA;
  vector<uint2> nbrsB;
  for (size_t i = 0; i < particleIds.size(); ++i) {
    const uint2 particleId = particleIds[i];
    const uint2 cellA = a.getCellIndexForParticle(particleId);
    const uint2 cellB = b.getCellIndexForParticle(particleId);
    if (cellA == vor::noNbr || cellB == vor::noNbr) {
      std::fprintf(stderr, "%s: missing cell for particle %u\n", label,
                   static_cast<unsigned>(particleId));
      return false;
    }
    nbrsA.clear();
    nbrsB.clear();
    const auto viewA = a.getCellView(cellA);
    const auto viewB = b.getCellView(cellB);
    for (uint1 facet = 0; facet < viewA.numFacets(); ++facet) {
      const uint2 nbrId = viewA.getNbr(facet);
      if (nbrId != vor::noNbr)
        nbrsA.push_back(nbrId);
    }
    for (uint1 facet = 0; facet < viewB.numFacets(); ++facet) {
      const uint2 nbrId = viewB.getNbr(facet);
      if (nbrId != vor::noNbr)
        nbrsB.push_back(nbrId);
    }
    sort(nbrsA.begin(), nbrsA.end());
    sort(nbrsB.begin(), nbrsB.end());
    if (nbrsA != nbrsB) {
      std::fprintf(stderr, "%s: neighbour mismatch for particle %u\n", label,
                   static_cast<unsigned>(particleId));
      return false;
    }
    const double volA = a.getGeometryView(cellA).getVolume();
    const double volB = b.getGeometryView(cellB).getVolume();
    if (std::abs(volA - volB) > tol) {
      std::fprintf(stderr, "%s: volume mismatch for particle %u\n", label,
                   static_cast<unsigned>(particleId));
      return false;
    }
  }
  return true;
}

bool testPlanarBoundaryVoronoiBuild() {
  std::mt19937_64 rng(91);
  std::uniform_real_distribution<double> ux(0.25, 0.75);
  std::uniform_real_distribution<double> uyz(0.0, 1.0);

  Box<double> box(array<double, 3>{1.0, 1.0, 1.0});
  CellComplex<double> complex(&box);
  SlabBoundary boundary(0.2, 0.8);
  complex.setBoundary(&boundary);

  vector<array<double, 3> > pos(96);
  for (size_t i = 0; i < pos.size(); ++i) {
    pos[i][0] = ux(rng);
    pos[i][1] = uyz(rng);
    pos[i][2] = uyz(rng);
  }

  complex.build(pos);

  const double fluidVolume = 0.6;
  const double tol = 1.0e-10;
  if (std::abs(sumVolumes(complex) - fluidVolume) > tol) {
    std::fprintf(stderr,
                 "planar boundary build volume mismatch: %.16e vs %.16e (boundary facets=%zu, "
                 "allInside=%d)\n",
                 sumVolumes(complex), fluidVolume, countBoundaryFacets(complex),
                 allVerticesInsideBoundary(complex, pos, boundary, 1.0e-9) ? 1 : 0);
    return false;
  }
  if (countBoundaryFacets(complex) == 0u) {
    std::fprintf(stderr, "planar boundary build produced no boundary facets\n");
    return false;
  }
  if (!allVerticesInsideBoundary(complex, pos, boundary, 1.0e-9)) {
    std::fprintf(stderr, "planar boundary build leaked vertices outside the admissible domain\n");
    return false;
  }
  return true;
}

bool testSphereBoundaryPowerCellReactivation() {
  std::mt19937_64 rng(13);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  const array<double, 3> center = {0.5, 0.5, 0.5};
  const double radius = 0.2;
  SphereHoleBoundary boundary(center, radius);

  Box<double> box(array<double, 3>{1.0, 1.0, 1.0});
  PowerCellComplex<double> updated(&box);
  PowerCellComplex<double> reference(&box);
  updated.setBoundary(&boundary);
  reference.setBoundary(&boundary);

  vector<array<double, 3> > start(64);
  start[0] = center;
  for (size_t i = 1; i < start.size(); ++i) {
    do {
      for (uint1 k = 0; k < 3; ++k)
        start[i][k] = uni(rng);
    } while (boundary.value(start[i]) <= 0.08);
  }
  vector<array<double, 3> > finish = start;
  finish[0] = array<double, 3>{0.85, 0.5, 0.5};

  vector<double> weights(start.size(), 0.0);
  updated.setWeights(weights);
  reference.setWeights(weights);
  updated.build(start);

  if (!updated.isParticleActive(0u) || updated.hasCell(0u)) {
    std::fprintf(stderr, "sphere boundary build did not leave particle 0 active without a cell\n");
    return false;
  }

  reference.build(finish);
  updated.update(finish);

  if (updated.getLastUpdateStats().rebuilt_from_scratch) {
    std::fprintf(stderr, "sphere boundary reactivation fell back to full rebuild\n");
    return false;
  }
  if (!updated.isParticleActive(0u) || !updated.hasCell(0u) ||
      updated.getCellIndexForParticle(0u) == vor::noNbr) {
    std::fprintf(stderr, "sphere boundary update failed to reactivate particle 0\n");
    return false;
  }

  vector<uint2> cellIds = reference.getCellParticleIds();
  if (!compareByParticleId(updated, reference, cellIds, 1.0e-9, "sphere-reactivation compare"))
    return false;
  if (!allVerticesInsideBoundary(updated, finish, boundary, 1.0e-8)) {
    std::fprintf(stderr, "sphere boundary update leaked vertices outside the admissible domain\n");
    return false;
  }

  return true;
}

bool testCylinderBoundaryVoronoiBuild() {
  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> ur(0.0, 1.0);
  std::uniform_real_distribution<double> utheta(0.0, 2.0 * kPi);
  std::uniform_real_distribution<double> uz(0.0, 1.0);

  const array<double, 3> center = {0.5, 0.5, 0.5};
  const double radius = 0.3;
  CylinderBoundary boundary(center, radius);

  Box<double> box(array<double, 3>{1.0, 1.0, 1.0});
  CellComplex<double> complex(&box);
  complex.setBoundary(&boundary);

  vector<array<double, 3> > pos(128);
  for (size_t i = 0; i < pos.size(); ++i) {
    const double rho = radius * std::sqrt(ur(rng));
    const double theta = utheta(rng);
    pos[i][0] = center[0] + rho * std::cos(theta);
    pos[i][1] = center[1] + rho * std::sin(theta);
    pos[i][2] = uz(rng);
  }

  complex.build(pos);

  const double fluidVolume = kPi * radius * radius;
  const double tol = 1.0e-3;
  const double volume = sumVolumes(complex);
  if (std::abs(volume - fluidVolume) > tol) {
    std::fprintf(stderr, "cylinder boundary build volume mismatch: %.16e vs %.16e\n", volume,
                 fluidVolume);
    return false;
  }
  if (countBoundaryFacets(complex) == 0u) {
    std::fprintf(stderr, "cylinder boundary build produced no boundary facets\n");
    return false;
  }
  if (!allVerticesInsideBoundary(complex, pos, boundary, 3.0e-4)) {
    std::fprintf(stderr,
                 "cylinder boundary build leaked vertices outside the admissible domain "
                 "(max violation %.16e)\n",
                 maxOutsideViolation(complex, pos, boundary));
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!testPlanarBoundaryVoronoiBuild())
    return 1;
  if (!testCylinderBoundaryVoronoiBuild())
    return 1;
  if (!testSphereBoundaryPowerCellReactivation())
    return 1;
  std::puts("sdf boundary regression tests passed");
  return 0;
}
