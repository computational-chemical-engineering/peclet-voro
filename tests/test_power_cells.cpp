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
using vor::uint1;
using vor::uint2;

namespace {

template <typename Complex>
void collectSortedNbrsByParticle(const Complex& complex, uint2 particleId, vector<uint2>& nbrs) {
  nbrs.clear();
  const uint2 cellIndex = complex.getCellIndexForParticle(particleId);
  if (cellIndex == vor::noNbr)
    return;
  const auto cell = complex.getCellView(cellIndex);
  nbrs.reserve(cell.numFacets());
  for (uint1 facet = 0; facet < cell.numFacets(); ++facet) {
    const uint2 nbrId = cell.getNbr(facet);
    if (nbrId != vor::noNbr)
      nbrs.push_back(nbrId);
  }
  sort(nbrs.begin(), nbrs.end());
}

template <typename real_t, typename ComplexA, typename ComplexB>
bool compareByParticleId(const ComplexA& a, const ComplexB& b, const vector<uint2>& particleIds,
                         real_t tol, const char* label) {
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

    collectSortedNbrsByParticle(a, particleId, nbrsA);
    collectSortedNbrsByParticle(b, particleId, nbrsB);
    if (nbrsA != nbrsB) {
      std::fprintf(stderr, "%s: neighbour mismatch for particle %u\n", label,
                   static_cast<unsigned>(particleId));
      return false;
    }

    const real_t volA = a.getGeometryView(cellA).getVolume();
    const real_t volB = b.getGeometryView(cellB).getVolume();
    if (std::abs(volA - volB) > tol) {
      std::fprintf(stderr, "%s: volume mismatch for particle %u (%.16e vs %.16e)\n", label,
                   static_cast<unsigned>(particleId), static_cast<double>(volA),
                   static_cast<double>(volB));
      return false;
    }
  }
  return true;
}

template <typename Complex>
double sumVolumes(const Complex& complex) {
  double sum = 0.0;
  for (size_t i = 0; i < complex.numCells(); ++i)
    sum += complex.getGeometryView(i).getVolume();
  return sum;
}

bool containsParticleId(const vector<uint2>& ids, uint2 particleId) {
  return std::find(ids.begin(), ids.end(), particleId) != ids.end();
}

vector<array<double, 3> > emptyCellBasePositions() {
  return {
      {0.35762972288842593, 0.40044261704406114, 0.68938331700276845},
      {0.55973557064111568, 0.57445129399171091, 0.20769052686175465},
      {0.028662699255185029, 0.68892447787945044, 0.46934335909645619},
      {0.20715259577310696, 0.0039322523926723441, 0.013029711513380176},
      {0.4204224235070908, 0.61619145011275966, 0.89488361197528166},
      {0.41063605523764318, 0.26076873296958158, 0.026728938913603544},
      {0.10794239628485092, 0.36668398076575126, 0.0038867651813674765},
      {0.94009240539334449, 0.69127590461556809, 0.91866556116855058},
      {0.41973236396173769, 0.80409550809910835, 0.59863092069694768},
      {0.0034475707276081986, 0.48831784346613871, 0.88220463956500617},
      {0.93426212893618066, 0.98568640117661421, 0.43128775419961934},
      {0.1280122559000238, 0.8701470159832827, 0.7715443311942376},
      {0.70017779223363641, 0.64422300333535787, 0.63574480654705856},
      {0.99782251503123542, 0.42168563320651387, 0.12090927419585995},
      {0.11603672135799319, 0.42463062200370522, 0.17197591419104333},
      {0.10271522335119472, 0.23924466043451259, 0.97581133195150127},
      {0.34827587779503755, 0.6520972071268466, 0.53047885693950803},
      {0.36015408101918561, 0.8782553409393542, 0.081504385787565192},
      {0.49550853258117367, 0.34277090024945756, 0.47266892936314048},
      {0.69610633248038689, 0.074221483445991027, 0.4074829149268463},
      {0.19676156842134138, 0.66208237557331973, 0.87714887528316987},
      {0.19423310105269692, 0.19988551003099334, 0.49071045583290823},
      {0.93905839164953642, 0.41111291482514584, 0.80049631857437809},
      {0.17111416888484424, 0.46618396539181711, 0.93820723582388887},
  };
}

vector<array<double, 3> > reactivatedPositions() {
  return {
      {0.36510115026022383, 0.38982460689789361, 0.71925953672840437},
      {0.57224005507598252, 0.59616014289516439, 0.19052709234510834},
      {0.037208616404880143, 0.67385432383230526, 0.47081027658156932},
      {0.23468062190261577, 0.007283890519058458, 0.0079913425850404399},
      {0.43324009028789612, 0.63575936558902135, 0.90752544551691017},
      {0.43480915651671009, 0.28496914619514813, 0.051836213439002368},
      {0.093389312858563153, 0.35826089709895836, 0.00049576914771624165},
      {0.96110321103600915, 0.68228697928972559, 0.92324043180559345},
      {0.44013516772687111, 0.82875640261941541, 0.59259880137465348},
      {0.99874962861492944, 0.46903045318649039, 0.90740427573393811},
      {0.94191977654103853, 0.95769928323363451, 0.40890049838010034},
      {0.11991383324979316, 0.89045628260140497, 0.78051114742104599},
      {0.6804238336733025, 0.61801439111428347, 0.65152281294715142},
      {0.97116132196435778, 0.40658386560414589, 0.095918362861830947},
      {0.10551083956998594, 0.41326980940435126, 0.17744093803714159},
      {0.11583560217485117, 0.2575177184194174, 0.96846000161777868},
      {0.34102828504865418, 0.63906561355831404, 0.55635853811274938},
      {0.33079865283744658, 0.8973493914733206, 0.094472959121629702},
      {0.4921681352592735, 0.32711651484203169, 0.48944065492423},
      {0.70345574111674225, 0.045337984160968764, 0.39824820450277848},
      {0.20845569898969524, 0.68524983085497382, 0.87667561117369275},
      {0.2122951522745562, 0.20799597134485523, 0.47759867814377388},
      {0.93193623673203985, 0.42958526923493295, 0.82972256611012196},
      {0.16880979446369687, 0.47770849191513565, 0.91738626340724083},
  };
}

bool testZeroWeightsMatchVoronoi() {
  typedef double real_t;

  std::mt19937_64 rng(17);
  std::uniform_real_distribution<real_t> uni(real_t(0), real_t(1));

  Box<real_t> box(array<real_t, 3>{1.0, 1.0, 1.0});
  CellComplex<real_t> voronoi(&box);
  PowerCellComplex<real_t> power(&box);

  vector<array<real_t, 3> > pos(96);
  for (size_t i = 0; i < pos.size(); ++i)
    for (uint1 k = 0; k < 3; ++k)
      pos[i][k] = uni(rng);

  vector<real_t> weights(pos.size(), real_t(0));
  voronoi.build(pos);
  power.setWeights(weights);
  power.build(pos);

  if (power.numCells() != pos.size()) {
    std::fprintf(stderr, "zero-weight power build produced %zu cells, expected %zu\n",
                 power.numCells(), pos.size());
    return false;
  }
  for (size_t i = 0; i < pos.size(); ++i) {
    if (!power.getParticleActivity()[i]) {
      std::fprintf(stderr, "zero-weight power build deactivated particle %zu\n", i);
      return false;
    }
  }

  vector<uint2> particleIds(pos.size());
  for (size_t i = 0; i < particleIds.size(); ++i)
    particleIds[i] = static_cast<uint2>(i);

  const real_t tol = 1.0e-10;
  if (!compareByParticleId<real_t>(voronoi, power, particleIds, tol, "zero-weight compare"))
    return false;
  if (std::abs(sumVolumes(power) - real_t(1)) > tol) {
    std::fprintf(stderr, "zero-weight power build violated volume conservation\n");
    return false;
  }

  return true;
}

bool testActiveWithoutCellState() {
  typedef double real_t;

  Box<real_t> box(array<real_t, 3>{1.0, 1.0, 1.0});
  PowerCellComplex<real_t> power(&box);
  vector<array<real_t, 3> > pos = emptyCellBasePositions();
  vector<real_t> weights(pos.size(), real_t(0));
  weights[0] = real_t(-0.05);

  power.setWeights(weights);
  power.build(pos);

  if (power.numCells() != pos.size() - 1u) {
    std::fprintf(stderr, "weighted build produced %zu cells, expected %zu\n", power.numCells(),
                 pos.size() - 1u);
    return false;
  }
  if (!power.isParticleActive(5u)) {
    std::fprintf(stderr, "expected particle 5 to remain active after weighted build\n");
    return false;
  }
  if (power.hasCell(5u)) {
    std::fprintf(stderr, "expected particle 5 to remain active without a cell\n");
    return false;
  }
  if (power.getCellIndexForParticle(5u) != vor::noNbr) {
    std::fprintf(stderr, "active no-cell particle 5 still has a packed cell entry\n");
    return false;
  }
  if (!power.isParticleActive(0u) || !power.hasCell(0u)) {
    std::fprintf(stderr, "weight-owning particle 0 lost active/cell-owning state\n");
    return false;
  }

  const vector<uint2>& activeIds = power.getActiveParticleIds();
  const vector<uint2>& cellIds = power.getCellParticleIds();
  if (activeIds.size() != pos.size()) {
    std::fprintf(stderr, "active particle list has %zu ids, expected %zu\n", activeIds.size(),
                 pos.size());
    return false;
  }
  if (cellIds.size() != pos.size() - 1u) {
    std::fprintf(stderr, "cell-owning particle list has %zu ids, expected %zu\n", cellIds.size(),
                 pos.size() - 1u);
    return false;
  }
  if (!containsParticleId(activeIds, 5u)) {
    std::fprintf(stderr, "active particle list is missing active no-cell particle 5\n");
    return false;
  }
  if (containsParticleId(cellIds, 5u)) {
    std::fprintf(stderr, "cell-owning particle list unexpectedly contains particle 5\n");
    return false;
  }

  return true;
}

bool testInactiveParticleReactivationViaUpdate() {
  typedef double real_t;

  Box<real_t> box(array<real_t, 3>{1.0, 1.0, 1.0});
  PowerCellComplex<real_t> updated(&box);
  PowerCellComplex<real_t> reference(&box);
  vector<array<real_t, 3> > start = emptyCellBasePositions();
  vector<array<real_t, 3> > finish = reactivatedPositions();
  vector<real_t> weights(start.size(), real_t(0));
  weights[0] = real_t(-0.05);

  updated.setWeights(weights);
  updated.build(start);
  if (!updated.isParticleActive(5u) || updated.hasCell(5u)) {
    std::fprintf(stderr, "reactivation test requires particle 5 to start active without a cell\n");
    return false;
  }

  reference.setWeights(weights);
  reference.build(finish);
  if (!reference.isParticleActive(5u) || !reference.hasCell(5u)) {
    std::fprintf(stderr, "reference weighted rebuild did not reactivate particle 5\n");
    return false;
  }

  updated.update(finish);
  if (updated.getLastUpdateStats().rebuilt_from_scratch) {
    std::fprintf(stderr, "reactivation path fell back to a full weighted rebuild\n");
    return false;
  }
  if (!updated.isParticleActive(5u)) {
    std::fprintf(stderr, "weighted update failed to reactivate particle 5\n");
    return false;
  }
  if (!updated.hasCell(5u) || updated.getCellIndexForParticle(5u) == vor::noNbr) {
    std::fprintf(stderr, "reactivated particle 5 is missing a packed cell entry\n");
    return false;
  }

  vector<uint2> particleIds(finish.size());
  for (size_t i = 0; i < particleIds.size(); ++i)
    particleIds[i] = static_cast<uint2>(i);

  const real_t tol = 1.0e-10;
  if (!compareByParticleId<real_t>(updated, reference, particleIds, tol, "reactivation compare"))
    return false;

  return true;
}

bool testWeightedUpdateActivityMaskSemantics() {
  typedef double real_t;

  Box<real_t> box(array<real_t, 3>{1.0, 1.0, 1.0});
  PowerCellComplex<real_t> updated(&box);
  PowerCellComplex<real_t> reference(&box);
  vector<array<real_t, 3> > pos = emptyCellBasePositions();
  vector<real_t> weights(pos.size(), real_t(0));
  weights[0] = real_t(-0.05);

  updated.setWeights(weights);
  updated.build(pos);
  if (!updated.isParticleActive(5u) || updated.hasCell(5u)) {
    std::fprintf(stderr, "mask update test requires particle 5 to start active without a cell\n");
    return false;
  }

  vector<uint8_t> inactiveMask(pos.size(), 1u);
  inactiveMask[5] = 0u;
  updated.update(pos, inactiveMask);
  if (updated.isParticleActive(5u) || updated.hasCell(5u) ||
      updated.getCellIndexForParticle(5u) != vor::noNbr) {
    std::fprintf(stderr, "disabling particle 5 did not clear active/hasCell state\n");
    return false;
  }

  vector<uint8_t> activeMask(pos.size(), 1u);
  updated.update(pos, activeMask);
  if (!updated.isParticleActive(5u)) {
    std::fprintf(stderr, "re-enabling particle 5 did not restore active state\n");
    return false;
  }
  if (updated.hasCell(5u) || updated.getCellIndexForParticle(5u) != vor::noNbr) {
    std::fprintf(stderr, "re-enabled particle 5 unexpectedly required immediate cell ownership\n");
    return false;
  }

  reference.setWeights(weights);
  reference.build(pos);
  const vector<uint2>& cellIds = reference.getCellParticleIds();
  const real_t tol = 1.0e-10;
  if (!compareByParticleId<real_t>(updated, reference, cellIds, tol, "masked-update compare"))
    return false;

  return true;
}

bool testRenumberPreservesActiveNoCellParticles() {
  typedef double real_t;

  Box<real_t> box(array<real_t, 3>{1.0, 1.0, 1.0});
  PowerCellComplex<real_t> power(&box);
  vector<array<real_t, 3> > pos = emptyCellBasePositions();
  vector<real_t> weights(pos.size(), real_t(0));
  weights[0] = real_t(-0.05);

  power.setWeights(weights);
  power.build(pos);
  if (!power.isParticleActive(5u) || power.hasCell(5u)) {
    std::fprintf(stderr, "renumber test requires particle 5 to start active without a cell\n");
    return false;
  }

  power.deactivateParticles(vector<uint2>(1, 2u));
  if (power.isParticleActive(2u)) {
    std::fprintf(stderr, "renumber test failed to deactivate particle 2 before compaction\n");
    return false;
  }

  const vor::ParticleRenumberResult map = power.renumberParticles(pos, false);
  if (map.old_to_new[2] != vor::noNbr) {
    std::fprintf(stderr, "deleted particle 2 survived weighted renumbering\n");
    return false;
  }
  if (map.old_to_new[5] == vor::noNbr) {
    std::fprintf(stderr, "active no-cell particle 5 was dropped during weighted renumbering\n");
    return false;
  }

  const uint2 newId = map.old_to_new[5];
  if (!power.isParticleActive(newId)) {
    std::fprintf(stderr, "renumbered active no-cell particle lost active state\n");
    return false;
  }
  if (power.hasCell(newId) || power.getCellIndexForParticle(newId) != vor::noNbr) {
    std::fprintf(stderr, "renumbered active no-cell particle unexpectedly owns a cell\n");
    return false;
  }
  if (pos.size() != weights.size() - 1u) {
    std::fprintf(stderr, "weighted renumbering produced %zu particles, expected %zu\n", pos.size(),
                 weights.size() - 1u);
    return false;
  }

  return true;
}

bool testWeightChangeUsesIncrementalUpdate() {
  typedef double real_t;

  std::mt19937_64 rng(23);
  std::uniform_real_distribution<real_t> uni(real_t(0), real_t(1));

  Box<real_t> box(array<real_t, 3>{1.0, 1.0, 1.0});
  PowerCellComplex<real_t> updated(&box);
  PowerCellComplex<real_t> reference(&box);

  vector<array<real_t, 3> > pos(64);
  for (size_t i = 0; i < pos.size(); ++i)
    for (uint1 k = 0; k < 3; ++k)
      pos[i][k] = uni(rng);

  vector<real_t> initialWeights(pos.size(), real_t(0));
  vector<real_t> changedWeights = initialWeights;
  changedWeights[0] = real_t(-0.01);
  changedWeights[7] = real_t(0.015);

  updated.setWeights(initialWeights);
  updated.build(pos);
  updated.setWeights(changedWeights);
  updated.update(pos);

  if (updated.getLastUpdateStats().rebuilt_from_scratch) {
    std::fprintf(stderr, "pure weight change triggered a full weighted rebuild\n");
    return false;
  }

  reference.setWeights(changedWeights);
  reference.build(pos);

  vector<uint2> particleIds(pos.size());
  for (size_t i = 0; i < particleIds.size(); ++i)
    particleIds[i] = static_cast<uint2>(i);

  const real_t tol = 1.0e-10;
  if (!compareByParticleId<real_t>(updated, reference, particleIds, tol, "weight-change compare"))
    return false;

  return true;
}

}  // namespace

int main() {
  if (!testZeroWeightsMatchVoronoi())
    return 1;
  if (!testActiveWithoutCellState())
    return 1;
  if (!testInactiveParticleReactivationViaUpdate())
    return 1;
  if (!testWeightedUpdateActivityMaskSemantics())
    return 1;
  if (!testRenumberPreservesActiveNoCellParticles())
    return 1;
  if (!testWeightChangeUsesIncrementalUpdate())
    return 1;

  std::puts("power-cell regression tests passed");
  return 0;
}
