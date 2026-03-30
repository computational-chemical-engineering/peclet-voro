/**
 * @file simulation.hpp
 * @brief Particle-dynamics simulation classes built on the Voronoi cell complex.
 *
 * Provides a hierarchy of simulation classes:
 *  - Simulation        – base class with time-stepping and gradient computation
 *  - ExplicitEuler     – compressible Euler solver (velocity-Verlet)
 *  - NavierStokes      – viscous Navier-Stokes solver
 *  - Incompressible    – divergence-free pressure-projection solver
 *  - IntfDyn           – multiphase interface-tension dynamics
 */

#pragma once

#include <algorithm>
#include <boost/random.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <vector>

#include "voronoi.hpp"

namespace vor {

template <typename real_t = float>
class Simulation {
 public:
  Simulation() : m_complex(&m_box), m_time(0), m_dens(0) {};
  virtual bool init();
  virtual bool restart(real_t time, const std::vector<std::array<real_t, 3> >& pos,
                       const std::vector<std::array<real_t, 3> >& vel);
  virtual void step(int numTimeSteps, real_t dt) {}
  virtual real_t getKineticEnergy() const;
  virtual real_t getInternalEnergy() const { return 0; }
  void setTime(real_t time) { m_time = time; }
  void setL(std::array<real_t, 3> L) { m_box.setL(L); }
  void setVelocities(const std::vector<std::array<real_t, 3> >& vel) { m_vel = vel; }
  void setPositions(const std::vector<std::array<real_t, 3> >& pos) { m_pos = pos; }
  void setMasses(const std::vector<real_t>& masses) { m_masses = masses; }
  void setMassDensity(real_t dens) { m_dens = dens; }
  real_t getTime() const { return m_time; }
  std::vector<std::array<real_t, 3> >& getPositions() { return m_pos; }
  const std::vector<std::array<real_t, 3> >& getPositions() const { return m_pos; }
  std::vector<std::array<real_t, 3> >& getVelocities() { return m_vel; }
  const std::vector<std::array<real_t, 3> >& getVelocities() const { return m_vel; }
  CellComplex<real_t>& getCellComplex() { return m_complex; }
  const CellComplex<real_t>& getCellComplex() const { return m_complex; }
  void putInBox() { m_box.putInBox(m_pos); }
  template <std::size_t m>
  void computeGradients(const std::vector<std::array<real_t, m> >& phi,
                        std::vector<std::array<std::array<real_t, m>, 3> >& grads) const;
  template <std::size_t m>
  void computeGradients2(const std::vector<std::array<real_t, m> >& phi,
                         std::vector<std::array<std::array<real_t, m>, 3> >& grads) const;
  void computeGradients(const std::vector<real_t>& phi,
                        std::vector<std::array<real_t, 3> >& grads) const;

 protected:
  Box<real_t> m_box;
  CellComplex<real_t> m_complex;
  std::vector<std::array<real_t, 3> > m_pos;
  std::vector<std::array<real_t, 3> > m_vel;
  std::vector<real_t> m_masses;
  real_t m_time, m_dens;
};

template <typename real_t = float>
class ExplicitEuler : public Simulation<real_t> {
 public:
  ExplicitEuler() : Simulation<real_t>() {}
  virtual bool init();
  virtual void step(int numTimeSteps, real_t t);
  virtual real_t getInternalEnergy() const;
  void setTypes(const std::vector<uint0>& types) { this->m_complex.getTypes() = types; }
  void setPressure(real_t press) { m_pressEq = press; }

 protected:
  virtual void computeForces();
  std::vector<std::array<real_t, 3> > m_forces;
  real_t m_volAvg, m_pressEq;
};

template <typename real_t = float>
class NavierStokes : public ExplicitEuler<real_t> {
 public:
  NavierStokes() : ExplicitEuler<real_t>(), m_extForceDens(0) {}
  virtual bool init();
  void setViscosity(real_t visc);
  void setBulkViscosity(real_t visc);
  void setViscosities(const std::vector<real_t>& visc) { m_visc = visc; }
  void setBulkViscosities(const std::vector<real_t>& visc) { m_bulkVisc = visc; }
  void setExtForceDens(real_t f) { m_extForceDens = f; }

 protected:
  virtual void computeForces();
  void computeViscousForces();
  void addExternalForces();
  std::vector<real_t> m_visc, m_bulkVisc;
  real_t m_extForceDens;
};

template <typename real_t = float>
class Incompressible : public NavierStokes<real_t> {
 public:
  Incompressible() : NavierStokes<real_t>() {}
  void buildConstraintMatrix();
};

template <typename real_t = float>
class IntfDyn : public NavierStokes<real_t> {
 public:
  IntfDyn() : NavierStokes<real_t>(), m_intfTension(0) {}
  real_t getIntfEnergy() const;
  void setIntfTension(real_t intfTension, uint0 i, uint0 j);

 private:
  virtual void computeForces();
  void computeIntfForces();
  real_t getIntfTension(uint0 i, uint0 j) const;
  uint nTypes;
  std::vector<real_t> m_intfTension;
};

// template<typename real_t = float>
// class StressModel
// {
// public:
//   virtual void update(std::array<std::array<real_t, 3>, 3> & stress, const
//   std::array<std::array<real_t, 3>, 3> & gradV) {} virtual void setParameter(char * name, real_t
//   value);
// };

// template<typename real_t = float>
// class ViscousStress: public StressModel<real_t>
// {
// public:
//   ViscousStress():m_two_third(2.0/3.0) {}
//   virtual void update(std::array<std::array<real_t, 3>, 3> & stress, const
//   std::array<std::array<real_t, 3>, 3> & gradV); virtual void setParameter(char * name, real_t
//   value);
// private:
//   const real_t m_two_third;
//   real_t m_visc, m_bulkVisc;
// };

template <typename real_t>
bool Simulation<real_t>::init() {
  const std::array<real_t, 3>& L(this->m_box.getL());
  real_t vol = L[0] * L[1] * L[2];
  if (vol == 0) {
    fprintf(stderr, "one of the box dimensions equals zero\n");
    return false;
  }
  size_t numPart = m_pos.size();
  if (numPart == 0) {
    fprintf(stderr, "number of positions equals zero\n");
    return false;
  }
  if (m_vel.size() != numPart) {
    fprintf(stderr, "vector of velocities equals %lu, but %lu is expected\n", m_vel.size(),
            numPart);
    return false;
  }
  if (m_dens == 0 & m_masses.size() != 0) {
    fprintf(stderr, "density not set\n");
    return false;
  }
  if (m_masses.size() != numPart) {
    real_t mass = vol * m_dens / numPart;
    m_masses.resize(numPart);
    for (size_t i = 0; i < numPart; ++i)
      m_masses[i] = mass;
  }
  this->m_complex.build(this->m_pos);
  return true;
}

template <typename real_t>
bool Simulation<real_t>::restart(real_t time, const std::vector<std::array<real_t, 3> >& pos,
                                 const std::vector<std::array<real_t, 3> >& vel) {
  if (m_pos.size() != pos.size() || m_vel.size() != vel.size()) {
    fprintf(stderr, "sizes of position and velocity vectors have changed\n");
    return false;
  } else {
    setTime(time);
    setPositions(pos);
    setVelocities(vel);
  }
  return this->init();
}

template <typename real_t>
template <std::size_t m>
void Simulation<real_t>::computeGradients(
    const std::vector<std::array<real_t, m> >& phi,
    std::vector<std::array<std::array<real_t, m>, 3> >& grads) const {
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
  if (this->m_complex.numParticles() > phi.size())
    return;
  grads.resize(phi.size());
#pragma omp parallel for
  for (size_t i = 0; i < grads.size(); ++i)
    for (int p = 0; p < m; ++p)
      for (uint0 k = 0; k < 3; ++k)
        grads[i][p][k] = 0;
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const GeometryView<real_t> geom(this->m_complex.getGeometryView(i));
    const uint2 selfId = cell.getID();
    for (uint0 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      for (int p = 0; p < m; ++p)
        for (uint0 k = 0; k < 3; ++k) {
          real_t dgrad = -phi[selfId][p] * geom.getdV(j)[k];
#pragma omp atomic
          grads[nbr][p][k] += dgrad;
#pragma omp atomic
          grads[selfId][p][k] -= dgrad;
        }
    }
  }
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const uint2 selfId = this->m_complex.getCellView(i).getID();
    const real_t vol = geometry.getView(i).getVolume();
    for (int p = 0; p < m; ++p)
      for (uint0 k = 0; k < 3; ++k)
        grads[selfId][p][k] /= vol;
  }
}

template <typename real_t>
template <std::size_t m>
void Simulation<real_t>::computeGradients2(
    const std::vector<std::array<real_t, m> >& phi,
    std::vector<std::array<std::array<real_t, m>, 3> >& grads) const {
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
  if (this->m_complex.numParticles() > phi.size())
    return;
  grads.resize(phi.size());
#pragma omp parallel for
  for (size_t i = 0; i < grads.size(); ++i)
    for (int p = 0; p < m; ++p)
      for (uint0 k = 0; k < 3; ++k)
        grads[i][p][k] = 0;
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const GeometryView<real_t> geom(this->m_complex.getGeometryView(i));
    const uint2 selfId = cell.getID();
    for (uint0 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      for (int p = 0; p < m; ++p) {
        real_t dphi = phi[nbr][p] - phi[selfId][p];
        for (uint0 k = 0; k < 3; ++k)
          grads[selfId][p][k] += dphi * geom.getdV(j)[k];
        ;
      }
    }
    real_t vol = geom.getVolume();
    for (int p = 0; p < m; ++p)
      for (uint0 k = 0; k < 3; ++k)
        grads[selfId][p][k] /= vol;
  }
}

template <typename real_t>
void Simulation<real_t>::computeGradients(const std::vector<real_t>& phi,
                                          std::vector<std::array<real_t, 3> >& grads) const {
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
  if (this->m_complex.numParticles() > phi.size())
    return;
  grads.resize(phi.size());
#pragma omp parallel for
  for (size_t i = 0; i < grads.size(); ++i)
    for (uint0 k = 0; k < 3; ++k)
      grads[i][k] = 0;
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const GeometryView<real_t> geom(this->m_complex.getGeometryView(i));
    const uint2 selfId = cell.getID();
    for (uint0 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      for (uint0 k = 0; k < 3; ++k) {
        real_t dgrad = -phi[selfId] * geom.getdV(j)[k];
#pragma omp atomic
        grads[nbr][k] += dgrad;
#pragma omp atomic
        grads[selfId][k] -= dgrad;
      }
    }
  }
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const uint2 selfId = this->m_complex.getCellView(i).getID();
    const real_t vol = geometry.getView(i).getVolume();
    for (uint0 k = 0; k < 3; ++k)
      grads[selfId][k] /= vol;
  }
}

template <typename real_t>
bool ExplicitEuler<real_t>::init() {
  const std::array<real_t, 3>& L(this->m_box.getL());
  if (!Simulation<real_t>::init())
    return false;
  m_volAvg =
      L[0] * L[1] * L[2] / static_cast<real_t>(std::max<size_t>(this->m_complex.numCells(), 1u));
  if (m_pressEq == 0) {
    fprintf(stderr, "pressure is not set, assuming zero pressure\n");
  }
  m_forces.resize(this->m_pos.size());
  this->computeForces();
  return true;
}

template <typename real_t>
real_t Simulation<real_t>::getKineticEnergy() const {
  real_t E = 0;
#pragma omp parallel for reduction(+ : E)
  for (size_t i = 0; i < m_vel.size(); ++i) {
    real_t Esub = 0;
    for (uint0 k = 0; k < 3; ++k)
      Esub += m_vel[i][k] * m_vel[i][k];
    E += m_masses[i] * Esub;
  }
  E *= 0.5;
  return E;
}

template <typename real_t>
void ExplicitEuler<real_t>::step(int numTimeSteps, real_t dt) {
  real_t halfDt = 0.5 * dt;
  for (int m(0); m < numTimeSteps; ++m) {
#pragma omp parallel for
    for (size_t i = 0; i < this->m_vel.size(); ++i)
      for (uint0 k = 0; k < 3; ++k)
        this->m_vel[i][k] += (m_forces[i][k] / this->m_masses[i]) * halfDt;
#pragma omp parallel for
    for (size_t i = 0; i < this->m_pos.size(); ++i)
      for (uint0 k = 0; k < 3; ++k)
        this->m_pos[i][k] += this->m_vel[i][k] * dt;
    this->m_complex.update(this->m_pos);
    this->computeForces();
#pragma omp parallel for
    for (size_t i = 0; i < this->m_vel.size(); ++i)
      for (uint0 k = 0; k < 3; ++k)
        this->m_vel[i][k] += (m_forces[i][k] / this->m_masses[i]) * halfDt;
    this->m_time += dt;
  }
}

template <typename real_t>
real_t ExplicitEuler<real_t>::getInternalEnergy() const {
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
  real_t E = 0;
#pragma omp parallel for reduction(- : E)
  for (size_t i = 0; i < geometry.numCells(); ++i)
    E -= m_pressEq * m_volAvg * log(geometry.getView(i).getVolume() / m_volAvg);
  return E;
}

template <typename real_t>
void ExplicitEuler<real_t>::computeForces() {
#pragma omp parallel for
  for (int i = 0; i < this->m_pos.size(); ++i)
    for (uint k(0); k < 3; ++k)
      m_forces[i][k] = 0.0;
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const GeometryView<real_t> geom(this->m_complex.getGeometryView(i));
    const uint2 selfId = cell.getID();
    real_t press = (m_pressEq * m_volAvg) / geom.getVolume();
    for (uint0 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      for (uint0 k = 0; k < 3; ++k) {
        real_t df = press * geom.getdV(j)[k];
#pragma omp atomic
        m_forces[nbr][k] += df;
#pragma omp atomic
        m_forces[selfId][k] -= df;
      }
    }
  }
}

//   template<typename real_t>
//   void NavierStokes<real_t>::computeForces()
//   {
template <typename real_t>
void NavierStokes<real_t>::computeForces() {
#pragma omp parallel for
  for (int i = 0; i < this->m_pos.size(); ++i)
    for (uint k(0); k < 3; ++k)
      this->m_forces[i][k] = 0.0;
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const GeometryView<real_t> geom(this->m_complex.getGeometryView(i));
    const uint2 selfId = cell.getID();
    real_t press = (this->m_pressEq * this->m_volAvg) / geom.getVolume();
    for (uint0 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      for (uint0 k = 0; k < 3; ++k) {
        real_t df = geom.getdV(j)[k] * press;
#pragma omp atomic
        this->m_forces[nbr][k] += df;
#pragma omp atomic
        this->m_forces[selfId][k] -= df;
      }
    }
  }
  computeViscousForces();
  addExternalForces();
}

template <typename real_t>
void NavierStokes<real_t>::computeViscousForces() {
  const real_t two_third = 2.0 / 3.0;
  std::vector<std::array<std::array<real_t, 3>, 3> > stresses;
  this->computeGradients(this->m_vel, stresses);
#pragma omp parallel for
  for (size_t i = 0; i < stresses.size(); ++i) {
    real_t divVel = 0;
    const std::array<std::array<real_t, 3>, 3>& gradV(stresses[i]);
    std::array<std::array<real_t, 3>, 3> stress;
    // compute viscous stress in cell
    for (uint0 k = 0; k < 3; ++k) {
      for (uint0 m = 0; m < 3; ++m) {
        stress[k][m] = m_visc[i] * (gradV[k][m] + gradV[m][k]);
      }
      divVel += gradV[k][k];
    }
    for (uint0 k = 0; k < 3; ++k)
      stress[k][k] += (m_bulkVisc[i] - two_third * m_visc[i]) * divVel;
    stresses[i] = stress;
  }
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const GeometryView<real_t> geom(this->m_complex.getGeometryView(i));
    const uint2 selfId = cell.getID();
    std::array<real_t, 3> f;
    for (uint k(0); k < 3; ++k)
      f[k] = 0;
    for (uint0 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      for (uint0 k = 0; k < 3; ++k)
        for (uint0 m = 0; m < 3; ++m)
          f[k] += (stresses[nbr][k][m] - stresses[selfId][k][m]) * geom.getdV(j)[m];
    }
    for (uint k(0); k < 3; ++k)
      this->m_forces[selfId][k] += f[k];
  }
}

template <typename real_t>
void NavierStokes<real_t>::addExternalForces() {
  static const real_t pi = 3.1415926535897932384626433832795;
  if (m_extForceDens == 0)
    return;
  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
  const std::array<real_t, 3>& L(this->m_box.getL());
#pragma omp parallel for
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    const uint2 particleId = this->m_complex.getCellView(i).getID();
    real_t sn = sin(2.0 * pi * this->m_pos[particleId][1] / L[1]);
    this->m_forces[particleId][0] += m_extForceDens * geometry.getView(i).getVolume() * sn;
  }
}

template <typename real_t>
bool NavierStokes<real_t>::init() {
  const std::array<real_t, 3>& L(this->m_box.getL());
  if (!Simulation<real_t>::init())
    return false;
  this->m_volAvg =
      L[0] * L[1] * L[2] / static_cast<real_t>(std::max<size_t>(this->m_complex.numCells(), 1u));
  if (this->m_pressEq == 0) {
    fprintf(stderr, "pressure is not set, assuming zero pressure\n");
  }
  size_t numPart = this->m_pos.size();
  this->m_forces.resize(numPart);
  if (m_visc.size() == 0) {
    fprintf(stderr, "viscosities not set, assuming zero\n");
    m_visc.resize(numPart, 0);
  } else if (m_visc.size() == 1) {
    m_visc.resize(numPart);
    for (int i = 1; i < m_visc.size(); ++i)
      m_visc[i] = m_visc[0];
  } else if (m_visc.size() != numPart) {
    fprintf(stderr, "Vector of viscosities incorrect size. Setting all viscosities to zero.\n");
    m_visc.clear();
    m_visc.resize(numPart, 0);
  }
  if (m_bulkVisc.size() == 0) {
    fprintf(stderr, "bulk viscosities not set, assuming zero\n");
    m_bulkVisc.resize(numPart, 0);
  } else if (m_bulkVisc.size() == 1) {
    m_bulkVisc.resize(numPart);
    for (int i = 1; i < m_bulkVisc.size(); ++i)
      m_bulkVisc[i] = m_bulkVisc[0];
  } else if (m_bulkVisc.size() != numPart) {
    fprintf(stderr,
            "Vector of bulk-viscosities incorrect size. Setting all viscosities to zero.\n");
    m_bulkVisc.clear();
    m_bulkVisc.resize(numPart, 0);
  }
  this->computeForces();
  return true;
}

template <typename real_t>
void NavierStokes<real_t>::setViscosity(real_t visc) {
  m_visc.resize(1);
  m_visc[0] = visc;
}

template <typename real_t>
void NavierStokes<real_t>::setBulkViscosity(real_t visc) {
  m_bulkVisc.resize(1);
  m_bulkVisc[0] = visc;
}

template <typename real_t>
void IntfDyn<real_t>::setIntfTension(real_t intfTension, uint0 i, uint0 j) {
  if (i != j) {
    uint2 m = (i < j ? (((uint2)j) * ((uint2)j - 1)) / 2 + ((uint2)i)
                     : (((uint2)i) * ((uint2)i - 1)) / 2 + ((uint2)j));
    if (m >= m_intfTension.size())
      m_intfTension.resize(m + 1);
    m_intfTension[m] = intfTension;
  }
}

template <typename real_t>
inline real_t IntfDyn<real_t>::getIntfTension(uint0 i, uint0 j) const {
  if (i == j)
    return 0;
  uint2 m = (i < j ? (((uint2)j) * ((uint2)j - 1)) / 2 + ((uint2)i)
                   : (((uint2)i) * ((uint2)i - 1)) / 2 + ((uint2)j));
  return m_intfTension[m];
}

template <typename real_t>
void IntfDyn<real_t>::computeForces() {
  this->NavierStokes<real_t>::computeForces();
  computeIntfForces();
}

template <typename real_t>
void IntfDyn<real_t>::computeIntfForces() {
  const std::vector<uint0>& types(this->m_complex.getTypes());
#pragma omp parallel for
  for (size_t i = 0; i < this->m_complex.numCells(); ++i) {
    std::vector<uint2> indxFacets;
    std::vector<std::array<real_t, 3> > grad;
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const uint2 selfId = cell.getID();
    if (types[selfId] == 0)
      continue;
    const CellGeometry<real_t>& geom(this->m_complex.getGeoms()[i]);
    for (uint1 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      if (types[nbr] < types[selfId]) {
        const std::array<real_t, 3>& areaV(geom.getAreas()[j]);
        const real_t area =
            std::sqrt(areaV[0] * areaV[0] + areaV[1] * areaV[1] + areaV[2] * areaV[2]);
        geom.gradFacetAreaSq(j, indxFacets, grad);
        const real_t intfTension = getIntfTension(types[selfId], types[nbr]);
        for (size_t m = 0; m < indxFacets.size(); ++m) {
          const uint2 nbr2 = cell.getNbr(indxFacets[m]);
          for (uint0 k = 0; k < 3; ++k) {
            const real_t df = -intfTension * grad[m][k] / (2.0 * area);
#pragma omp atomic
            this->m_forces[nbr2][k] += df;
#pragma omp atomic
            this->m_forces[selfId][k] -= df;
          }
        }
      }
    }
  }
}

template <typename real_t>
real_t IntfDyn<real_t>::getIntfEnergy() const {
  real_t E = 0;
  const std::vector<uint0>& types(this->m_complex.getTypes());
#pragma omp parallel for reduction(+ : E)
  for (size_t i = 0; i < this->m_complex.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const uint2 selfId = cell.getID();
    if (types[selfId] == 0)
      continue;
    const CellGeometry<real_t>& geom(this->m_complex.getGeoms()[i]);
    for (uint1 j = 0; j < cell.numFacets(); ++j) {
      const uint2 nbr = cell.getNbr(j);
      if (types[nbr] < types[selfId]) {
        const std::array<real_t, 3>& areaV(geom.getAreas()[j]);
        const real_t area =
            std::sqrt(areaV[0] * areaV[0] + areaV[1] * areaV[1] + areaV[2] * areaV[2]);
        E += getIntfTension(types[selfId], types[nbr]) * area;
      }
    }
  }
  return E;
}

template <typename real_t>
void Incompressible<real_t>::buildConstraintMatrix() {
  clock_t start;
  real_t duration;
  start = clock();
  // find next nearest neighbors
  for (size_t i = 0; i < this->m_complex.numCells(); ++i) {
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    std::vector<uint2> nbrs;
    nbrs.reserve(300);
    for (size_t j = 0; j < cell.numFacets(); ++j) {
      uint2 nbr = cell.getNbr(static_cast<uint1>(j));
      nbrs.push_back(nbr);
      const uint2 nbrCellIndex = this->m_complex.getCellIndexForParticle(nbr);
      if (nbrCellIndex == vor::noNbr)
        continue;
      const CellView<real_t> nbrCell(this->m_complex.getCellView(nbrCellIndex));
      for (size_t k = 0; k < nbrCell.numFacets(); ++k)
        nbrs.push_back(nbrCell.getNbr(static_cast<uint1>(k)));
    }
    std::sort(nbrs.begin(), nbrs.end());
    size_t iNew = 0;
    for (size_t iOld = 0; iOld < nbrs.size(); ++iNew) {
      nbrs[iNew] = nbrs[iOld];
      ++iOld;
      for (; iOld < nbrs.size() && nbrs[iOld] == nbrs[iNew]; ++iOld) {
      }
    }
    nbrs.resize(iNew + 1);
  }
  duration = real_t(clock() - start);
  printf("sparsity pattern in %f seconds\n", duration / (real_t(CLOCKS_PER_SEC)));

  const GeometryArena<real_t>& geometry(this->m_complex.getGeometryArena());
  std::vector<MatrixEntry<real_t> > matdV;

  start = clock();
  matdV.reserve(60 * geometry.numCells());
  for (size_t i = 0; i < geometry.numCells(); ++i) {
    MatrixEntry<real_t> triplet;
    const GeometryView<real_t> geom(this->m_complex.getGeometryView(i));
    const CellView<real_t> cell(this->m_complex.getCellView(i));
    const uint2 selfId = cell.getID();
    real_t dVii[3];
    for (uint0 k = 0; k < 3; ++k)
      dVii[k] = 0;
    for (uint0 j = 0; j < cell.numFacets(); ++j) {
      for (uint0 k = 0; k < 3; ++k) {
        dVii[k] -= geom.getdV(j)[k];
        triplet.col() = selfId;
        triplet.row() = (3 * cell.getNbr(j)) + k;
        triplet.value() = geom.getdV(j)[k];
        matdV.push_back(triplet);
      }
    }
    for (uint0 k = 0; k < 3; ++k) {
      triplet.col() = selfId;
      triplet.row() = 3 * selfId + k;
      triplet.value() = dVii[k];
    }
    matdV.push_back(triplet);
  }
  duration = real_t(clock() - start);
  printf("collecting in %f seconds\n", duration / (real_t(CLOCKS_PER_SEC)));
  start = clock();
  //    printf("matdV size per cell: %lu\n", matdV.size()/geoms.size());
  std::sort(matdV.begin(), matdV.end(), CompareMatrixEntryRow<real_t>());
  duration = real_t(clock() - start);
  printf("sort 1 in %f seconds\n", duration / (real_t(CLOCKS_PER_SEC)));

  start = clock();
  std::vector<MatrixEntry<real_t> > matConstr;
  matConstr.reserve(450 * geometry.numCells());
  size_t iEnd = 0;
  while (iEnd < matdV.size()) {
    size_t row = matdV[iEnd].row();
    size_t iCell = row / 3;
    size_t iBegin;
    for (iBegin = iEnd; (matdV[iBegin].row() == matdV[iEnd].row()) && (iEnd < matdV.size());
         ++iEnd) {
    }
    // printf("\n");
    for (size_t i = iBegin; i < iEnd; ++i)
      for (size_t j = iBegin; j < iEnd && matdV[j].col() <= matdV[i].col();
           ++j) {  // only construct lower triangular part of matrix
        //	  printf("i j: %lu %lu\n",i,j);
        MatrixEntry<real_t> triplet;
        triplet.col() = matdV[i].col();
        triplet.row() = matdV[j].col();
        triplet.value() = matdV[i].value() * matdV[j].value() / this->m_masses[iCell];
        matConstr.push_back(triplet);
      }
  }
  duration = real_t(clock() - start);
  printf("multiply in %f seconds\n", duration / (real_t(CLOCKS_PER_SEC)));

  //    printf("matConstr size per cell: %lu\n", matConstr.size()/geoms.size());
  start = clock();
  std::sort(matConstr.begin(), matConstr.end(), CompareMatrixEntryRow<real_t>());
  duration = real_t(clock() - start);
  printf("sort 2 %f seconds\n", duration / (real_t(CLOCKS_PER_SEC)));
  start = clock();
  size_t iNew = 0;
  for (size_t iOld = 0; iOld < matConstr.size(); ++iNew) {
    matConstr[iNew] = matConstr[iOld];
    ++iOld;
    for (; iOld < matConstr.size() && matConstr[iOld].row() == matConstr[iNew].row() &&
           matConstr[iOld].col() == matConstr[iNew].col();
         ++iOld)
      matConstr[iNew].value() += matConstr[iOld].value();
  }
  matConstr.resize(iNew + 1);
  duration = real_t(clock() - start);
  printf("summation in %f seconds\n", duration / (real_t(CLOCKS_PER_SEC)));

  printf("matConstr size per cell (after summation): %lu\n",
         matConstr.size() / geometry.numCells());
}

// template<typename real_t>
// void ViscousStress<real_t>::update(std::array<std::array<real_t, 3>, 3> & stress, const
// std::array<std::array<real_t, 3>, 3> & gradV)
// {
//   real_t divVel=0;
//   // compute viscous stress in cell
//   for(uint0 k=0; k<3; ++k){
//     for(uint0 m=0; m<3; ++m){
// 	stress[k][m] = m_visc*(gradV[k][m]+gradV[m][k]);
//     }
//     divVel += gradV[k][k];
//   }
//   for(uint0 k=0; k<3; ++k)
//     stress[k][k] += (m_bulkVisc-m_two_third*m_visc)*divVel;
// }

}  // namespace vor
