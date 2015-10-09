#ifndef VOR_SIMULATION_H
#define VOR_SIMULATION_H

#include "voronoi.hpp"
#include "simulation.hpp"
#include <boost/random.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <algorithm>

using std::vector;
using vor::Array;
using vor::uint0;
using vor::uint1;
using vor::uint2;
using vor::CellGeometry;

namespace vor {

  template<typename real_t = float>
  class Simulation
  {
  public:
    Simulation(): m_complex(&m_box), m_time(0), m_dens(0){};
    virtual bool init();
    virtual bool restart(real_t time, const vector<Array<real_t, 3> > & pos, const vector<Array<real_t, 3> > & vel);
    virtual void step(int numTimeSteps, real_t dt) {}
    virtual real_t getKineticEnergy() const;
    virtual real_t getInternalEnergy() const {}
    void setTime(real_t time) {m_time=time;}
    void setL(Array<real_t, 3> L) {m_box.setL(L);}
    void setVelocities(const vector<Array<real_t, 3> > & vel) {m_vel = vel;}
    void setPositions(const vector<Array<real_t, 3> > & pos) {m_pos = pos;}    
    void setMassDensity(real_t dens) {m_dens=dens;}
    real_t getTime() const {return m_time;}
    const vector<Array<real_t, 3> > & getPositions() const {return m_pos;}
    vector<Array<real_t, 3> > & getVelocities() {return m_vel;}
    const vector<Array<real_t, 3> > & getVelocities() const {return m_vel;}
    const CellComplex<real_t> & getCellComplex() const {return m_complex;}
    void putInBox() {m_box.putInBox(m_pos);}
  protected:
    Box<real_t> m_box;
    CellComplex<real_t> m_complex;
    vector<Array<real_t, 3> > m_pos;
    vector<Array<real_t, 3> > m_vel;
    vector<real_t > m_masses;
    real_t m_time, m_dens;
  };

  template<typename real_t = float>
  class ExplicitEuler: public Simulation<real_t>
  {
  public:
    ExplicitEuler(): Simulation<real_t>() {}
    virtual bool init();
    virtual void step(int numTimeSteps, real_t t);
    virtual real_t getInternalEnergy() const;
    void setTypes(const vector<uint0> types) {this->m_complex.getTypes() = types;}
    void setPressure(real_t press) {m_pressEq=press;}
  protected:
    virtual void computeForces();    
    vector< Array<real_t, 3> > m_forces;
    real_t m_volAvg, m_pressEq;
  };

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

  template<typename real_t = float>
  class NavierStokes: public ExplicitEuler<real_t>
  {
  public:
    NavierStokes():ExplicitEuler<real_t>(), m_visc(0), m_bulkVisc(0) {}
    void setViscosity(real_t visc) {m_visc = visc;}
    void setBulkViscosity(real_t visc) {m_bulkVisc = visc;}
  protected:
    virtual void computeForces();
    real_t m_visc, m_bulkVisc;
  };
  
  template<typename real_t>
  bool Simulation<real_t>::init()
  {
    const Array<real_t, 3> & L(this->m_box.getL());
    real_t vol = L[0]*L[1]*L[2];
    if (vol ==0){
      fprintf(stderr,"one of the box dimensions equals zero\n");
      return false;
    }
    size_t numPart = m_pos.size();
    if (numPart ==0){
      fprintf(stderr,"number of positions equals zero\n");
      return false;
    }
    if (m_vel.size() != numPart){
      fprintf(stderr,"vector of velocities equals %lu, but %lu is expected\n", m_vel.size(), numPart);
      return false;
    }
    if(m_dens ==0){
      fprintf(stderr,"density not set\n");
      return false;
    }
    real_t mass = vol*m_dens/numPart;
    m_masses.resize(numPart);
    for(size_t i=0; i<numPart; ++i)
      m_masses[i]=mass;
    this->m_complex.build(this->m_pos);
    return true;
  }

  template<typename real_t>
  bool Simulation<real_t>::restart(real_t time, const vector<Array<real_t, 3> > & pos, const vector<Array<real_t, 3> > & vel)
  {
    if(m_pos.size() != pos.size() || m_vel.size() != vel.size()){
      fprintf(stderr,"sizes of position and velocity vectors have changed\n");
      return false;
    }else{
      setTime(time);
      setPositions(pos);
      setVelocities(vel);
    }
    return this->init();
  }

  template<typename real_t>
  bool ExplicitEuler<real_t>::init()
  {
    const Array<real_t, 3> & L(this->m_box.getL());
    m_volAvg = L[0]*L[1]*L[2]/(this->m_pos.size());    
    if (!Simulation<real_t>::init())
      return false;
    if (m_pressEq==0){
      fprintf(stderr,"pressure is not set\n");
      return false;
    }
    m_forces.resize(this->m_pos.size());
    this->computeForces();
    return true;
  }
    
  template<typename real_t>
  real_t Simulation<real_t>::getKineticEnergy() const
  {
    real_t E=0;
#pragma omp parallel for reduction(+:E)
    for(size_t i=0; i < m_vel.size(); ++i){
      real_t Esub = 0;
      for(uint0 k=0; k<3; ++k)
	Esub += m_vel[i][k]*m_vel[i][k];
      E += m_masses[i]*Esub;
    }
    E *= 0.5;
    return E;
  }
  
  template<typename real_t>
  void ExplicitEuler<real_t>::step(int numTimeSteps, real_t dt){
    real_t halfDt=0.5*dt;
    for(int m(0); m < numTimeSteps; ++m){
#pragma omp parallel for
      for(size_t i=0; i< this->m_vel.size(); ++i)
	for(uint0 k=0; k<3; ++k)
	  this->m_vel[i][k] += (m_forces[i][k]/this->m_masses[i])*halfDt;
#pragma omp parallel for
      for(size_t i=0; i< this->m_pos.size(); ++i)
	for(uint0 k=0; k<3; ++k)
	  this->m_pos[i][k] += this->m_vel[i][k]*dt;
      this->m_complex.update(this->m_pos);
      this->computeForces();
#pragma omp parallel for
      for(size_t i=0; i< this->m_vel.size(); ++i)
	for(uint0 k=0; k<3; ++k)
	  this->m_vel[i][k] += (m_forces[i][k]/this->m_masses[i])*halfDt;
      this->m_time +=dt;
    }
  }
  
  template<typename real_t>
  real_t ExplicitEuler<real_t>::getInternalEnergy() const
  {
    const vector<CellGeometry<real_t> > & geoms(this->m_complex.getGeoms());
    real_t E=0;
#pragma omp parallel for reduction(-:E)
    for(size_t i=0; i < this->m_pos.size(); ++i)
      E -= m_pressEq*m_volAvg*log(geoms[i].getVolume()/m_volAvg);
    return E;
  }

  template<typename real_t>
  void ExplicitEuler<real_t>::computeForces()
  {
    for (int i =0; i< this->m_pos.size(); ++i)
       for(uint k(0); k<3; ++k)
 	m_forces[i][k] = 0.0;
     vector<CellGeometry<real_t> > & geoms(this->m_complex.getGeoms());
#pragma omp parallel for
    for(size_t i=0; i< geoms.size(); ++i){
      const vector< Array<real_t, 3> > & dV(geoms[i].getdV());
      const Cell<real_t> & cell(this->m_complex.getCells()[i]);
      geoms[i].diffVolume();
      real_t press = (m_pressEq*m_volAvg)/geoms[i].getVolume();
      for(uint0 j=0; j< cell.numFacets(); ++j){
	uint2 nbr = cell.getNbr(j);
	for(uint0 k=0; k<3; ++k){
	  real_t df = press*dV[j][k];
#pragma omp atomic
	  m_forces[nbr][k] += df;
#pragma omp atomic
	  m_forces[i][k] -= df;
	}
      }
    }
  }

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

    template<typename real_t>
  void NavierStokes<real_t>::computeForces()
  {
    for (int i =0; i< this->m_pos.size(); ++i)
      for(uint k(0); k<3; ++k)
 	this->m_forces[i][k] = 0.0;
    const real_t two_third = 2.0/3.0;
    vector<CellGeometry<real_t> > & geoms(this->m_complex.getGeoms());
#pragma omp parallel for
    for(size_t i=0; i< geoms.size(); ++i){
      geoms[i].computeAll();
      //geoms[i].diffVolume();
      const Cell<real_t> & cell(this->m_complex.getCells()[i]);
      Array<Array<real_t, 3>, 3> gradVel(geoms[i].velocityGradient(this->m_vel));
      real_t press = (this->m_pressEq*this->m_volAvg)/geoms[i].getVolume();
      real_t divVel=0;
      Array<Array<real_t, 3>, 3> stress;
      // compute stress in cell
      for(uint0 k=0; k<3; ++k){
	for(uint0 m=0; m<3; ++m)
	  stress[k][m] = m_visc*(gradVel[k][m]+gradVel[m][k]);
	divVel += gradVel[k][k];
      }
      for(uint0 k=0; k<3; ++k)
	stress[k][k] += (-press + (m_bulkVisc-two_third*m_visc)*divVel);
      const vector< Array< Array< Array<real_t, 3>, 3 >, 3> > & omega(geoms[i].getOmega());
      for(uint0 j=0; j< cell.numFacets(); ++j){
	uint2 nbr = cell.getNbr(j);
	for(uint0 k=0; k<3; ++k){
	  real_t df = 0;
	  for(int l(0); l<3; ++l)
	    for(int m(0); m<3; ++m)
	      df -= omega[j][k][l][m]*stress[l][m];
	  //df = geoms[i].getdV()[j][k]*press;
#pragma omp atomic
	  this->m_forces[nbr][k] += df;
#pragma omp atomic
	  this->m_forces[i][k] -= df;
	}
      }
    }
  }
  
}
#endif
