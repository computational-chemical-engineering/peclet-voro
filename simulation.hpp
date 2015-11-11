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
    void setMasses(const vector<real_t> & masses) {m_masses = masses;}
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
    void setTypes(const vector<uint0> & types) {this->m_complex.getTypes() = types;}
    void setPressure(real_t press) {m_pressEq=press;}
  protected:
    virtual void computeForces();    
    vector< Array<real_t, 3> > m_forces;
    real_t m_volAvg, m_pressEq;
  };

  template<typename real_t = float>
  class NavierStokes: public ExplicitEuler<real_t>
  {
  public:
    NavierStokes():ExplicitEuler<real_t>() {}
    virtual bool init();
    void setViscosity(real_t visc);
    void setBulkViscosity(real_t visc);
    void setViscosities(const vector<real_t> & visc) {m_visc = visc;}
    void setBulkViscosities(const vector<real_t> & visc) {m_bulkVisc = visc;}
  protected:
    virtual void computeForces();
    vector<real_t> m_visc, m_bulkVisc;
  };

  template<typename real_t = float>
  class IntfDyn: public NavierStokes<real_t>
  {
  public:
    IntfDyn(): NavierStokes<real_t>(), m_intfTension(0) {}
    real_t getIntfEnergy() const;
    void setIntfTension(real_t intfTension, uint0 i, uint0 j);
  private:
    virtual void computeForces();
    void computeIntfForces();
    real_t getIntfTension(uint0 i, uint0 j) const;
    uint nTypes;
    vector<real_t> m_intfTension;
  };

  // template<typename real_t = float>
  // class StressModel
  // {
  // public:
  //   virtual void update(Array<Array<real_t, 3>, 3> & stress, const Array<Array<real_t, 3>, 3> & gradV) {}
  //   virtual void setParameter(char * name, real_t value);
  // };

  // template<typename real_t = float>
  // class ViscousStress: public StressModel<real_t>
  // {
  // public:
  //   ViscousStress():m_two_third(2.0/3.0) {}
  //   virtual void update(Array<Array<real_t, 3>, 3> & stress, const Array<Array<real_t, 3>, 3> & gradV);
  //   virtual void setParameter(char * name, real_t value);
  // private:
  //   const real_t m_two_third;
  //   real_t m_visc, m_bulkVisc;
  // };
  
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
    if(m_dens ==0 & m_masses.size()!=0) {
      fprintf(stderr,"density not set\n");
      return false;
    }
    if (m_masses.size() != numPart){
      real_t mass = vol*m_dens/numPart;
      m_masses.resize(numPart);
      for(size_t i=0; i<numPart; ++i)
	m_masses[i]=mass;
    }
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
#pragma omp parallel for
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


//   template<typename real_t>
//   void NavierStokes<real_t>::computeForces()
//   {
// #pragma omp parallel for
//     for (int i =0; i< this->m_pos.size(); ++i)
//       for(uint k(0); k<3; ++k)
//  	this->m_forces[i][k] = 0.0;
//     const real_t two_third = 2.0/3.0;
//     vector<CellGeometry<real_t> > & geoms(this->m_complex.getGeoms());
//     const vector<Cell<real_t> > & cells(this->m_complex.getCells());
// #pragma omp parallel for
//     for(size_t i=0; i< geoms.size(); ++i){
//       //geoms[i].computeAll();
//       geoms[i].diffVolume();
//       real_t press = (this->m_pressEq*this->m_volAvg)/geoms[i].getVolume();
//       const vector< Array<real_t, 3> > & dV(geoms[i].getdV());
//       for(uint0 j=0; j< cells[i].numFacets(); ++j){
// 	uint2 nbr = cells[i].getNbr(j);
// 	for(uint0 k=0; k<3; ++k){
// 	  real_t df = -press*dV[j][k];
// #pragma omp atomic
// 	  this->m_forces[nbr][k] -= df;
// #pragma omp atomic
// 	  this->m_forces[i][k] += df;
// 	}
// 	if (nbr > i){
// 	  real_t visc, bulkVisc;
// 	  visc = (this->m_masses[i]+this->m_masses[nbr])/(this->m_masses[i]/m_visc[i]+this->m_masses[nbr]/m_visc[nbr]);
// 	  bulkVisc = (this->m_masses[i]+this->m_masses[nbr])/(this->m_masses[i]/m_bulkVisc[i]+this->m_masses[nbr]/m_bulkVisc[nbr]);
// 	  const Array<real_t, 3> & r(geoms[i].getConnVect()[j]);
// 	  real_t rSq = geoms[i].getConnVectSq()[j];
// 	  const Array<real_t, 3> & area(geoms[i].getAreas()[j]);
// 	  real_t inp1=0, inp2=0;
// 	  Array<real_t, 3> dv;
// 	  for(uint0 k=0; k<3; ++k){
// 	    dv[k] = this->m_vel[nbr][k] - this->m_vel[i][k];
// 	    inp1 += r[k]*dv[k];
// 	    inp2 += r[k]*area[k];
// 	  }
// 	  real_t c1 = visc*inp2/(3.0*rSq);
// 	  real_t c2 = (bulkVisc+visc/3.0)*inp1/(3.0*rSq);
// 	  for(uint0 k=0; k<3; ++k){
// 	    real_t df;
// 	    df = c1*dv[k] + c2*area[k];
// #pragma omp atomic
// 	    this->m_forces[nbr][k] -= df;
// #pragma omp atomic
// 	    this->m_forces[i][k] += df;
// 	  }
// 	}
//       }
//     }
//   }
  
  template<typename real_t>
  void NavierStokes<real_t>::computeForces()
  {
#pragma omp parallel for
    for (int i =0; i< this->m_pos.size(); ++i)
      for(uint k(0); k<3; ++k)
 	this->m_forces[i][k] = 0.0;
    const real_t two_third = 2.0/3.0;
    vector<CellGeometry<real_t> > & geoms(this->m_complex.getGeoms());
#pragma omp parallel for
    for(size_t i=0; i< geoms.size(); ++i){
      geoms[i].diffVolume();
      const Cell<real_t> & cell(this->m_complex.getCells()[i]);
      Array<Array<real_t, 3>, 3> gradVel(geoms[i].velocityGradient(this->m_vel));
      real_t press = (this->m_pressEq*this->m_volAvg)/geoms[i].getVolume();
      real_t divVel=0;
      Array<Array<real_t, 3>, 3> stress;
      // compute viscous stress in cell
      for(uint0 k=0; k<3; ++k){
	for(uint0 m=0; m<3; ++m){
	  stress[k][m] = m_visc[i]*(gradVel[k][m]+gradVel[m][k]);
	}
	divVel += gradVel[k][k];
      }
      for(uint0 k=0; k<3; ++k)
	stress[k][k] += (m_bulkVisc[i]-two_third*m_visc[i])*divVel;
      const vector<Array<real_t, 3> > & areas(geoms[i].getAreas());
      for(uint0 j=0; j< cell.numFacets(); ++j){
	uint2 nbr = cell.getNbr(j);
	for(uint0 k=0; k<3; ++k){
	  real_t df = 0;
	  for(int m(0); m<3; ++m)
	    df -= areas[j][m]*stress[m][k];
	  df += geoms[i].getdV()[j][k]*press;
	  df *= 0.5;
#pragma omp atomic
	  this->m_forces[nbr][k] += df;
#pragma omp atomic
	  this->m_forces[i][k] -= df;
	}
      }
    }
  }

  
  template<typename real_t>
  bool NavierStokes<real_t>::init()
  {
    const Array<real_t, 3> & L(this->m_box.getL());
    this->m_volAvg = L[0]*L[1]*L[2]/(this->m_pos.size());    
    if (!Simulation<real_t>::init())
      return false;
    if (this->m_pressEq==0){
      fprintf(stderr,"equilibrium pressure is not set\n");
      return false;
    }
    size_t numPart = this->m_pos.size();
    this->m_forces.resize(numPart);
    if (m_visc.size() ==0){
      fprintf(stderr,"viscosities not set, assuming zero\n");
      m_visc.resize(numPart,0);
    } else if (m_visc.size() ==1) {
      m_visc.resize(numPart);
      for(int i=1; i<m_visc.size(); ++i)
	m_visc[i] = m_visc[0];
    } else if (m_visc.size() != numPart) {
      fprintf(stderr,"Vector of viscosities incorrect size. Setting all viscosities to zero.\n");
      m_visc.clear();
      m_visc.resize(numPart,0);
    }
    if (m_bulkVisc.size() ==0){
      fprintf(stderr,"bulk viscosities not set, assuming zero\n");
      m_bulkVisc.resize(numPart,0);
    } else if (m_bulkVisc.size() ==1) {
      m_bulkVisc.resize(numPart);
      for(int i=1; i<m_bulkVisc.size(); ++i)
	m_bulkVisc[i] = m_bulkVisc[0];
    } else if (m_bulkVisc.size() != numPart) {
      fprintf(stderr,"Vector of bulk-viscosities incorrect size. Setting all viscosities to zero.\n");
      m_bulkVisc.clear();
      m_bulkVisc.resize(numPart,0);
    }    
    this->computeForces();
    return true;
  }

  template<typename real_t>
  void NavierStokes<real_t>::setViscosity(real_t visc)
  {
    m_visc.resize(1);
    m_visc[0] = visc;
  }

  template<typename real_t>
  void NavierStokes<real_t>::setBulkViscosity(real_t visc)
  {
    m_bulkVisc.resize(1);
    m_bulkVisc[0] = visc;
  }

  template<typename real_t>
  void IntfDyn<real_t>::setIntfTension(real_t intfTension, uint0 i, uint0 j)
  {
    if (i != j){
      uint2 m = (i<j ? (((uint2) j)*((uint2) j -1))/2+((uint2) i):(((uint2) i)*((uint2) i -1))/2+((uint2) j));
      if (m >= m_intfTension.size())
	m_intfTension.resize(m+1);
      m_intfTension[m] = intfTension;
    }
  }    

  template<typename real_t>
  inline real_t IntfDyn<real_t>::getIntfTension(uint0 i, uint0 j) const
  {
    if (i==j) return 0;
    uint2 m = (i<j ? (((uint2) j)*((uint2) j -1))/2+((uint2) i):(((uint2) i)*((uint2) i -1))/2+((uint2) j));
    return m_intfTension[m];
  }
  

  template<typename real_t>
  void IntfDyn<real_t>::computeForces()
  {
    this->NavierStokes<real_t>::computeForces();
    computeIntfForces();
  }

  template<typename real_t>
  void IntfDyn<real_t>::computeIntfForces()
  {
    const vector<uint0> & types(this->m_complex.getTypes());
    vector<uint2> indxFacets;
    vector<Array<real_t, 3> > grad;
#pragma omp parallel for
    for(size_t i=0; i< types.size(); ++i){
      if(types[i] > 0){
	const Cell<real_t> & cell(this->m_complex.getCells()[i]);
	const CellGeometry<real_t> & geom(this->m_complex.getGeoms()[i]);
	for(uint1 j=0; j< cell.numFacets(); ++j){
	  uint2 nbr = cell.getNbr(j);
	  if(types[nbr] < types[i]){
	    const Array<real_t, 3> & areaV(geom.getAreas()[j]);
	    real_t area= sqrt(areaV[0]*areaV[0]+areaV[1]*areaV[1]+areaV[2]*areaV[2]);
	    geom.gradFacetAreaSq(j, indxFacets, grad);
	    real_t intfTension;
	    intfTension = getIntfTension(types[i], types[nbr]);
	    for(int m=0; m< indxFacets.size(); ++m){
	      uint2 nbr2 = cell.getNbr(indxFacets[m]);
	      for(uint0 k=0; k<3; ++k){
		real_t df = -intfTension*grad[m][k]/(2.0*area);
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
    const vector<uint0> & types(this->m_complex.getTypes());
    vector<uint2> indxFacets;
    vector<Array<real_t, 3> > grad;
#pragma omp parallel for reduction(+:E)
    for(size_t i=0; i< types.size(); ++i){
      if(types[i] >0){
	const Cell<real_t> & cell(this->m_complex.getCells()[i]);
	const CellGeometry<real_t> & geom(this->m_complex.getGeoms()[i]);
	for(uint1 j=0; j< cell.numFacets(); ++j){
	  uint2 nbr = cell.getNbr(j);
	  if(types[nbr] < types[i]){
	    //printf("%d: type %d, %d: type %d\n", i, types[i], nbr,types[nbr]);
	    const Array<real_t, 3> & areaV(geom.getAreas()[j]);
	    real_t area= sqrt(areaV[0]*areaV[0]+areaV[1]*areaV[1]+areaV[2]*areaV[2]);
	    E += getIntfTension(types[i], types[nbr])*area;
	  }
	}
      }
    }
    return E;
  }

  // template<typename real_t>
  // void ViscousStress<real_t>::update(Array<Array<real_t, 3>, 3> & stress, const Array<Array<real_t, 3>, 3> & gradV)
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
  
}
#endif
