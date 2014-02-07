/*****************************************************************************
 *   Copyright (C) 2004-2013 The PaGMO development team,                     *
 *   Advanced Concepts Team (ACT), European Space Agency (ESA)               *
 *   http://apps.sourceforge.net/mediawiki/pagmo                             *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Developers  *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Credits     *
 *   act@esa.int                                                             *
 *                                                                           *
 *   This program is free software; you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation; either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program; if not, write to the                           *
 *   Free Software Foundation, Inc.,                                         *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.               *
 *****************************************************************************/

#include <string>
#include <boost/math/constants/constants.hpp>
#include <vector>
#include <numeric>
#include <cmath>

#include "mga_incipit_cstrs.h"
#include "../keplerian_toolbox/keplerian_toolbox.h"

namespace pagmo { namespace problem {

 
/// Constructor
/**
 * Constructs a global optimization problem (box-bounded, continuous) representing an interplanetary trajectory modelled
 * as a Multiple Gravity Assist trajectory that allows one only Deep Space Manouvre between each leg.
 *  
 * @param[in] seq std::vector of kep_toolbox::planet_ptr containing the encounter sequence for the trajectoty (including the initial planet)
 * @param[in] t0_l kep_toolbox::epoch representing the lower bound for the launch epoch
 * @param[in] t0_u kep_toolbox::epoch representing the upper bound for the launch epoch
 * @param[in] tof time-of-flight vector containing lower and upper bounds (in days) for the various legs time of flights
 * @param[in] tmax maximum time of flight
 * @param[in] dmin minimum distance from Jupiter (for radiation protection)
 * @param[in] thrust technological limitation on the thrust
 * @param[in] a_final orbit insertion semimajor axis
 * @param[in] e_final orbit insertion eccentricity
 * @param[in] i_final orbit insertion inclination
 *
 * @throws value_error if the planets in seq do not all have the same central body gravitational constant
 * @throws value_error if tof has a size different from seq.size()
 */

mga_incipit_cstrs::mga_incipit_cstrs(
			 const std::vector<kep_toolbox::planet_ptr> seq,
			 const kep_toolbox::epoch t0_l,
			 const kep_toolbox::epoch t0_u,
			 const std::vector<std::vector<double> > tof,
			 const double tmax,
			 const std::vector<double> dmin,
			 const double thrust,
			 const double a_final,
			 const double e_final,
			 const double i_final
				    ) : base(4*seq.size()+2,0,1,compute_number_of_c(tmax,dmin,thrust,a_final,e_final,i_final),compute_number_of_ic(tmax,dmin,thrust),1E-3), m_tof(tof), m_tmax(tmax), m_dmin(dmin), m_thrust(thrust), m_a_final(a_final), m_e_final(e_final), m_i_final(i_final)
{
	// We check that all planets have equal central body
	std::vector<double> mus(seq.size());
	for (std::vector<kep_toolbox::planet_ptr>::size_type i = 0; i< seq.size(); ++i) {
		mus[i] = seq[i]->get_mu_central_body();
	}
	if ((unsigned int)std::count(mus.begin(), mus.end(), mus[0]) != mus.size()) {
		pagmo_throw(value_error,"The planets do not all have the same mu_central_body");  
	}
	// We check the consistency of the time of flights
	if (tof.size() != seq.size()) {
		pagmo_throw(value_error,"The time-of-flight vector (tof) has the wrong length");  
	}
	for (size_t i = 0; i < tof.size(); ++i) {
		if (tof[i].size()!=2) pagmo_throw(value_error,"Each element of the time-of-flight vector (tof)  needs to have dimension 2 (lower and upper bound)"); 
	}
	
	// We check the consistency of the constraints input
	if (tmax<0){
		pagmo_throw(value_error,"The maximum time of flight must be a positive value");  
	}
	if (dmin.size() != seq.size()-1) {
		pagmo_throw(value_error,"The vector of allowed minimum distance to the center of the system has the wrong length");  
	}
	for (size_t i = 0; i < seq.size()-1; ++i) {
		if (dmin[i]<0) pagmo_throw(value_error,"The minimum distance to the center of the system must be positive"); 
	}
	if (thrust<0){
		pagmo_throw(value_error,"The technological constraint on the thrust need to be positive");  
	}
	if (a_final<=0 && a_final!= -1.0){
		pagmo_throw(value_error,"The final semimajor axis must be positive");  
	}
	if ((e_final<0 || e_final >1) && (e_final!= -1.0)){
		pagmo_throw(value_error,"The final eccentricity must be between 0 and 1");  
	}
	if ((i_final<0 || i_final > 2*boost::math::constants::pi<double>()) && (i_final!= -1.0)){
		pagmo_throw(value_error,"The final inclination must be between 0 and 2*PI");  
	}
	
	
	// Filling in the planetary sequence data member. This requires to construct the polymorphic planets via their clone method 
	for (std::vector<kep_toolbox::planet>::size_type i = 0; i < seq.size(); ++i) {
		m_seq.push_back(seq[i]->clone());
	}
	
	// Now setting the problem bounds
	size_type dim(4*m_tof.size()+2);
	decision_vector lb(dim), ub(dim);
	
	// First leg
	lb[0] = t0_l.mjd2000(); ub[0] = t0_u.mjd2000();
	lb[1] = 0; lb[2] = 0; ub[1] = 1; ub[2] = 1;
	lb[3] = m_tof[0][0]; ub[3] = m_tof[0][1];
	
	// Successive legs
	for (std::vector<kep_toolbox::planet>::size_type i = 1; i < m_tof.size(); ++i) {
		lb[4*i] = - 2 * boost::math::constants::pi<double>();    ub[4*i] = 2 * boost::math::constants::pi<double>();
		lb[4*i+1] = 1.1;  ub[4*i+1] = 30;
		lb[4*i+2] = 1e-5; ub[4*i+2] = 1-1e-5;
		lb[4*i+3] = m_tof[i][0]; ub[4*i+3] = m_tof[i][1];
	}
	
	// Adjusting the minimum and maximum allowed fly-by rp to the one defined in the kep_toolbox::planet class
	for (std::vector<kep_toolbox::planet>::size_type i = 0; i < m_tof.size()-1; ++i) {
		lb[4*i+5] = m_seq[i]->get_safe_radius() / m_seq[i]->get_radius();
		ub[4*i+5] = (m_seq[i]->get_radius() + 2000000) / m_seq[i]->get_radius(); //from gtoc6 problem description
	}
	
	//adding variables for the last flyby and insertion into orbit
	lb[4*m_tof.size()] = - 2 * boost::math::constants::pi<double>(); ub[4*m_tof.size()] = 2 * boost::math::constants::pi<double>();
	lb[4*m_tof.size()+1] = 1.1; ub[4*m_tof.size()+1] = 30;
	
	set_bounds(lb,ub);
}

/// Copy Constructor. Performs a deep copy
mga_incipit_cstrs::mga_incipit_cstrs(const mga_incipit_cstrs &p) :
	 base(p.get_dimension(),p.get_i_dimension(),p.get_f_dimension(),p.get_c_dimension(),p.get_ic_dimension(),p.get_c_tol()),
	 m_tof(p.m_tof),
	 m_tmax(p.m_tmax),
	 m_dmin(p.m_dmin),
	 m_thrust(p.m_thrust),
	 m_a_final(p.m_a_final),
	 m_e_final(p.m_e_final),
	 m_i_final(p.m_i_final)	 
{
	for (std::vector<kep_toolbox::planet_ptr>::size_type i = 0; i < p.m_seq.size();++i) {
		m_seq.push_back(p.m_seq[i]->clone());
	}
	set_bounds(p.get_lb(),p.get_ub());
}

/// Clone method.
base_ptr mga_incipit_cstrs::clone() const
{
	return base_ptr(new mga_incipit_cstrs(*this));
}


pagmo::problem::base::c_size_type mga_incipit_cstrs::compute_number_of_c(const double tmax, const std::vector<double> dmin, const double thrust, const double a_final, const double e_final, const double i_final) const{
	pagmo::problem::base::c_size_type n_count = 0;
  
	if(tmax > 0.0) n_count++;
	for (size_t i = 0; i < m_seq.size()-1; ++i) {
	    if (dmin[i] > 0.0) n_count++; 
	}
	if(thrust > 0.0) n_count = n_count+m_seq.size();
	
	if(a_final != -1.0) n_count++;
	if(e_final != -1.0) n_count++;
	if(i_final != -1.0) n_count++;
	
  	return n_count;
}

pagmo::problem::base::c_size_type mga_incipit_cstrs::compute_number_of_ic(const double tmax, const std::vector<double> dmin, const double thrust) const{
  	pagmo::problem::base::c_size_type n_count = 0;
  
	if(tmax > 0.0) n_count++;
	for (size_t i = 0; i < m_seq.size()-1; ++i) {
	    if (dmin[i] > 0.0) n_count++; 
	}
	if(thrust > 0.0) n_count = n_count+m_seq.size();
  	
  	return n_count;
}

/// Implementation of the objective function.
void mga_incipit_cstrs::objfun_impl(fitness_vector &f, const decision_vector &x) const
{
try {
	double common_mu = m_seq[0]->get_mu_central_body();
	// 1 -  we 'decode' the chromosome recording the various times of flight (days) in the list T
	std::vector<double> T(m_seq.size(),0.0);
	
	for (size_t i = 0; i<m_seq.size(); ++i) {
		T[i] = x[4*i+3];
	}
	// 2 - We compute the epochs and ephemerides of the planetary encounters
	std::vector<kep_toolbox::epoch>   t_P(m_seq.size());
	std::vector<kep_toolbox::array3D> r_P(m_seq.size());
	std::vector<kep_toolbox::array3D> v_P(m_seq.size());
	std::vector<double> DV(m_seq.size());
	for (size_t i = 0; i<r_P.size(); ++i) {
		t_P[i] = kep_toolbox::epoch(x[0] + std::accumulate(T.begin(),T.begin()+1+i,0.0));
		m_seq[i]->get_eph(t_P[i], r_P[i], v_P[i]);
	}

	// 3 - We start with the first leg
	double theta = 2*boost::math::constants::pi<double>()*x[1];
	double phi = acos(2*x[2]-1)-boost::math::constants::pi<double>() / 2;
	double d,d2,ra,ra2;
	kep_toolbox::array3D r = { {ASTRO_JR*1000*cos(phi)*sin(theta), ASTRO_JR*1000*cos(phi)*cos(theta), ASTRO_JR*1000*sin(phi)} };
	kep_toolbox::array3D v;
	kep_toolbox::lambert_problem l(r,r_P[0],T[0]*ASTRO_DAY2SEC,common_mu,false,false);
	kep_toolbox::array3D v_beg_l = l.get_v1()[0];
	kep_toolbox::array3D v_end_l = l.get_v2()[0];

	DV[0] = std::abs(kep_toolbox::norm(v_beg_l)-3400.0);
	
	// 4 - And we proceed with each successive leg (if any)
	kep_toolbox::array3D v_out;
	for (size_t i = 1; i<m_seq.size(); ++i) {
		// Fly-by
		kep_toolbox::fb_prop(v_out, v_end_l, v_P[i-1], x[4*i+1] * m_seq[i-1]->get_radius(), x[4*i], m_seq[i-1]->get_mu_self());
		r = r_P[i-1];
		v = v_out;
		// s/c propagation before the DSM
		kep_toolbox::propagate_lagrangian(r,v,x[4*i+2]*T[i]*ASTRO_DAY2SEC,common_mu);
		kep_toolbox::closest_distance(d, ra, r_P[i-1], v_out, r, v, common_mu);

		// Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
		double dt = (1-x[4*i+2])*T[i]*ASTRO_DAY2SEC;
		kep_toolbox::lambert_problem l2(r,r_P[i],dt,common_mu,false,false);
		v_end_l = l2.get_v2()[0];
		v_beg_l = l2.get_v1()[0];
		kep_toolbox::closest_distance(d2,ra2,r,v_beg_l, r_P[i], v_end_l, common_mu);
		if (d < d2)
		{
			d = d/ASTRO_JR;
		} else {
			d = d2/ASTRO_JR;
		}

		// DSM occuring at time nu2*T2
		kep_toolbox::diff(v_out, v_beg_l, v);
		DV[i] = kep_toolbox::norm(v_out);// + std::max((2.0-d),0.0) * 1000.0;
	}
	// Now we return the objective(s) function
	f[0] = std::accumulate(DV.begin(),DV.end(),0.0); 
//Here the lambert solver or the lagrangian propagator went wrong
} catch (...) {
	f[0] = boost::numeric::bounds<double>::highest();
} 
}

/// Implementation of the objective function.
void mga_incipit_cstrs::compute_constraints_impl(constraint_vector &c, const decision_vector &x) const
{
try {
	double common_mu = m_seq[0]->get_mu_central_body();
	// 1 -  we 'decode' the chromosome recording the various times of flight (days) in the list T
	std::vector<double> T(m_seq.size(),0.0);
	std::vector<double> d(m_seq.size()-1,0.0);

	for (size_t i = 0; i<m_seq.size(); ++i) {
		T[i] = x[4*i+3];
	}
	// 2 - We compute the epochs and ephemerides of the planetary encounters
	std::vector<kep_toolbox::epoch>   t_P(m_seq.size());
	std::vector<kep_toolbox::array3D> r_P(m_seq.size());
	std::vector<kep_toolbox::array3D> v_P(m_seq.size());
	std::vector<double> DV(m_seq.size());
	for (size_t i = 0; i<r_P.size(); ++i) {
		t_P[i] = kep_toolbox::epoch(x[0] + std::accumulate(T.begin(),T.begin()+1+i,0.0));
		m_seq[i]->get_eph(t_P[i], r_P[i], v_P[i]);
	}

	// 3 - We start with the first leg
	double theta = 2*boost::math::constants::pi<double>()*x[1];
	double phi = acos(2*x[2]-1)-boost::math::constants::pi<double>() / 2;
	double d2,ra,ra2;
	kep_toolbox::array6D E;
	kep_toolbox::array3D r = { {ASTRO_JR*1000*cos(phi)*sin(theta), ASTRO_JR*1000*cos(phi)*cos(theta), ASTRO_JR*1000*sin(phi)} };
	kep_toolbox::array3D v;
	kep_toolbox::lambert_problem l(r,r_P[0],T[0]*ASTRO_DAY2SEC,common_mu,false,false);
	kep_toolbox::array3D v_beg_l = l.get_v1()[0];
	kep_toolbox::array3D v_end_l = l.get_v2()[0];

	DV[0] = std::abs(kep_toolbox::norm(v_beg_l)-3400.0);

	// 4 - And we proceed with each successive leg (if any)
	kep_toolbox::array3D v_out;
	for (size_t i = 1; i<m_seq.size(); ++i) {
		// Fly-by
		kep_toolbox::fb_prop(v_out, v_end_l, v_P[i-1], x[4*i+1] * m_seq[i-1]->get_radius(), x[4*i], m_seq[i-1]->get_mu_self());
		r = r_P[i-1];
		v = v_out;
		// s/c propagation before the DSM
		kep_toolbox::propagate_lagrangian(r,v,x[4*i+2]*T[i]*ASTRO_DAY2SEC,common_mu);
		kep_toolbox::closest_distance(d[i-1], ra, r_P[i-1], v_out, r, v, common_mu);

		// Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
		double dt = (1-x[4*i+2])*T[i]*ASTRO_DAY2SEC;
		kep_toolbox::lambert_problem l2(r,r_P[i],dt,common_mu,false,false);
		v_end_l = l2.get_v2()[0];
		v_beg_l = l2.get_v1()[0];
		kep_toolbox::closest_distance(d2,ra2,r,v_beg_l, r_P[i], v_end_l, common_mu);
		if (d[i-1] < d2)
		{
			d[i-1] = d[i-1]/ASTRO_JR;
		} else {
			d[i-1] = d2/ASTRO_JR;
		}

		// DSM occuring at time nu2*T2
		kep_toolbox::diff(v_out, v_beg_l, v);
		DV[i] = kep_toolbox::norm(v_out);// + std::max((2.0-d),0.0) * 1000.0;
	}
	
	
	// compute final insertion orbit
	kep_toolbox::fb_prop(v_out, v_end_l, v_P[m_seq.size()-1], x[4*m_seq.size()+1] * m_seq[m_seq.size()-1]->get_radius(), x[4*m_seq.size()], m_seq[m_seq.size()-1]->get_mu_self());
	r = r_P[m_seq.size()-1];
	v = v_out;	
	kep_toolbox::ic2par(r, v, common_mu, E);
	
	pagmo::problem::base::c_size_type index_cstrs = 0.0;
	
	// Now we return the constraints
	if(m_tmax > 0.0) c[index_cstrs] = std::accumulate(T.begin(),T.end(),0.0) - m_tmax;
	
	for (size_t i = 0; i<m_seq.size()-1; ++i){
	  if(m_dmin[i] > 0.0){
	    c[index_cstrs] = m_dmin[i-1] - d[i-1];
	    index_cstrs++;
	  }
	}

	if(m_thrust > 0.0){
	  for (size_t i = 0; i<m_seq.size(); ++i) { 
	     c[index_cstrs] =  DV[i]/T[i] - m_thrust;
	     index_cstrs++;
	  }
	}
	
	//semi major axis equality constraint
	if(m_a_final != -1.0){
	  c[index_cstrs] = E[0] - m_a_final;
	  index_cstrs++;
	}
	//eccentricity equality constraint
	if(m_e_final != -1.0){
	  c[index_cstrs] = E[1] - m_e_final;
	  index_cstrs++;
	} 
	//inclination equality constraint
	if(m_i_final != -1.0){
	  c[index_cstrs] = E[2] - m_i_final;
	  index_cstrs++;
	}
	
	//std::cout<<E[0]<<", "<<E[1]<<", "E[2]<<std::endl;
	
//Here the lambert solver or the lagrangian propagator went wrong
} catch (...) {
	for (size_t i = 0; i<get_c_dimension(); ++i) {
	  c[i] = boost::numeric::bounds<double>::highest();
	}
}
}

/// Outputs a stream with the trajectory data
/**
 * While the chromosome contains all necessary information to describe a trajectory, mission analysits
 * often require a different set of data to evaluate its use. This method outputs a stream with
 * information on the trajectory that is otherwise 'hidden' in the chromosome
 *
 * \param[in] x chromosome representing the trajectory in the optimization process
 * \returns an std::string with launch dates, DV magnitues and other information on the trajectory
 */

std::string mga_incipit_cstrs::pretty(const std::vector<double> &x) const {
  
	// We set the std output format
	std::ostringstream s;
	s.precision(15);
	s << std::scientific;
	
	double d,ra,d2,ra2;

	double common_mu = m_seq[0]->get_mu_central_body();
	// 1 -  we 'decode' the chromosome recording the various times of flight (days) in the list T
	std::vector<double> T(m_seq.size(),0.0);
	
	for (size_t i = 0; i<m_seq.size(); ++i) {
		T[i] = x[4*i+3];
	}
	// 2 - We compute the epochs and ephemerides of the planetary encounters
	std::vector<kep_toolbox::epoch>   t_P(m_seq.size());
	std::vector<kep_toolbox::array3D> r_P(m_seq.size());
	std::vector<kep_toolbox::array3D> v_P(m_seq.size());
	std::vector<double> DV(m_seq.size());
	for (size_t i = 0; i<r_P.size(); ++i) {
		t_P[i] = kep_toolbox::epoch(x[0] + std::accumulate(T.begin(),T.begin()+1+i,0.0));
		m_seq[i]->get_eph(t_P[i], r_P[i], v_P[i]);
	}

	// 3 - We start with the first leg
	double theta = 2*boost::math::constants::pi<double>()*x[1];
	double phi = acos(2*x[2]-1)-boost::math::constants::pi<double>() / 2;
	kep_toolbox::array3D r = { {ASTRO_JR * 1000*cos(phi)*sin(theta), ASTRO_JR * 1000*cos(phi)*cos(theta), ASTRO_JR * 1000*sin(phi)} };
	kep_toolbox::array3D v;
	
	kep_toolbox::lambert_problem l(r,r_P[0],T[0]*ASTRO_DAY2SEC,common_mu,false,false);
	kep_toolbox::array3D v_beg_l = l.get_v1()[0];
	kep_toolbox::array3D v_end_l = l.get_v2()[0];
	kep_toolbox::closest_distance(d,ra,r,v_beg_l, r_P[0], v_end_l, common_mu);

	DV[0] = std::abs(kep_toolbox::norm(v_beg_l)-3400.0);
	kep_toolbox::array3D v_out,mem_vin,mem_vout,mem_vP;
	
	s << "\nFirst Leg: 1000JR to " << m_seq[0]->get_name() << std::endl; 
	s << "\tDeparture: " << kep_toolbox::epoch(x[0]) << " (" << x[0] << " mjd2000) " << std::endl; 
	s << "\tDuration: " << T[0] << "days" << std::endl; 
	s << "\tInitial Velocity Increment (m/s): " << DV[0] << std::endl; 
	kep_toolbox::diff(v_out, v_end_l, v_P[0]);
	s << "\tArrival relative velocity at " << m_seq[0]->get_name() << " (m/s): " << kep_toolbox::norm(v_out)  << std::endl; 
	s << "\tClosest distance: " << d / ASTRO_JR;
	
	// 4 - And we proceed with each successive leg (if any)
	for (size_t i = 1; i<m_seq.size(); ++i) {
		// Fly-by
		kep_toolbox::fb_prop(v_out, v_end_l, v_P[i-1], x[4*i+1] * m_seq[i-1]->get_radius(), x[4*i], m_seq[i-1]->get_mu_self());
		// s/c propagation before the DSM
		r = r_P[i-1];
		v = v_out;
		mem_vout = v_out;
		mem_vin = v_end_l;
		mem_vP = v_P[i-1];
		
		kep_toolbox::propagate_lagrangian(r,v,x[4*i+2]*T[i]*ASTRO_DAY2SEC,common_mu);
		kep_toolbox::closest_distance(d, ra, r_P[i-1], v_out, r, v, common_mu);
		
		// Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
		double dt = (1-x[4*i+2])*T[i]*ASTRO_DAY2SEC;
		kep_toolbox::lambert_problem l2(r,r_P[i],dt,common_mu,false,false);
		v_end_l = l2.get_v2()[0];
		v_beg_l = l2.get_v1()[0];
		kep_toolbox::closest_distance(d2,ra2,r,v_beg_l, r_P[i], v_end_l, common_mu);
		
		if (d < d2)
		{
			d = d/ASTRO_JR;
			ra = ra/ASTRO_JR;
		} else {
			d = d2/ASTRO_JR;
			ra = ra2/ASTRO_JR;
		}
		// DSM occuring at time nu2*T2
		kep_toolbox::diff(v_out, v_beg_l, v);
		DV[i] = kep_toolbox::norm(v_out);
		s <<  "\nleg no. " << i+1 << ": " << m_seq[i-1]->get_name() << " to " << m_seq[i]->get_name() << std::endl; 
		s <<  "\tDuration: (days)" << T[i] << std::endl; 
		s <<  "\tFly-by epoch: " << t_P[i-1] << " (" << t_P[i-1].mjd2000() << " mjd2000) " << std::endl; 
		s <<  "\tFly-by altitude (km): " << (x[4*i+1]*m_seq[i-1]->get_radius()-m_seq[i-1]->get_radius())/1000.0 << std::endl; 
		s <<  "\tPlanet position (m): " << r_P[i-1] << std::endl; 
		s <<  "\tPlanet velocity (m/s): " << mem_vP << std::endl; 
		s <<  "\tV in (m/s): " << mem_vin << std::endl; 
		s <<  "\tV out (m/s): " << mem_vout << std::endl << std::endl;

		s <<  "\tDSM after (days): "  << x[4*i+2]*T[i] << std::endl; 
		s <<  "\tDSM magnitude (m/s): " << DV[i] << std::endl; 
		s <<  "\tClosest distance (JR): " << d << std::endl; 
		s <<  "\tApoapsis at closest distance (JR): " << ra << std::endl; 
 	}
	
	s << "\nArrival at " << m_seq[m_seq.size()-1]->get_name() << std::endl; 
	kep_toolbox::diff(v_out, v_end_l, v_P[m_seq.size()-1]);
	s <<  "Arrival epoch: "  << t_P[m_seq.size()-1] << " (" << t_P[m_seq.size()-1].mjd2000() << " mjd2000) " << std::endl; 
	s <<  "Arrival Vinf (m/s): " << v_out << " - " << kep_toolbox::norm(v_out) << std::endl; 
	s <<  "Total mission time (days): " << std::accumulate(T.begin(),T.end(),0.0) << std::endl; 
	return s.str();
}
std::string mga_incipit_cstrs::get_name() const
{
	return "MGA-INCIPIT (CAPTURE AT JUPITER) - Constrained version";
}


/// Gets the times of flight
/**
 * @return[out] vector of times of flight 
 */
const std::vector<std::vector<double> >& mga_incipit_cstrs::get_tof() const {
	return m_tof;
}

/// Sets the times of flight
/**
 * This setter changes the problem bounds as to define a minimum and a maximum allowed total time of flight
 *
 * @param[in] tof vector of times of flight 
 */
void mga_incipit_cstrs::set_tof(const std::vector<std::vector<double> >& tof) {
	if (tof.size() != (m_seq.size())) {
		pagmo_throw(value_error,"The time-of-flight vector (tof) has the wrong length");  
	}
	m_tof = tof;
	for (size_t i=0; i< m_seq.size(); ++i) {
		set_bounds(3+4*i,tof[i][0],tof[i][1]);
	}
}

/// Gets the planetary sequence defining the interplanetary mga-1dsm mission
/**
 * @return An std::vector containing the kep_toolbox::planets
 */
std::vector<kep_toolbox::planet_ptr> mga_incipit_cstrs::get_sequence() const {
	return m_seq;
}

/// Extra human readable info for the problem.
/**
 * Will return a formatted string containing the values vector, the weights vectors and the max weight. It is concatenated
 * with the base::problem human_readable
 */
std::string mga_incipit_cstrs::human_readable_extra() const
{
	std::ostringstream oss;
	oss << "\n\tSequence: ";
	for (size_t i = 0; i<m_seq.size() ;++i) {
		oss << m_seq[i]->get_name() << " ";
	}
	oss << "\n\tTime of flights?: ";
	for (size_t i=0; i<m_seq.size(); ++i) {
	  oss << m_tof[i]<<' ';
	}
	oss << "\n\tMaximum time of flight [days]: " << m_tmax;
	oss << "\n\tMinimum distance from Jupiter: " << m_dmin;
	oss << "\n\tTechnological limitation on the impulsive maneuver: " << m_thrust;
	oss << "\n\tOrbit insertion in the Jupiter system, a [m]: " << m_a_final;
	oss << "\n\tOrbit insertion in the Jupiter system, e [-]: " << m_e_final;
	oss << "\n\tOrbit insertion in the Jupiter system, i [rad]: " << m_i_final;
	
	return oss.str();
}

}} //namespaces

BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::problem::mga_incipit_cstrs)
