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

#ifndef PAGMO_PROBLEM_MGA_INCIPIT_CSTRS_H
#define PAGMO_PROBLEM_MGA_INCIPIT_CSTRS_H

#include <string>

#include "../config.h"
#include "../serialization.h"
#include "../types.h"
#include "../types.h"
#include "base.h"
#include "../keplerian_toolbox/planet_js.h"
#include "../keplerian_toolbox/epoch.h"


namespace pagmo{ namespace problem {

/// The beginning of the GTOC6 Jupiter Capture Trajectory
/**
 *
 * A PyGMO global optimization problem (box-bounded, continuous) representing a capture
 * in the Jupiter system. Two constraints are considered: 1. Closest distance, 2. Time of flight
 *
 * Decision vector:
 * [t0,u,v,T0] + [beta1, rp1/rP1, eta1,T1] + .... 
 * 
 * @author Dario Izzo (dario.izzo@esa.int)
 */
class __PAGMO_VISIBLE mga_incipit_cstrs: public base
{
	public:
		mga_incipit_cstrs(const std::vector<kep_toolbox::planet_ptr> = construct_default_sequence(),
			 const kep_toolbox::epoch t0_l = kep_toolbox::epoch(7305.0),
			 const kep_toolbox::epoch t0_u = kep_toolbox::epoch(11323.0),
			 const std::vector<std::vector<double> > tof = construct_default_tofs(),
			 const double tmax = 0.0,
			 const std::vector<double> dmin = std::vector<double>(2,0.0),
			 const double thrust = 0.0,
			 const double a_final = -1.0,
			 const double e_final = -1.0,
			 const double i_final = -1.0);
		mga_incipit_cstrs(const mga_incipit_cstrs&);
		base_ptr clone() const;
		
		std::string get_name() const;
		std::string pretty(const std::vector<double> &x) const;
		void set_tof(const std::vector<std::vector<double> >&);
		const std::vector<std::vector<double> >& get_tof() const;
		std::vector<kep_toolbox::planet_ptr> get_sequence() const;
		
		pagmo::problem::base::c_size_type compute_number_of_c(const double tmax, const std::vector<double> dmin, const double thrust, const double a_final, const double e_final, const double i_final) const;
		pagmo::problem::base::c_size_type compute_number_of_ic(const double tmax, const std::vector<double> dmin, const double thrust) const;
		
	protected:
		void objfun_impl(fitness_vector &, const decision_vector &) const;
		void compute_constraints_impl(constraint_vector &, const decision_vector &) const;
		std::string human_readable_extra() const;
		
	private:
		static const std::vector<kep_toolbox::planet_ptr> construct_default_sequence() {
			std::vector<kep_toolbox::planet_ptr> retval;
			retval.push_back(kep_toolbox::planet_js("io").clone());
			retval.push_back(kep_toolbox::planet_js("io").clone());
			retval.push_back(kep_toolbox::planet_js("europa").clone());
			return retval;
		}
		static const std::vector<std::vector<double> > construct_default_tofs() {
			std::vector<std::vector<double> > retval;
			std::vector<double> dumb(2);
			dumb[0] = 100;dumb[1] = 200;
			retval.push_back(dumb);
			dumb[0] = 3;dumb[1] = 200;
			retval.push_back(dumb);
			dumb[0] = 4;dumb[1] = 100;
			retval.push_back(dumb);
			return retval;
		}
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive &ar, const unsigned int)
		{
			ar & boost::serialization::base_object<base>(*this);
			ar & m_seq;
			ar & m_tof;
			ar & const_cast<double &>(m_tmax);
			ar & const_cast<std::vector<double> &>(m_dmin);
			ar & const_cast<double &>(m_thrust);
			ar & const_cast<double &>(m_a_final);
			ar & const_cast<double &>(m_e_final);
			ar & const_cast<double &>(m_i_final);
		}
		std::vector<kep_toolbox::planet_ptr> m_seq;
		std::vector<std::vector<double> > m_tof;
		
		//constraints parameters
		const double m_tmax; //maximum time of flight
		const std::vector<double> m_dmin; //minimum distance to center of the system at each leg
		const double m_thrust; //technological constraint on DV
		const double m_a_final; //final semi major axis
		const double m_e_final; //final eccentricity
		const double m_i_final; //final inclination
};

}} // namespaces

BOOST_CLASS_EXPORT_KEY(pagmo::problem::mga_incipit_cstrs)
#endif // PAGMO_PROBLEM_MGA_INCIPIT_CSTRS_H
