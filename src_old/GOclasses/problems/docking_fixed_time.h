/*****************************************************************************
 *   Copyright (C) 2004-2009 The PaGMO development team,                     *
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

// Created by Juxi Leitner on 2009-12-11.

#ifndef PAGMO_PROBLEM_DOCKING_H
#define PAGMO_PROBLEM_DOCKING_H

#include <string>
#include <vector>

#include "../../ann_toolbox/neural_network.h"
#include "../../config.h"
#include "base.h"

typedef std::vector<double> state_type;

namespace pagmo {
namespace problem {	
	
class DynamicSystem;	

// Docking problem.
// "Docking problem, using ANN to develop a robust controller"; } TODO add description

class __PAGMO_VISIBLE docking : public base {
	public:
		// Constructors
		docking(ann_toolbox::neural_network *ann_, double max_time = 10, double max_thr = 0.1);
		
		virtual docking 	*clone() const { return new docking(*this); };
		virtual std::string	id_object() const {
			return "Docking problem, using ANN to develop a robust controller"; }
		
		void set_start_condition(double* , size_t );
		void set_start_condition(std::vector<double> &);
		
		// control variable setter
		void set_log_genome(bool );
		void set_take_best(bool );
		
		// The ODE system we want to integrate needs to be able to be called 
		// by the integrator. Here we have the Hill's equations.
		void operator()( state_type &x , state_type &dxdt , double t ) const;
//		replacing the function: static void hill_equations( state_type & , state_type & , double );
		
	private:
		virtual double	objfun_(const std::vector<double> &) const;
		virtual void	pre_evolution(population &po) const { std::cout << "testing <onweroandf PRE!" << std::endl << "test" << std::endl; };
		virtual void	post_evolution(population &pop) const { std::cout << "testing <onweroandf PPOST!" << std::endl << "test" << std::endl; };
		
		void 			scale_outputs(std::vector<double> &) const;

		mutable std::vector<double>	starting_condition;
		// Variables/Constants for the ODE
		double nu, max_thrust, mR, max_docking_time;
		
		// Reference to the neural network representation
		ann_toolbox::neural_network *ann;
		
		// control variables
		bool take_best;
		bool log_genome;
		
		// TODO: Add integrator as class ...
		//integrator		*solver;
		friend class DynamicSystem;
};

class DynamicSystem {
	private: 
		const docking *prob;
		std::vector<double> outputs;
	public:
		DynamicSystem(const docking *in) : prob(in) {	}
		void operator()( std::vector<double> &x , std::vector<double> &dxdt , double t );
		std::vector<double> get_last_outputs();
};
	
}
}
#endif
