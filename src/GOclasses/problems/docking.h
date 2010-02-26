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
//		docking(ann_toolbox::neural_network *ann_, size_t random_positions, double max_time = 20, double max_thr = 0.1);
		docking(ann_toolbox::neural_network *ann_, size_t random_positions, size_t in_pre_evo_strat = 1, double max_time = 20, double max_thr = 0.1);
				
		virtual docking 	*clone() const { return new docking(*this); };
		virtual std::string	id_object() const {
			return "Docking problem, using ANN to develop a robust controller"; }
		
		void set_start_condition(double* , size_t );
		void set_start_condition(std::vector<double> &);
		
		// set starting condition to a predefined one
		void set_start_condition(size_t );
		// control variable setter
		void set_log_genome(bool );
		void set_timeneuron_threshold(double );
		void set_fitness_function(int );
		
		// The ODE system we want to integrate needs to be able to be called 
		// by the integrator. Here we have the Hill's equations.
		void operator()( state_type &x , state_type &dxdt , double t ) const;
//		replacing the function: static void hill_equations( state_type & , state_type & , double );
		
		virtual double	objfun_(const std::vector<double> &) const;
		
		const static size_t FIXED_POS = 1;
		const static size_t SPOKE_POS = 2;
		const static size_t RAND_POS  = 3;

	private:
		virtual void	pre_evolution(population &po) const;// { std::cout << "testing <onweroandf PRE!" << std::endl << "test" << std::endl; };
//		virtual void	post_evolution(population &pop) const { std::cout << "testing <onweroandf PPOST!" << std::endl << "test" << std::endl; };

		// generating various types of "randomized" starting positions
		void generate_spoke_positions(double, double) const;
		void generate_random_positions(double, double) const;			// bad cuz it is not const ;)
		
		// calculate fitnesses (for one start position)
		double 	one_run(std::string &) const;
		// evaluate the actual fitness here
		std::vector<double> evaluate_fitness(std::vector<double> , std::vector<double> , double, double) const;
		
		std::vector<double> scale_outputs(std::vector<double> ) const;

		mutable std::vector<double>	starting_condition;
		mutable std::vector< std::vector<double> > random_start;

		// Variables/Constants for the ODE
		double nu, max_thrust, mR, max_docking_time;
		double time_neuron_threshold;
		
		// Reference to the neural network representation
		mutable ann_toolbox::neural_network *ann;
		
		// control variables
		bool log_genome;
		size_t needed_count_at_goal;		// how long does the s/c need to stay within the target area before the optimization stops
		size_t random_starting_postions;	// how many random starting positions exist/need to be generated
		size_t pre_evolution_strategy;		// which strategy for the generation of the random numbers is used
		size_t fitness_function;			// how to calculate the fitness
		
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
//		std::vector<double> get_last_outputs();
		void set_outputs(std::vector<double> );		
};
	
}
}
#endif
