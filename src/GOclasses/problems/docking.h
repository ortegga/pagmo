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

namespace pagmo { namespace problem {	
	
class DynamicSystem;	


// TODO:
//	- try to make the loggin better!!
//	- think about the const in some functions!!

/// Spacecraft docking problem.
/**
 * "Docking problem, using ANN to develop a robust controller
 * TODO add description
 *
 * @author Juxi Leitner <juxi.leitner@gmail.com>
 */
class __PAGMO_VISIBLE docking : public base {
	public:
		/// Constructor setting some of the variables and conditions
		/**
		 * Initialise the values and conditions for the docking problem. The chromosome from PaGMO will be used as
		 * weights in an artificial neural network (ANN), with the inputs coming from the state of the spacecraft (ideally 
		 * position, speed, and orientation). The output neurons of the ANN are controlling the thrusters of the s/c.
		 *
		 * Based on the HCW equations the spacecraft trajectory is integrated and a fitness is returned depending on 
		 * the achieved (docking) capabilities. 
		 *
		 * @param[in] ann_ a pointer to a neural network.
		 * @param[in] random_positions number of positions each individual is evaluated against.
		 * @param[in] in_pre_evo_strat the strategy to generate the starting positions.
		 * @param[in] max_time the maximum time to test the individual for in each position (needed in the integrator).
		 * @param[in] max_thr the maximum thrust that each thruster on the spacecraft can achieve (needed in the integrator).
		 */		
		docking(ann_toolbox::neural_network *ann_, 
				size_t random_positions,
				size_t in_pre_evo_strat = 1,
				double max_time = 20,
				double max_thr = 0.1 );
		
		// Object functions needed
		virtual docking 	*clone() const { return new docking(*this); };
		virtual std::string	id_object() const {
			return "Docking problem, using ANN to develop a robust controller"; }
		
		/// Setter methods to set the current starting positions for the evaluation
		void set_start_condition(double* , size_t );
		void set_start_condition(std::vector<double> &);
		
		/// Setter for the starting condition, chosing from a predefined one
		void set_start_condition(size_t );
		
		/// Setter methods for control variables
		void set_fitness_function(int );
		void set_log_genome(bool );
		void set_timeneuron_threshold(double );
		void set_time_step(double );		
		void set_vicinity_distance(double );
		void set_vicinity_speed(double );
		void set_vicinity_orientation(double );
						
		/// The ODE system we want to integrate needs to be able to be called 
		/// by the integrator. Here we use the Hill's equations.
		void operator()( state_type &x , state_type &dxdt , double t ) const;
		
		
		/// The objective funtion to be optimized. It returns a fitness based on the 
		/// input (vector).
		virtual double	objfun_(const std::vector<double> &) const;

		/// Generate starting positions for the run of the individuals.
		void generate_starting_positions() const;
				
		/// CONSTANTS for 
		const static size_t FIXED_POS = 1;
		const static size_t SPOKE_POS = 2;
		const static size_t SPOKE_POS_HALF 	= 20;
		const static size_t SPOKE_8_POS 	= 200;		
		const static size_t RAND_POS  = 3;
		const static size_t DONUT_FACING = 33;
		const static size_t CLOUD_POS = 4;	
		const static size_t FULL_GRID = 99;		
		
	private:
		/// Before every evolution this function is called to reset the (randomly)
		/// created starting points for the individuals.
		virtual void	pre_evolution(population &po) const;
//		virtual void	post_evolution(population &pop) const;

		/// Generator functions for various types of "randomized" starting positions
		void generate_spoke_positions(double, double, int half = 0) const;
		void generate_multi_spoke_positions(double, double, int ) const;
		void generate_random_positions(double, double) const;
		void generate_random_positions_facing_origin(double, double) const;
		void generate_cloud_positions(double, double, double ) const;
		void generate_full_grid_positions(int, int) const;
				
		/// Calculates the fitness for one genome for one specific starting position
		// TODO maybe put the starting position here as a parameter?
		double 	one_run(std::string &) const;

		/// The function that evaluates the run and returns the fitness of the chromsome. 
		/// This function is calling various handler functions depending on the fitness_function
		/// set by the control variable of that name.
		std::vector<double> evaluate_fitness(std::vector<double> , std::vector<double> , double, double) const;
		
		/// Scale the outputs to be between -max_thrust and +max_thrust
		std::vector<double> scale_outputs(std::vector<double> ) const;

		////////////////////////////////////
		// Variables of the object
		mutable std::vector<double>	starting_condition;
		mutable std::vector< std::vector<double> > random_start;

		// Reference to the neural network representation
		mutable ann_toolbox::neural_network *ann;
		
		// Variables/Constants for the ODE
		double nu, max_thrust, mR, max_docking_time;
		double time_neuron_threshold;
		
		// control variables
		bool log_genome;					// is the genome logged in the log string 
		size_t needed_count_at_goal;		// how long does the s/c need to stay within the target area before the optimization stops
		size_t random_starting_positions;	// how many random starting positions exist/need to be generated
		size_t pre_evolution_strategy;		// which strategy for the generation of the random numbers is used
		size_t fitness_function;			// how to calculate the fitness
	
		double vicinity_distance;			// the size of the vicinity around the origin that we take as close enough
		double vicinity_speed;				// the maximum speed around the origin that we take as small enough
		double vicinity_orientation;		// the needed orientation around the origin that we take as good enough
		
		double time_step; 					// for integrator		
		
		// Integrator / solver;
		friend class DynamicSystem;
};

class DynamicSystem {
	private: 
		const docking *prob;
		std::vector<double> outputs;		
	public:
		DynamicSystem(const docking *in) : prob(in) {	}
		void operator()( std::vector<double> &x , std::vector<double> &dxdt , double t );
		void set_outputs(std::vector<double> );		
};
	
}
}
#endif
