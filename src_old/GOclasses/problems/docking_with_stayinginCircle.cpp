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

// Created by Juxi Leitner on 2009-12-11

#include <exception>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "base.h"
#include "docking.h"

#include "../../exceptions.h"
#include "../basic/population.h"
#include "../../ann_toolbox/neural_network.h"
#include "../../ann_toolbox/multilayer_perceptron.h"
#include "../../odeint/odeint.hpp"


extern std::string max_log_string;
extern double max_log_fitness;

namespace pagmo {
namespace problem {	
	
// Constructors
docking::docking(ann_toolbox::neural_network* ann_, int needed_cnt_at_g, double max_time, double max_thr) :
	base(ann_->get_number_of_weights()),
	ann(ann_),
	max_docking_time(max_time + .1),	
	max_thrust(max_thr),
	needed_count_at_goal(needed_cnt_at_g)
{						
	// the docking problem needs:
	// 	3 inputs (and starting conditions): x, z, theta
	//  2 outputs (control): ul, ur		(the two thrusters, [0,1] needs to be mapped to [-1,1])
	// and final conditions: x, z,	// later maybe theta, v
		
	// Set the boundaries for the values in the genome, this is important for the ANN
	set_ub(	std::vector<double> (ann->get_number_of_weights(), 10.0) );
	set_lb(	std::vector<double> (ann->get_number_of_weights(), -10.0) );
	
	// take the best result during the whole integration steps not the one at the end!!
	take_best = true;
	
	// disable genome logging
	log_genome = false;
}

void docking::set_start_condition(double *start_cnd, size_t size) {
	starting_condition = std::vector<double> (start_cnd, start_cnd + size);
}

void docking::set_start_condition(std::vector<double> &start_cond) {
	starting_condition = start_cond;
}

void docking::set_log_genome(bool b) {
	log_genome = b;
}
void docking::set_take_best(bool b) {
	take_best = b;
}

void docking::pre_evolution(population &pop) const {
/*	//Re-evaluate the population with respect to the new seed (Internal Sampling Method)
	for (size_t i=0; i < pop.size(); ++i) {
		pop[i] = individual(*this, pop[i].get_decision_vector(), pop[i].get_velocity());
	}*/
}

// Objective function
double docking::objfun_(const std::vector<double> &v) const {
	if(v.size() != ann->get_number_of_weights()) {
		pagmo_throw(value_error, "wrong number of weights in the chromosome");
	}
	
	// Helper variables
	double retval, distance, theta, speed, best_retval = 0.0;
	int counter_at_goal = 0;
	
	// Integrator System
	DynamicSystem sys(this);
	// Create Integration Stepper
	odeint::ode_step_runge_kutta_4< std::vector<double>, double > stepper;

	// initialize the inputs (= starting conditions) to the ANN and allocate the outputs 
	std::vector<double> inputs = starting_condition, out;

	// Initialize ANN and interpret the chromosome
	ann->set_weights(v);
	
	///////////////////////////////////////////////////////
	// LOGGER
	std::stringstream ss (std::stringstream::out);
	ss << *(ann_toolbox::multilayer_perceptron*)ann << std::endl;
	// Logging for the best individual
	std::string log = ss.str() + "\tx\tvx\ty\tvy\ttheta\tomega\tul\tur\n";
	char h[999];
	////////////////////////////////////////////////////////

	// run evaluation of the ANN
	double  dt = .2, t;
	for(t = 0;t <= max_docking_time;t += dt) {
		// Perform the integration step
		stepper.next_step( sys, inputs, t, dt );
		
		// distance to the final position (0,0) = sqrt(x^2 + z^2)
		distance = sqrt(inputs[0] * inputs[0] + inputs[2] * inputs[2]) ;
		speed = sqrt(inputs[1]*inputs[1] + inputs[3]*inputs[3]);		// sqrt(vx^2 + vy^2)
		theta = inputs[4];

		// keep theta between -180° and +180°
		if(theta > M_PI) theta -= 2 * M_PI;
		if(theta < -1*M_PI) theta += 2 * M_PI;
		inputs[4] = theta;

		// Calculate return value
		retval = 1.0/((1 + distance) * (1 + fabs(theta)) * (1 + speed));
		
		// m_pi/6 = 30°
		if(distance < .1 && fabs(theta) < M_PI/6)
			counter_at_goal++;
		else counter_at_goal = 0;
		
		if(counter_at_goal >= needed_count_at_goal) retval = 1.0 + 2/t;		// we reached the goal!!! 2/t to give it a bit more range
		if(take_best)
			if(retval > best_retval) best_retval = retval;

		////////////////////////////////
		// LOGGING 
		// get more outputs
		out = sys.get_last_outputs();

		// Log the result (for later output & plotting)
		//printf("%.2f:\t%.3f\t%.3f\t%.4f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\n", t, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);
		sprintf(h, "%.2f:\t%.3f\t%.3f\t%.4f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\tCalc: %f\t%f\t%f\t%f", 
		 			t, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], out[0], out[1],
					retval, distance, theta, speed
		);	
		if(log_genome) {
			std::stringstream oss (std::stringstream::out);
			oss << *(ann_toolbox::multilayer_perceptron*)ann << std::endl;
			sprintf(h, "%s\tGenome:%s", h, oss.str().c_str());
		}			
		log = log + h + "\n";
		////////////////////////////////
		if(retval > 1.0) break;
	}
	
	// if take_best is FALSE we do not take the best overall but the result
	// at the end of the iteration (max_docking_time)
	if(!take_best) best_retval = retval;
	
	
	//////////////////////////////////////
	// Add the best fitness to the logger
 	if(max_log_fitness < best_retval) {
		sprintf(h, "ObjFun: return value:  %f\tdist\n", best_retval); //:%f theta: %f speed: %f\n, distance, theta, speed);
		log = log + h;

		max_log_fitness = best_retval;
		max_log_string = log;
	}
	/////////////////////////////////////
	
	
	// PaGMO minimizes the objective function!! therefore the minus here
	return -best_retval;
}

void docking::scale_outputs(std::vector<double> &outputs) const {
	for(size_t i = 0; i < outputs.size(); i++)  {
	 	outputs[i] = (outputs[i] - 0.5) * 2;		// to have the thrust from 0 and 1 to -1 to 1
	 	outputs[i] = outputs[i] * max_thrust;		// scale it
	}
}


// The dynamic system including the hill's equations
// state_type = x, vx, y, vy, theta, omega
void DynamicSystem::operator()( state_type &state , state_type &dxdt , double t ) {
	if(state.size() != 6) {
		pagmo_throw(value_error, "wrong number of parameters for the integrator");
	}
	
	// constants:
	const double nu = .08, mR = (1.5*.5);	// constant mass * radius of the s/c	
	
	double x = state[0];
	double vx = state[1];
	// not used: double y = state[2];
	double vy = state[3];
	double theta = state[4];
	double omega = state[5];
	
	// Send to the ANN to compute the outputs
	outputs = prob->ann->compute_outputs(state);
	prob->scale_outputs(outputs);
	
	double ul = outputs[0];
	double ur = outputs[1];	// maybe save them somewhere?
	
	if(t >= prob->breakdown_time && t <= prob->breadkdown_time + prob->breakdown_duration)
		ul = 0.0;
	
	dxdt[0] = vx;
	dxdt[1] = 2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta);
	dxdt[2] = vy;
	dxdt[3] = -2 * nu * vx + (ul + ur) * sin(theta);
	dxdt[4] = omega;
	dxdt[5] = (ul - ur) * 1/mR;
}

std::vector<double> DynamicSystem::get_last_outputs() { return outputs; }

}
}
