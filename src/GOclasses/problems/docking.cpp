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

#include "base.h"
#include "docking.h"

#include "../../exceptions.h"
#include "../basic/population.h"
#include "../../ann_toolbox/neural_network.h"
#include "../../odeint/odeint.hpp"


extern std::string max_log_string;
extern double max_log_fitness;

namespace pagmo {
namespace problem {	
	
// Constructors
docking::docking(ann_toolbox::neural_network* ann_) :
	base(ann_->get_number_of_weights()),
	ann(ann_),
	max_thrust(0.1)
{						
	// the docking problem needs:
	// 	3 inputs (and starting conditions): x, z, theta
	//  2 outputs (control): ul, ur		(the two thrusters, [0,1] needs to be mapped to [-1,1])
	// and final conditions: x, z,	// later maybe theta, v
		
	// Set the boundaries for the values in the genome, this is important for the ANN
	set_ub(	std::vector<double> (ann->get_number_of_weights(), 10.0) );
	set_lb(	std::vector<double> (ann->get_number_of_weights(), -10.0) );
}

void docking::set_start_condition(double *start_cnd, size_t size) {
	starting_condition = std::vector<double> (start_cnd, start_cnd + size);
}

void docking::set_start_condition(std::vector<double> &start_cond) {
	starting_condition = start_cond;
}

// Objective function
double docking::objfun_(const std::vector<double> &v) const
{
	if(v.size() != ann->get_number_of_weights()) {
		pagmo_throw(value_error, "wrong number of weights in the chromosome");
	}
	
	// Create Integration Stepper
	odeint::ode_step_runge_kutta_4< state_type, double > stepper;    
	
	// Initialize ANN and interpret the chromosome
	ann->set_weights(v);

	// initialize the inputs (= starting conditions) to the ANN and allocate the outputs 
	std::vector<double> inputs = starting_condition, outputs;
	if(inputs.size() == 0) {	// no starting condition is defined
		// TODO generate random numbers for x and y (maybe also theta)
		double x = -2.0, y = -2.0;
		// Starting Conditions:  x, vx, y, vy, theta, omega
		double start_cnd[] = { x, 0.0, y, 0.0, 0.0, 0.0 };	
		inputs = std::vector<double> (start_cnd, start_cnd + 6);
	}
	
	// vector for the integrator to consist of the inputs and the outputs of the ANN
	state_type state;

	// Logging for the best individual
	std::string log = "\tx\tvx\ty\tvy\ttheta\tomega\tul\tur\n";
	char h[999];

	// run evaluation of the ANN
	double max_docking_time = 9.0/*, integration_steps = 50*/;
	double  dt = .15, t;
	for(t = 0;t <= max_docking_time;t += dt) {
		// get outputs from the network, using the current inputs
		outputs = ann->compute_outputs(inputs);

		// Scale the outputs
		scale_outputs(outputs);
		
		// Create the state vector for the integrator
		state = inputs;
		state.insert(state.end(), outputs.begin(), outputs.end());

		// Perform the integration step
		stepper.next_step( hill_equations, state, t, dt );

		// Log the result (for later output & plotting)
		//printf("%.2f:\t%.3f\t%.3f\t%.4f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\n", t, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);
		sprintf(h, "%.2f:\t%.3f\t%.3f\t%.4f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\n", t, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);		
		log = log + h;
		
		inputs = std::vector<double> (state.begin(), state.end()-2);
	}
	
	// distance to the final position (0,0) = sqrt(x^2 + z^2)
	double distance = sqrt(inputs[0] * inputs[0] + inputs[2] * inputs[2]) ;
	double speed = sqrt(inputs[1]*inputs[1] + inputs[3]*inputs[3]);		// sqrt(vx^2 + vy^2)
	double theta = inputs[4];

	// keep theta between -180° and +180°
	if(theta > M_PI) theta -= 2 * M_PI;
	if(theta < -1*M_PI) theta += 2 * M_PI;

	// Calculate return value
	double retval = 1.0/((1 + distance) * (1 + fabs(theta)) * (1 + speed));
	
	// Add the best fitness to the logger
 	if(max_log_fitness < retval) {
		sprintf(h, "ObjFun: return value:  %f\tdist:%f theta: %f speed: %f\n\n", retval, distance, theta, speed);
		log = log + h;

		max_log_fitness = retval;
		max_log_string = log;
	}
	
	// PaGMO minimizes the objective function!! therefore the minus here
	return -retval;
}

void docking::scale_outputs(std::vector<double> &outputs) const {
	for(size_t i = 0; i < outputs.size(); i++)  {
	 	outputs[i] = (outputs[i] - 0.5) * 2;		// to have the thrust between 1 and -1
	 	outputs[i] = outputs[i] * max_thrust;		// scale it
	}
}

// state_type = x, vx, y, vy, theta, omega
void docking::hill_equations( state_type &state , state_type &dxdt , double t ) {	
	if(state.size() != 8) {
		pagmo_throw(value_error, "wrong number of parameters for the integrator");
	}
	
	// constants:
	const double nu = .08, mR = (1.5*.5);	// constant mass * radius of the s/c	
	
	double x = state[0];
	double vx = state[1];
	// not used double y = state[2];
	double vy = state[3];
	double theta = state[4];
	double omega = state[5];
	double ul = state[6];
	double ur = state[7];
	
	dxdt[0] = vx;
	dxdt[1] = 2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta);
	dxdt[2] = vy;
	dxdt[3] = -2 * nu * vx + (ul + ur) * sin(theta);
	dxdt[4] = omega;
	dxdt[5] = (ul - ur) * 1/mR;
}


}
}
