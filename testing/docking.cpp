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
extern std::string *log_strings;
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
	
	// Starting Conditions:  x, vx, y, vy, theta, omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
	starting_conditions = std::vector<double> (start_cnd, start_cnd + 6);
	
	// Set the boundaries for the values in the genome, this is important for the ANN
	set_ub(	std::vector<double> (ann->get_number_of_weights(), 10.0) );
	set_lb(	std::vector<double> (ann->get_number_of_weights(), -10.0) );
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
	std::vector<double> inputs = starting_conditions, outputs;
	
	// vector for the integrator to consist of the inputs and the outputs of the ANN
	state_type state;

	// testing optimal control...
	// double UL[] = { 0.0000000000000000e+00, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 4.2628501724780124e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -2.1765169987709620e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02	}; 
	// double UR[] = { 0.0000000000000000e+00, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, 5.2147180378961100e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 5.0142709254069967e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, -9.9999999999999992e-02, 1.0000000000000001e-01, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02, 9.9999999999999992e-02 };
	// size_t idx = 0;

	std::string log = "\tx\tvx\ty\tvy\ttheta\tomega\tul\tur\n";
	char h[999];	

	// run evaluation of the ANN
	double max_docking_time = 9.0/*, integration_steps = 50*/;
	double  dt = .15, t;
	for(t = 0;t <= max_docking_time;t += dt) {
		// get outputs from the network, using the current inputs
		outputs = ann->compute_outputs(inputs);

 		// FOR TESTING!!!
		// outputs.clear();
		// outputs.push_back(UL[idx]);
		// outputs.push_back(UR[idx]);
		// idx++;
		
		// Scale the outputs
		scale_outputs(outputs);
		
		// Create the state vector for the integrator
		state = inputs;
		state.insert(state.end(), outputs.begin(), outputs.end());

		// Perform the integration step
		stepper.next_step( hill_equations, state, t, dt );

		// Log the result (for later plotting)
		// also try to put that somewhere else then!!
		//printf("%.2f:\t%.3f\t%.3f\t%.4f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\n", t, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);
		sprintf(h, "%.2f:\t%.3f\t%.3f\t%.4f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\n", t, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);		
		log = log + h;
		
		inputs = std::vector<double> (state.begin(), state.end()-2);
	}
	
	// distance to the final position (0,0) = sqrt(x^2 + z^2)
	
//	inputs[0] = inputs[2] = 0.0;
//	inputs[1] = inputs[3] = 0.0;
	double distance = sqrt(inputs[0] * inputs[0] + inputs[2] * inputs[2]) ;
	double speed = sqrt(inputs[1]*inputs[1] + inputs[3]*inputs[3]);		// sart(vx^2 + vy^2)
	double theta = inputs[4];
	// keep theta between -180° and +180°
	if(theta > M_PI) theta -= 2 * M_PI;
	if(theta < -1*M_PI) theta += 2 * M_PI;

//	printf("\tx\tvx\ty\tvy\ttheta\tomega\n");	
//	printf("%.2f:\t%.3f\t%.3f\t%.4f\t%.2f\t%.2f\t%.2f\n", t, state[0], state[1], state[2], state[3], state[4], state[5]);
	
	double retval = 1.0/((1 + distance) * (1 + fabs(theta)) * (1 + speed));
	sprintf(h, "ObjFun: return value:  %f\tdist:%f theta: %f speed: %f\n\n", retval, distance, theta, speed);
	log = log + h;
	
 	if(max_log_fitness < retval) {
		max_log_fitness = retval;
		max_log_string = log;
		log_strings[1] = log;		
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
	//printf("ul: %f\t ur: %f\n", ul, ur);
}


}
}
