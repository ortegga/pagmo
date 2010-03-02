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

// Created by Juxi Leitner on 2010-02-11

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
//#include "../../ann_toolbox/ctrnn.h"
#include "../../odeint/odeint.hpp"


extern std::string max_log_string;
extern double max_log_fitness;

namespace pagmo {
namespace problem {	
	
// Constructors
docking::docking(ann_toolbox::neural_network* ann_, size_t random_positions, size_t in_pre_evo_strat, double max_time, double max_thr) :
	base(ann_->get_number_of_weights()),
	max_thrust(max_thr),
	max_docking_time(max_time),	
	time_neuron_threshold(.99),
	ann(ann_),
	random_starting_postions(random_positions),
	pre_evolution_strategy(in_pre_evo_strat)
{						
	// the docking problem needs:
	// 	3 inputs (and starting conditions): x, z, theta
	//  2 outputs (control): ul, ur		(the two thrusters, [0,1] needs to be mapped to [-1,1])
	// and final conditions: x, z,	// later maybe theta, v
		
	// Set the boundaries for the values in the genome, this is important for the multilayer_perceptron!!
	set_lb(	std::vector<double> (ann->get_number_of_weights(), -10.0) );
	set_ub(	std::vector<double> (ann->get_number_of_weights(),  10.0) );
	
	// disable genome logging
	log_genome = false;
	
	random_start.clear();
	
	needed_count_at_goal = 5;
	vicinity_distance = 0.15;
	vicinity_orientation = M_PI/10;
}

void docking::set_start_condition(size_t number) {
	if(number < random_start.size())
		starting_condition = random_start[number];
	else
		pagmo_throw(value_error, "wrong index for random start position");
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

void docking::set_timeneuron_threshold(double t) {
	time_neuron_threshold = t;
}

void docking::set_fitness_function(int f) {
	fitness_function = f;
}

void docking::pre_evolution(population &pop) const {	
	std::cout << "Generating new starting positions ... [" << pre_evolution_strategy << "] ";
	// Change the starting positions to random numbers (given by random_starting_positions number)
	random_start.clear();
		
	if(pre_evolution_strategy == docking::FIXED_POS) {
		// Fixed postions
		
		// depending on the ann->get_number_of_inputs() we use 4 or 6
		// i.e. (we use the attitude or not)
		double cnd[] = { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
		random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
		std::cout << "1,";

		cnd[0] = -1; cnd[2] = -1;
		random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
		std::cout << "2,";
		
		cnd[0] = -2; cnd[2] = 0;	
		random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
		std::cout << "3, done! ";
		
/*		// DEBUG
		std::cout << "XY@: " << random_start[0][0] << "," << random_start[0][2] << ","<< random_start[0][4];
		std::cout << " XY@: " << random_start[1][0] << "," << random_start[1][2] << ","<< random_start[1][4];
		std::cout << " XY@: " << random_start[2][0] << "," << random_start[2][2] << ","<< random_start[2][4];
		std::cout << std::endl;*/
		
		return;
	}
	
	switch(pre_evolution_strategy) {
	case SPOKE_POS:
		// generate starting positions one every 360/n° 
		generate_spoke_positions(1.5, 2.0);
		break;		
	case RAND_POS:
		// generate complete random starting positions (in doughnut)
		generate_random_positions(1.5, 2.0);
		break;
	}

	std::cout << "done!         \n";	
	
	std::cout << "\rRe-evaluating population ... ";
	//Re-evaluate the population with respect to the new seed (Internal Sampling Method)
	for (size_t i=0; i < pop.size(); ++i) {
		pop[i] = individual(*this, pop[i].get_decision_vector(), pop[i].get_velocity());
	}
	std::cout << "done!           \n";
	std::cout.flush();
}


void docking::generate_spoke_positions(double r1, double r2) const {
	rng_double drng = rng_double(static_rng_uint32()());
	double r, theta, x, y;	
	
	for(int i = 0; random_start.size() < random_starting_postions; i += 2*M_PI/random_starting_postions) {
		r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
		x = r * cos(i);
		y = r * sin(i);
		theta = drng() * 2 * M_PI;	// theta between 0-2Pi
		// Start Condt:  x,  vx, y,  vy, theta, omega
		double cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
		random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
	//	printf("\tPos%2d:%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", i+1,
	//		random_start[i][0], random_start[i][1], random_start[i][2], random_start[i][3], random_start[i][4], random_start[i][5]);
	}

}
void docking::generate_random_positions(double r1, double r2) const {
	rng_double drng = rng_double(static_rng_uint32()());
	double r, a, theta, x, y;	
	
	while(random_start.size() < random_starting_postions) {
		r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
		a = drng() * 2 * M_PI;	// alpha between 0-2Pi
		x = r * cos(a);
		y = r * sin(a);
		theta = drng() * 2 * M_PI;	// theta between 0-2Pi
		// Start Condt:  x,  vx, y,  vy, theta, omega
		double cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
		random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
	//	printf("\tPos%2d:%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", i+1,
	//		random_start[i][0], random_start[i][1], random_start[i][2], random_start[i][3], random_start[i][4], random_start[i][5]);
	}
}

// Objective function to be minimized
double docking::objfun_(const std::vector<double> &v) const {	
	if(v.size() != ann->get_number_of_weights()) {
		pagmo_throw(value_error, "wrong number of weights in the chromosome");
	}
	std::string log = "", runlog = "";
	char h[999] = "";
	
	///////////////////////////////////
	// LOGGING
	if(log_genome) {
		std::stringstream ss (std::stringstream::out);
		ss << *(ann_toolbox::multilayer_perceptron*)ann << std::endl;
		log = ss.str();	
	}
//	log += "\tx\tvx\ty\tvy\ttheta\tomega\tul\tur\tt-neuron\n";
	log += "\tx\ty\ttheta : ul\tur\n";	
	////////////////////////////////
		
	size_t i;

//	std::cout << "#";std::cout.flush();

	double average = 0.0;	
	for(i = 0;i < random_start.size();i++) {		
//		std::cout << "OBJFUN_" << random_start.size() << " " << log << " # " << std::endl;

		// Initialize ANN and interpret the chromosome
		ann->set_weights(v);

		// change starting position
		starting_condition = random_start[i];
			
		log += "Starting Conditions: ";
		std::stringstream s;
		std::copy(starting_condition.begin(), starting_condition.end(), std::ostream_iterator<double>(s, ", "));
		log += s.str() + "\n";
		
		average += one_run(log);	
	}
	average = average / i;
	
	//////////////////////////////////////
	// Add the best fitness to the logger
 	if(max_log_fitness > average) {
		sprintf(h, "docking::objfun_: return value:  %f\n", average); //:%f theta: %f speed: %f\n, distance, theta, speed);
		log = log + h;

		max_log_fitness = average;
		max_log_string = log;
	}
	/////////////////////////////////////
	
	return average;
}

double docking::one_run(std::string &log) const {	
	// Helper variables
	double retval = 0.0;//, best_retval = 0.0;
	double dt = .1, t = 0.0;

	// Integrator System
	DynamicSystem sys(this);
	// Create Integration Stepper
	//odeint::ode_step_euler< std::vector<double>, double > stepper;	
	odeint::ode_step_runge_kutta_4< std::vector<double>, double > stepper;

	// initialize the inputs (= starting conditions) to the ANN and allocate the outputs 
	std::vector<double> inputs = starting_condition, outputs, retvals;
	double initial_distance = sqrt(inputs[0] * inputs[0] + inputs[2] * inputs[2]);	
	double distance = initial_distance, speed;
	double theta = 0.0, omega = 0.0;

	size_t counter_at_goal = 0;

	// run evaluation of the ANN
	for(t = 0.0;t < max_docking_time /*+ 0.0001*/;t += dt) {		

		// distance to the final position (0,0) = sqrt(x^2 + z^2)
		if(ann->get_number_of_inputs() == 7) inputs.push_back(distance);
		if(ann->get_number_of_inputs() == 4 && inputs.size() == 6) {
			omega = inputs[5]; inputs.pop_back();			
			theta = inputs[4]; inputs.pop_back();
		}	
		
		// Send to the ANN to compute the outputs
		outputs = ann->compute_outputs(inputs);	
	
		// scale the outputs and send the first two (meaning the thruster!)
		// to the integrator
		std::vector<double> tmp = scale_outputs(outputs);
		if(tmp.size() > ann->get_number_of_outputs()) tmp.pop_back();	//delete last
		sys.set_outputs(tmp);
		
		// state for here has to be 6 elements (for the integrator!)
		if(inputs.size() == 4) {
			inputs.push_back(theta);
			inputs.push_back(omega);
		} 
		if(inputs.size() > 6) inputs.pop_back();		// delete distance again

		// Perform the integration step
		stepper.next_step( sys, inputs, t, dt );
		
		// now time is already dt-ed !!!
		
		// for time neuron stuff
		//if( out[2] > time_neuron_threshold || 		// if the time neuron tells us the network is finished
		//   tdt > max_docking_time - 0.0001 ) { 	// or the maximum time is reached
		// evaluate the output of the network

		retvals = evaluate_fitness(inputs, outputs, initial_distance, t+dt);

		retval 	= retvals[0];
		distance= retvals[1];
		speed 	= retvals[2];
		theta 	= inputs[4];
				
		////////////////////////////////
		// LOGGING
		// Log the result (for later output & plotting)
		char h[999];
		sprintf(h, "%2.1f: %2.6f\t %2.6f\t %2.6f : %1.3f\t %1.3f \tDst/Theta/Speed: \t%f\t%f\t%f \tF: %.2f (%d)", 
		 			t+dt, inputs[0], inputs[2], inputs[4], tmp[0], tmp[1], //outputs[2], \t%1.4f 
					distance, theta, speed, retval, counter_at_goal
		);	
	/*	printf("%.2f:\t%.6f\t%.6f\t%.6f\t%.6f\t%.2f\t%.2f\t%.3f\t%.3f\t%.3f\tCalc: %f\t%f\t%f\t%f\n", 
		 			t+dt, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], out[0], out[1], out[2],
					retval, distance, theta, speed
		);	*/
	    log = log + h + "\n";
		////////////////////////////////		
		
		
		// stop conditions
				
		// If we chose to do it this way
	//	if( (distance < vicinity_distance && fabs(theta) < vicinity_orientation)
		if(distance < vicinity_distance)
			counter_at_goal++;
		else counter_at_goal = 0;
		if(counter_at_goal >= needed_count_at_goal) {
			retval = 1.0 + 2/(t+dt);		// we reached the goal!!! 2/t to give it a bit more range
			break;
		}
		
	}
		
	// PaGMO minimizes the objective function!! therefore the minus here
	return -retval;
}

std::vector<double> docking::scale_outputs(const std::vector<double> outputs) const {
	std::vector<double> out(outputs.size());
	for(size_t i = 0; i < outputs.size(); i++)  {
	 	out[i] = (outputs[i] - 0.5) * 2;		// to have the thrust from 0 and 1 to -1 to 1
	 	out[i] = out[i] * max_thrust;		// scale it
	}
	return out;
}

std::vector<double> docking::evaluate_fitness(std::vector<double> state, std::vector<double> out, double init_distance, double tdt) const {
	double fitness = 0.0;
	double distance = sqrt(state[0] * state[0] + state[2] * state[2]);
	double speed    = sqrt(state[1] * state[1] + state[3] * state[3]);		// sqrt(vx^2 + vy^2)

	double theta = state[4];
	// keep theta between -180 and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;	
	
	switch(fitness_function) {
		case 0:	fitness = 1/( 1 + distance );
			break;
		case 1:	fitness = 1/( (1+distance) * (1+speed) * (1+fabs(theta)) );
			break;
		case 10:fitness = 1/( (1+distance) * (1+speed) );	// simpler?
			break;
		case 2:
			// keep theta between -180° and +180°
			if(theta > M_PI) theta -= 2 * M_PI;
			if(theta < -1*M_PI) theta += 2 * M_PI;
			// Calculate return value			
			fitness = 0.0;
			if(distance < init_distance) {
				fitness = 1.0/((1 + distance) * (1 + fabs(theta)) * (1 + speed));				
				if(distance < vicinity_distance && fabs(theta) < vicinity_orientation && speed < 0.1)
					fitness += fitness * (max_docking_time - tdt)/max_docking_time;
			}		
			break;
	}
	std::vector<double> ret;

	ret.push_back(fitness);
	ret.push_back(distance);
	ret.push_back(speed);
	
	return ret;
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
	double y = state[2];
	double vy = state[3];
	double theta = state[4];	
	double omega = state[5];
	
	double distance = sqrt(x * x + y * y);
	std::vector<double> in = state;
	in.push_back(distance);
	
	// keep theta between -180° and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;
	state[4] = theta;
	
	double ul = outputs[0];
	double ur = outputs[1];
	
/*	if(t >= prob->breakdown_time && t <= prob->breadkdown_time + prob->breakdown_duration)
		ul = 0.0;*/
	
	dxdt[0] = vx;
	dxdt[1] = 2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta);
	dxdt[2] = vy;
	dxdt[3] = -2 * nu * vx + (ul + ur) * sin(theta);
	dxdt[4] = omega;
	dxdt[5] = (ul - ur) * 1/mR;
	
//	printf("\t\t%f, %f, %f", outputs[0], outputs[1], outputs[2]);
//	printf("\t| %f, %f, %f, %f, %f, %f", dxdt[0], dxdt[1], dxdt[2], dxdt[3], dxdt[4], dxdt[5]);
//	printf("\n");
}

void DynamicSystem::set_outputs(std::vector<double> out) { outputs = out; }

}
}
