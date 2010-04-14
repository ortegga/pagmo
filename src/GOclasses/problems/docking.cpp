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
#include "../../odeint/odeint.hpp"


extern std::string max_log_string;
extern double max_log_fitness;
extern bool pre_evolve;

namespace pagmo {
namespace problem {	
	
// Constructor for the docking problem
docking::docking(ann_toolbox::neural_network* ann_, size_t random_positions, size_t in_pre_evo_strat, double max_time, double max_thr) :
	base(ann_->get_number_of_weights()),
	max_thrust(max_thr),
	max_docking_time(max_time),	
	ann(ann_),
	random_starting_positions(random_positions),
	pre_evolution_strategy(in_pre_evo_strat)
{						
	// the docking problem needs:
	// 	3 inputs (and starting conditions): x, y, theta
	//  2 outputs (control): ul, ur		(the two thrusters, [0,1] needs to be mapped to [-1,1])
	// and final conditions: x, y,	// later maybe theta, v
		
	// Set the boundaries for the values in the genome, this is important for the multilayer_perceptron!!
	set_lb(	std::vector<double> (ann->get_number_of_weights(), -10.0) );
	set_ub(	std::vector<double> (ann->get_number_of_weights(),  10.0) );
	
	random_start.clear();

	// set controller variables
	// disable genome logging
	log_genome = false;
	
	// for time neuron
	time_neuron_threshold = .99;

	// for vicinity stopping
	needed_count_at_goal = 5;
	vicinity_distance = vicinity_speed = 0.1;
	vicinity_orientation = M_PI/8;	
}


// the pre_evolution function, called before every evolution
// should generate new starting positions
// ( for now only once )
void docking::pre_evolution(population &pop) const {	
	if(pre_evolve) {
		pre_evolve = false;
		random_start.clear();
	}
	// Change the starting positions to random numbers (given by random_starting_positions number)	
	/// we do not want to change the position every pre_evolve!
	if( random_start.size() == random_starting_positions ) {
		return;
	} 
	
	std::cout << "Generating starting positions ...";

	generate_starting_positions();
	
	std::cout << "done!";
	

	// this is needed else we have a 'nan' result!
//	pop[0] = individual(*this, pop[0].get_decision_vector(), pop[0].get_velocity());
	std::cout << "\tRe-evaluating population ... "; 	std::cout.flush();
	//Re-evaluate the population with respect to the new seed (Internal Sampling Method)
	for (size_t i=0; i < pop.size(); ++i) {
		pop[i] = individual(*this, pop[i].get_decision_vector(), pop[i].get_velocity());
	}
	std::cout << " done!  " << std::endl;
}

// generating the starting positions
// depends on the strategy chosen
// TODO: check if we really are independent of attitude (4/6 in this function)
void docking::generate_starting_positions() const {
	// Fixed positions
	if(pre_evolution_strategy == docking::FIXED_POS) {
		// depending on the ann->get_number_of_inputs() we use 4 or 6
		// i.e. (we use the attitude or not)
		if(random_starting_positions >= 1) {
			double cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
			random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
		}

		if(random_starting_positions >= 2) {
			double cnd[] = { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
			random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
		}
		
		if(random_starting_positions >= 3) {
			double cnd[] = { -1.0, 0.0, -1.0, 0.0, 0.0, 0.0 };		
			random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
		}
		

/*		// DEBUG
		std::cout << "XY@: " << random_start[0][0] << "," << random_start[0][2] << ","<< random_start[0][4];
		std::cout << " XY@: " << random_start[1][0] << "," << random_start[1][2] << ","<< random_start[1][4];
		std::cout << " XY@: " << random_start[2][0] << "," << random_start[2][2] << ","<< random_start[2][4];
		std::cout << std::endl;*/		
		return;
	}
	
	switch(pre_evolution_strategy) {
	case SPOKE_POS:
		// generate starting positions one every (360/n) degree
		generate_spoke_positions(2.0, 2.0);
		break;
		
	case RAND_POS:
		// generate complete random starting positions (in doughnut)
		generate_random_positions(1.8, 2.0);
		break;

	case DONUT_FACING:
		// generate complete random starting positions (in doughnut)
		generate_random_positions_facing_origin(1.8, 2.0);
		break;
		
	case CLOUD_POS:
		generate_cloud_positions(2.0, M_PI, 0.1);
		break;
		
	case SPOKE_POS_HALF:
		// generate starting positions one every 360/n° 
		// -1 ==> means only in the negative x axis!
		generate_spoke_positions(1.8, 2.0, -1);
		break;	

	case SPOKE_8_POS:
		// generate starting positions random_starting_positions/m every (360/m)°
		generate_multi_spoke_positions(1.8, 2.0, 8);
		break;	
		
	case FULL_GRID:
		generate_full_grid_positions(5, 5);
		break;
	
	}	
}

void docking::generate_multi_spoke_positions(double r1, double r2, int spokes) const {
	rng_double drng = rng_double(static_rng_uint32()());
	double r, theta, x, y;	
	
	for(double a = 0; random_start.size() < random_starting_positions; a += (2*M_PI)/spokes) {
		r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
		x = r * cos(a);
		y = r * sin(a);
		theta = drng() * 2 * M_PI;	// theta between 0-2Pi
		// Start Condt:  x,  vx, y,  vy, theta, omega
		double cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
		random_start.push_back(std::vector<double> (cnd, cnd + 6)); //ann->get_number_of_inputs()
	}

}
void docking::generate_spoke_positions(double r1, double r2, int half) const {
	rng_double drng = rng_double(static_rng_uint32()());
	double r, theta, x, y;	
	
	for(double a = 0; random_start.size() < random_starting_positions; a += (2*M_PI)/random_starting_positions) {
		r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
		x = r * cos(a);
		// if we select a half the points should be in that half!
		if( (half == -1 && x > 0.0) || 
			(half == 1  && x < 0.0)  )  x = -x;		 
		y = r * sin(a);
		theta = drng() * 2 * M_PI;	// theta between 0-2Pi
		// Start Condt:  x,  vx, y,  vy, theta, omega
		double cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
		random_start.push_back(std::vector<double> (cnd, cnd + 6)); //ann->get_number_of_inputs()
	//	printf("\tPos%2d:%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", i+1,
	//		random_start[i][0], random_start[i][1], random_start[i][2], random_start[i][3], random_start[i][4], random_start[i][5]);
	}

}

void docking::generate_random_positions(double r1, double r2) const {
	rng_double drng = rng_double(static_rng_uint32()());
	double r, a, theta, x, y;	
	
	while(random_start.size() < random_starting_positions) {
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

void docking::generate_random_positions_facing_origin(double r1, double r2) const {
	rng_double drng = rng_double(static_rng_uint32()());
	double r, a, theta, x, y;	
	
	while(random_start.size() < random_starting_positions) {
		r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
		a = drng() * 2 * M_PI;	// alpha between 0-2Pi
		x = r * cos(a);
		y = r * sin(a);
		theta = atan2(-y, -x);	// theta is facing 0/0
		if(theta < 0) theta += 2 * M_PI;
		
		// Start Condt:  x,  vx, y,  vy, theta, omega
		double cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
		random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
	//	printf("\tPos%2d:%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", i+1,
	//		random_start[i][0], random_start[i][1], random_start[i][2], random_start[i][3], random_start[i][4], random_start[i][5]);
	}
}


void docking::generate_cloud_positions(double d, double angle, double rin) const {
        rng_double drng = rng_double(static_rng_uint32()());
        double r, theta, a, x, y;

        double x_start = d * cos(angle);
        double y_start = d * sin(angle);

        while(random_start.size() < random_starting_positions) {
                r = rin * drng();       // between 0 and rin
                a = drng() * 2 * M_PI;  // alpha between 0-2Pi
                x = x_start + r * cos(a);
                y = y_start + r * sin(a);
                theta = drng() * 2 * M_PI;      // theta between 0-2Pi
                // Start Condt:  x,  vx, y,  vy, theta, omega
                double cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
                random_start.push_back(std::vector<double> (cnd, cnd + 6));
        //      printf("\tPos%2d:%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", i+1,
        //              random_start[i][0], random_start[i][1], random_start[i][2], random_start[i][3], random_start[i][4], random_start[i][5]);
        }
}

void docking::generate_full_grid_positions(int h, int v) const {
	double r, theta, x, y;	
	double minx = -2, maxx = 2;
	double miny = -2, maxy = 2;	
	for(int i = 0; i < h; i++) {
		x = i * (maxx-minx) / (h - 1) + minx;
		for(int j = 0; j < v; j++) {
			y = j * (maxy-miny) / (v - 1) + miny;
			theta = 0;//drng() * 2 * M_PI;	// theta between 0-2Pi
			// Start Condt:  x,  vx, y,  vy, theta, omega
			double cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
			random_start.push_back(std::vector<double> (cnd, cnd + ann->get_number_of_inputs()));
		}
	}

}

// Objective function to be minimized
double docking::objfun_(const std::vector<double> &v) const {	
	static int cnt = 0;
	if(v.size() != ann->get_number_of_weights()) {
		pagmo_throw(value_error, "wrong number of weights in the chromosome");
	}
	std::string log = "", runlog = "";
	char h[999] = "";
	double average = 0.0;	
	size_t i; 

	///////////////////////////////////
	// LOGGING
	if(log_genome) {
		std::stringstream ss (std::stringstream::out);
		ss << *(ann_toolbox::multilayer_perceptron*)ann << std::endl;
		log = ss.str();	
	}
//	log += "\tx\tvx\ty\tvy\ttheta\tomega\tul\tur\tt-neuron\n";
	log += "\tx\ty\ttheta : ul\tur\tt-neur\n";
	////////////////////////////////
		
	if(++cnt == 50) {
		cnt = 0;
		std::cout << "#";std::cout.flush();
	}

	for(i = 0;i < random_start.size();i++) {		

		// Initialize ANN and interpret the chromosome
		ann->set_weights(v);

		// change starting position
		starting_condition = random_start[i];
		
		// Logging	
		log += "\nStarting Conditions: ";
		std::stringstream s;
		std::copy(starting_condition.begin(), starting_condition.end(), std::ostream_iterator<double>(s, ", "));
		log += s.str() + "\n";

		// evaluate the run with this starting position
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
	double dt = time_step, t = 0.0;

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
		
		
/*		
		//////////CHRISTOS
		// For christo#s (loading the chromosome from TwoDee) we have to change the order
		std::vector<double> temp;
		temp.push_back(inputs[0]);		// X
		temp.push_back(inputs[2]);		// Y
		temp.push_back(inputs[4]);		// rot
		temp.push_back(inputs[1]);		// Vx	
		temp.push_back(inputs[3]);		// Vy
		temp.push_back(inputs[5]);		// Vrot
		temp.push_back(inputs[6]);		// distance
		
		inputs = temp;
		//////////CHRISTOS
*/		
		
		// Send to the ANN to compute the outputs
		outputs = ann->compute_outputs(inputs);	

/*		///////////////////////
		// For testing ENGINGE FAILUREs
		outputs[0] = 0.5;
		outputs[1] = 0.5;
		
		// just one engine (test!)
		outputs[0] = 1.0;
		outputs[1] = 0.5;
		
		/////////////////////		*/
	
		// scale the outputs and send the first two (meaning the thruster!)
		// to the integrator
		std::vector<double> tmp = scale_outputs(outputs);
		if(tmp.size() > 2) tmp.pop_back();	//delete last		
		sys.set_outputs(tmp);
		
		if(inputs.size() == 4) {
			inputs.push_back(theta);
			inputs.push_back(omega);
		}
		if(inputs.size() > 6) inputs.pop_back();		// delete distance again

		// Perform the integration step
		stepper.next_step( sys, inputs, t, dt );
		
		// now time is already dt-ed !!!
		
		// evaluate the fitness
		retvals = evaluate_fitness(inputs, outputs, initial_distance, t+dt);

		retval 	= retvals[0];
		distance= retvals[1];
		speed 	= retvals[2];
		theta 	= inputs[4];
				
		////////////////////////////////
		// LOGGING
		// Log the result (for later output & plotting)
		char h[999];
		sprintf(h, "%2.1f: %2.6f\t %2.6f\t %2.6f : %1.3f\t %1.3f \t%1.2f \tDst/Theta/Speed: \t%f\t%f\t%f \tF: %.2f (%d)", 
		 			t+dt, inputs[0], inputs[2], inputs[4], tmp[0], tmp[1], outputs[2], //\t%1.4f 
					distance, theta, speed, retval, counter_at_goal
		);	
	    log = log + h + "\n";
	
//	std::cout << h << std::endl;
		////////////////////////////////		


		// TODO: needs a bit of cleanup ;)

		if( ann->get_number_of_outputs() == 3) {
			
			// using a time neuron
		    if(outputs[2] > time_neuron_threshold || 	// if the time neuron tells us the network is finished
		    t+dt > max_docking_time - 0.0001 ) {         // or the maximum time is reached
		    	// evaluate the output of the network
		        break; //return -retval;
		    }

		} else {
			
			// using stop condition

         	// if( (distance < vicinity_distance && fabs(theta) < vicinity_orientation)
			if(distance < vicinity_distance)
					counter_at_goal++;
		    else 	counter_at_goal = 0;
		
		    if(counter_at_goal >= needed_count_at_goal) {
				// add time bonus
//	            retval += retval * (max_docking_time - t+dt)/max_docking_time;
// //			retval = 1.0 + 2/(t+dt);		// we reached the goal!!! 2/t to give it a bit more range	
	            break;
		    }
		
		}
		
	}
		
	// PaGMO minimizes the objective function!! therefore the minus here
	return -retval;
}

/// Scale the outputs of the ANN to be more physically useful in the integrator.
/**
 * @param[in] outputs the outputs from the neural network, assumed to be between 0 and 1.
 * @return the inputs scaled to be between -MAX_THRUST and +MAX_THRUST.
 */
std::vector<double> docking::scale_outputs(const std::vector<double> outputs) const {
	std::vector<double> out(outputs.size());
	for(size_t i = 0; i < outputs.size(); i++)  {
	 	out[i] = (outputs[i] - 0.5) * 2;	// to have the thrust from 0 and 1 to -1 to 1
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
		case 10: { // something simpler?
			fitness = 0.0;
			if(distance < init_distance) {
				fitness = 1/( (1+distance) /* (1+speed) */);	
			}
		}break;
		
		case 2: {
			// no attitude in the fitness!!
			// Calculate return value			
			fitness = 0.0;
			if(distance < init_distance/2) {
				fitness = 1.0/((1 + distance) * (1 + speed));				
				if(distance < vicinity_distance && speed < 0.1)
					fitness += fitness * (max_docking_time - tdt)/max_docking_time;
			}		
		}break;
		
		case 88: {
			// based on Christos' TwoDee function
			// but only if fitness is high we give time bonus
			double timeBonus = (max_docking_time - tdt)/max_docking_time;
			fitness = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
			if (init_distance > distance/2) {
    			if(fitness > 0.87)
      				fitness += fitness * timeBonus;	
			} else
				fitness = 0;
		}break;			
			
		case 99: {
			// based on Christos' TwoDee function
			double timeBonus = (max_docking_time - tdt)/max_docking_time;
			double alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
			if (init_distance > distance/2) {
    			if (distance < vicinity_distance && fabs(theta) < vicinity_orientation && speed < vicinity_speed)
      				fitness = alpha + alpha * timeBonus;	
				else
					fitness = alpha;
			} else
				fitness = 0;
		}break;

		case 101: {
			// christo's but as soon as we reach the vicinity the
			// individual gets 1.00 as fitness + then the timeBonus
			double timeBonus = (max_docking_time - tdt)/max_docking_time;
			double alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
			if (init_distance > distance/2) {
    			if (distance < vicinity_distance && fabs(theta) < vicinity_orientation && speed < vicinity_speed)
      				fitness = 1 + timeBonus;	
				else
					fitness = alpha;
			} else
				fitness = 0;
		}break;
				
		
		default:
			pagmo_throw(value_error, "no such fitness function");
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
	const double nu = 0.08, mR = (1.5 * 0.5);	// constant mass * radius of the s/c	
	
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
//	state[4] = theta;
	
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


/// Setter for the starting condition, selecting from predefined conditions
void docking::set_start_condition(size_t number) {
	if(number < random_start.size())
		starting_condition = random_start[number];
	else
		pagmo_throw(value_error, "wrong index for random start position");
}

/// Setter for the starting condition (now only used for initialization)
void docking::set_start_condition(double *start_cnd, size_t size) {
	starting_condition = std::vector<double> (start_cnd, start_cnd + size);
}

/// Setter for the starting condition
void docking::set_start_condition(std::vector<double> &start_cond) {
	starting_condition = start_cond;
}

/// Setter
void docking::set_log_genome(bool b) {
	log_genome = b;
}

/// Setter
void docking::set_timeneuron_threshold(double t) {
	time_neuron_threshold = t;
}

/// Setter
void docking::set_fitness_function(int f) {
	fitness_function = f;
}

/// Setter
void docking::set_time_step(double dt) {
	time_step = dt;
}

/// Setter
void docking::set_vicinity_distance(double d) {
	vicinity_distance = d;
}

/// Setter
void docking::set_vicinity_speed(double d) {
	vicinity_speed = d;
}

/// Setter
void docking::set_vicinity_orientation(double d) {
	vicinity_orientation = d;
}


}
}
