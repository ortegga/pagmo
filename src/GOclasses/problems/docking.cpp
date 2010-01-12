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

#include "../../exceptions.h"
#include "../../Functions/rng/rng.h"
#include "../../ann_toolbox/neural_network.h"
#include "base.h"
#include "docking.h"
#include "../basic/population.h"
#include "../../odeint/odeint.hpp"

typedef std::vector<double> state_type;

namespace pagmo {
namespace problem {

// Constructors
docking::docking(ann_toolbox::neural_network* ann_) :
	base(ann_->get_number_of_weights()),
	ann(ann_)	
{						
	// the docking problem needs:
	// 	3 inputs (and starting conditions): x, z, theta
	//  2 outputs (control): ul, ur		(the two thrusters, [0,1] needs to be mapped to [-1,1])
	// and final conditions: x, z,	// later maybe theta, v
	
//	double start_cnd[] = { 0.0, -2.0, M_PI, 0.0, 0.0, 0.0 };	
	// Starting Conditions:  x, vx, y, vy, theta, omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
	starting_conditions = std::vector<double> (start_cnd, start_cnd + 6);
}

double docking::objfun_(const std::vector<double> &v) const
{
	// std::cout<< "Chromosome length: "<< v.size() << std::endl;// << "vector: " << v[0] << std::endl;
	
	if(v.size() != ann->get_number_of_weights()) {
		pagmo_throw(value_error, "wrong number of weights in the chromosome");
	}
	
	// interpret chromosome as ANN
	ann->set_weights(v);

	//////////////////////////////////////////////
	// for testing purposes print that here :)
	/*char tmp[512] = "";
	sprintf(tmp, "%f", v[0]);
	for(int i = 1; i < v.size(); i++) {
		sprintf(tmp, "%s,%f", tmp, v[i]);
	}	
	sprintf(tmp, "%s\0", tmp);
	std::cout << "Weights (" << ann->get_number_of_weights() << "): [" << tmp << "]" << std::endl;
	*/
	std::vector<double> inputs = starting_conditions, outputs;
	
	// Create Integrator Stepper
	odeint::ode_step_runge_kutta_4< state_type, double > stepper;
    
	
	// run evaluation of the ANN
	double docking_time = 9.0, integration_steps = 50, dt = docking_time/integration_steps;
	for(double t = 0;t <= docking_time;t += dt) {
		// get outputs from the network, with the current inputs
		outputs = ann->compute_outputs(inputs);
		std::cout<< "Output: Thruster0:"<< outputs[0] << "Thruster1:"<< outputs[1] << std::endl;		
		// ul = left thruster  = outputs[0]
		// ur = right thruster = outputs[1]

		// integration step
		stepper.next_step( hill, inputs, t, dt );

	}
	
	double retval = sqrt(inputs[0] * inputs[0] + inputs[2] * inputs[2]) ;
	// distance to the final position is the return value! sqrt(x^2 + z^2)
	
	return retval;
}

}
}