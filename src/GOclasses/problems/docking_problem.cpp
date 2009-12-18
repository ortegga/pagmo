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
#include "GOproblem.h"
#include "../basic/population.h"
#include "docking_problem.h"

// Constructors
docking_problem::docking_problem(ann_toolbox::neural_network* ann_) :
	GOProblem(ann_->get_number_of_weights()),
	ann(ann_)	
{						
	// the docking problem needs:
	// 	3 inputs (and starting conditions): x, z, theta
	//  2 outputs (control): ul, ur		(the two thrusters, [0,1] needs to be mapped to [-1,1])
	// and final conditions: x, z,	// later maybe theta, v
	
	// x, z, theta (orientation), xdot, zdot, thetadot/omega
	double start_cnd[] = { 0.0, -2.0, M_PI, 0.0, 9.0, 0.0 };	
	starting_conditions = std::vector<double> (start_cnd, start_cnd + 6);
}

double docking_problem::objfun_(const std::vector<double> &v) const
{
	// std::cout<< "Chromosome length: "<< v.size() << std::endl;// << "vector: " << v[0] << std::endl;
	
	if(v.size() != ann->get_number_of_weights()) {
		pagmo_throw(value_error, "wrong number of weights in the chromosome");
	}
	
	// interpret chromosome as ANN
	ann->set_weights(v);
	
	std::vector<double> inputs = starting_conditions, outputs;
	
	// run evaluation of the ANN
	double docking_time = 9.0, integration_steps = 18;
	for(double t = 0;t <= docking_time;t += docking_time/integration_steps) {
		// get outputs from the network, with the current inputs
		outputs = ann->compute_outputs(inputs);
		std::cout<< "Output: Thruster0:"<< outputs[0] << "Thruster1:"<< outputs[1] << std::endl;		
		// ul = left thruster  = outputs[0]
		// ur = right thruster = outputs[1]

		// integrate
		// d_x/dt = xdot
//		integrate()
		// d_z/dt = ydot
		// d_theta/dt = omega
		// d_xdot/dt = 2 * eta * vz + 3 * eta * eta * x + (ul + ur) * cos (theta)
		// d_zdot/dt = -2 * eta * vx + (ul + ur) * sin (theta)
		// d_omega/dt = (ul - ur) * 1/(m*R)
	}
	
	double retval = 0;
	// distance to the final position is the return value!
	
	return retval;
}