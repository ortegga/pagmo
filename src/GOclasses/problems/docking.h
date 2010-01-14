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
// Docking problem.

class __PAGMO_VISIBLE docking : public base {
	public:
		docking(ann_toolbox::neural_network *ann_);

		virtual docking *clone() const { return new docking(*this); };
		virtual std::string id_object() const { return "Docking problem, using ANN to develop a robust controller"; }
		
		// The ODE system we want to integrate needs to be passed to the 
		// integrator. Here we have the Hill's equations.
		static void hill_equations( state_type &state , state_type &dxdt , double t );

	private:
		virtual double 		objfun_(const std::vector<double> &) const;
		void scale_outputs(std::vector<double> &) const;

		std::vector<double>	starting_conditions;
		
		// Reference to the neural network representation
		ann_toolbox::neural_network *ann;
		
		// TODO: Add integrator!
		//integrator		*solver;		
		// Variables/Constants for the ODE
		double nu, max_thrust, mR;
		
		
	//		mutable size_t		m_random_seed;
		
};
}
}
#endif
