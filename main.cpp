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
 *   the Free Software Foundation; either version 3 of the License, or       *
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

#include <iostream>

#include "src/GOclasses/algorithms/CS.h"
#include "src/GOclasses/basic/island.h"
#include "src/GOclasses/problems/docking_problem.h"

#include "src/ann_toolbox/perceptron.h"
#include "src/ann_toolbox/multilayer_perceptron.h"
#include "src/ann_toolbox/elman_network.h"
#include "src/ann_toolbox/ctrnn.h"

using namespace std;

int main(){
// input = full state (6), output = thrusters (2)

// Perceptron
//	ann_toolbox::neural_network *ann = new ann_toolbox::perceptron(6, 2);
// MultiLayerPerceptron
//	ann_toolbox::neural_network *ann = new ann_toolbox::multilayer_perceptron(6, 2, 2);
// ElmanNetwork
//	ann_toolbox::neural_network *ann = new ann_toolbox::elman_network(6, 2, 2);
// CTRNN
	ann_toolbox::neural_network *ann = new ann_toolbox::ctrnn(6, 2, 2);

	docking_problem prob = docking_problem(ann);
	CSalgorithm algo(0.001);
	island isl = island(prob, algo, 1);
//	isl.evolve();
//	isl.join();

    cout << "-------------------- CTRNN: Best: " << isl.best().getFitness() << endl;

	return 0;
}
