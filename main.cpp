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
#include <fstream>

#include "src/GOclasses/basic/archipelago.h"
#include "src/GOclasses/basic/island.h"

#include "src/GOclasses/problems/docking.h"
#include "src/GOclasses/algorithms/asa.h"
#include "src/GOclasses/algorithms/de.h"
#include "src/GOclasses/algorithms/sga.h"

#include "src/ann_toolbox/multilayer_perceptron.h"
#include "src/ann_toolbox/ctrnn.h"

using namespace std;
using namespace pagmo;

extern std::string max_log_string = "";
extern double max_log_fitness = 99.0;


std::vector<double> load_chromosome_twodee(std::string fname) {
	fstream file;
	file.open(fname.c_str(), ios::in);
	if ( file.fail() ) {
	    printf("Error opening %s - %s\n", fname.c_str(), strerror(errno));
		exit(0);
	}
	unsigned int len;
    file >> len;
    std::vector<double> chromosome(len, 0);

    for (int i = 0; i < len; i++) {
        file >> chromosome[i];
    }
    file.close();

	fprintf(stderr, "Network weights loaded from %s\n", fname.c_str());
	
	return chromosome;	
}

void evaluate_twodee() {
	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
	//	double start_cnd[] = { -1.615471, 0.0, 0.438620, 0.0, 0.0, 0.0 };	
	ann_toolbox::multilayer_perceptron ann(7, 20, 3);
	std::vector<double> v = load_chromosome_twodee("chromosome.log");

	problem::docking prob = problem::docking(&ann, 8, problem::docking::SPOKE_POS/*FIXED_POS*/, 25, 0.1);
	prob.set_start_condition(start_cnd, 6);	
	prob.set_timeneuron_threshold(.99);
	prob.set_log_genome(true);
	prob.set_fitness_function(99);	// 10  = no attitude

	// TESTING
	cout << "Created the problem!" << endl;	
	cout << "Calling the objfun_ ... " << endl;
	max_log_fitness = 0;
	//prob.pre_evolution(pop);
	prob.generate_starting_positions();
	max_log_fitness = prob.objfun_(v);
	cout << "\r=== Twodee fitness: " << max_log_fitness << endl;	
	cout << max_log_string << endl;	

	ofstream myfile;
	myfile.open ("bestrun.dat");
	myfile << max_log_string << endl;
	myfile.close();	
	max_log_fitness = 0;


	exit(0);
	
}

int main(){
	double best_fitness = 1;	// minimizing!!
	
	// for evaluating twodee chromosome
	// evaluate_twodee();

	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
	////////////////////////////////////////////////
	// Define the neural network
	// // CTRNN -- CTRNN -- CTRNN -- CTRNN -- CTRNN -- CTRNN
	// // CTRNN -- CTRNN -- CTRNN -- CTRNN -- CTRNN -- CTRNN
	// // CTRNN, 6 inputs, 10 hidden, 3 outputs
	// ann_toolbox::ctrnn ann(6, 5, 3);
	// ann.set_weights_lower_bound(-1.0);
	// ann.set_weights_upper_bound( 1.0);	
	// ann.set_bias_lower_bound(-1.0);
	// ann.set_bias_upper_bound( 1.0);
	// ann.set_tau_lower_bound(-1.0);	// from Christo's calculations
	// ann.set_tau_upper_bound( 2.0);	// my try (Christo's 2.3)

	// MultiLayer Perceptron
	// MultiLayer Perceptron
	ann_toolbox::multilayer_perceptron ann(7, 20, 3);



	////////////////////////////////////////////////
	// Define the problem						positions, strategy, 				max_time, max_thrust
	problem::docking prob = problem::docking(&ann, 8, problem::docking::SPOKE_POS/*FIXED_POS*/, 25, 0.1);
	prob.set_start_condition(start_cnd, 6);	
	prob.set_timeneuron_threshold(.99);
	prob.set_log_genome(true);
	prob.set_fitness_function(99);	// 10  = no attitude, 99 = christo's
	
	algorithm::sga algo( 2, 	// Generations
					0.9,	// CR		the value drng has to be above to stop
					0.1,	// Mutation	
					1);		// Elitism
/*	algorithm::sga algo( 20, 	// Generations
						0.5,	// CR
						0.15,	// Mutation	
						1, 	// Elitism
						0.0,
						2,	//random
						0); 	// no roulette selection
						*/
						
//	algorithm::de algo(20, 0.7, 0.5, 2);


	////////////////////////////////////////////////
	// Create the archipelag/islands
	cout << "Creating an archipelago...";
	archipelago arch = archipelago(prob, algo, 1, 100);	// 1 island, with x individuals
	cout << "Created!";



	///////////////////////////////////////////////
	// Start evolution
	cout << "\rStarting evolution ...                        " << endl;	
	ofstream myfile;
	int i = 0;
	
	// run until we are quite good
	while(best_fitness > -1.2/*i++ < 6/**/) { 
		cout << "\rGeneration #" << i << " ["; cout.flush();
		
		arch.evolve();
		arch.join();
		i++;

 		cout << "] best: " << arch.best().get_fitness() << "                     " << endl;

		//////////////////////////////////////////
		// logging
		if(max_log_fitness < best_fitness) {
			best_fitness = max_log_fitness;	
			cout << "\r=== Best increased @ #" << i << ": " << max_log_fitness << endl;
			myfile.open ("bestrun.dat");
			myfile << arch[0] << endl;
			myfile << max_log_string << endl;
			myfile.close();	
			max_log_fitness = 0;
		}
		std::string h = boost::lexical_cast<std::string>(i);
		while(h.length() < 3) h = "0" + h;
		std::string s = "best-" + h + ".dat";
		myfile.open (s.c_str());
		myfile << max_log_string << endl;
		myfile.close();	
		//////////////////////////////////////////
		cout.flush();		
	}	
	
	// finished
	cout << "==================== Best Overall: " << best_fitness << "\t(i=" << i << ")" << endl;

	return 0;
}
