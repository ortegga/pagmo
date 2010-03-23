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
#include <string>

#include "src/GOclasses/basic/archipelago.h"
#include "src/GOclasses/basic/island.h"
#include "src/GOclasses/basic/population.h"
#include "src/GOclasses/basic/individual.h"

#include "src/GOclasses/problems/docking.h"
#include "src/GOclasses/algorithms/asa.h"
#include "src/GOclasses/algorithms/de.h"
#include "src/GOclasses/algorithms/sga.h"

#include "src/ann_toolbox/multilayer_perceptron.h"
#include "src/ann_toolbox/ctrnn.h"

#include "src/config_toolbox/config_parser.h"

using namespace std;
using namespace pagmo;

extern std::string max_log_string = "";
extern double max_log_fitness = 99.0;
extern bool pre_evolve = false;


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

int main(int argc, char *argv[]){
	// for evaluating twodee chromosome
	// evaluate_twodee();

	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
	char *config_filename;
	if ( argc >= 2 ) config_filename = argv[1];
	else	config_filename = NULL;	
	config_parser config(config_filename);
	
	/////////////////////////////////
	// Definitions
	int ann_input_neurons  = 7;			config.read_parameter("INPUTS", ann_input_neurons);
	int ann_hidden_neurons = 11;		config.read_parameter("HIDDEN", ann_hidden_neurons);
	int ann_output_neurons = 2;			config.read_parameter("OUTPUTS", ann_output_neurons);
	
	int prob_positions = 1;				config.read_parameter("POSITIONS", prob_positions);
	int prob_pos_strategy = problem::docking::CLOUD_POS;	config.read_parameter("POS_STRATEGY", prob_pos_strategy);
	int prob_fitness_function = 99;		config.read_parameter("FITNESS", prob_fitness_function);
	int prob_timeneuron_threshold = 99;	config.read_parameter("TIMENEURON_THRESHOLD", prob_timeneuron_threshold);
	int prob_maximum_time = 25;			config.read_parameter("MAX_TIME", prob_maximum_time);

	double integrator_timestep = 0.1;	config.read_parameter("INTEGRATOR_STEP", integrator_timestep);
	int evolution_stuck_threshold = 11; config.read_parameter("STUCK_THRESHOLD", evolution_stuck_threshold);
	
	int algo_generations = 11;		config.read_parameter("GENERATIONS", algo_generations);
	int algo_crossover = 90;    	config.read_parameter("CROSSOVER", algo_crossover);
	int algo_mutation = 30;     	config.read_parameter("MUTATION_RATE", algo_mutation);
	int algo_elitism = 1;			config.read_parameter("ELITISM", algo_elitism);

	int islands = 1;				config.read_parameter("ISLANDS", islands);
	int individuals = 33;			config.read_parameter("INDIVIDUALS", individuals);

	double vicinity_distance = 0.1;		config.read_parameter("VICINITY_DISTANCE", vicinity_distance);
	double vicinity_speed = 0.1;		config.read_parameter("VICINITY_SPEED", vicinity_speed);
	double vicinity_orientation = 0.1;	config.read_parameter("VICINITY_ORIENTATION", vicinity_orientation);	
	
	
	time_t seconds = time (NULL);
	std::string run_id = boost::lexical_cast<std::string>(seconds);
	//////////////////////////////////////////
	config.report();
	
	/////////////////////////////////////////
	// Print all values!
	string configs = "";
	configs += "Run Information: ID=" + boost::lexical_cast<std::string>(run_id) + "\n";
	configs += "Input Neurons:\t\t" + boost::lexical_cast<std::string>(ann_input_neurons) + "\n";
	configs += "Hidden Neurons:\t\t" + boost::lexical_cast<std::string>(ann_hidden_neurons) + "\n";
	configs += "Output Neurons:\t\t" + boost::lexical_cast<std::string>(ann_output_neurons) + "\n";
	
	configs += "Positions: \t\t" + boost::lexical_cast<std::string>(prob_positions) + "\n";
	configs += "Pos. Strategy:\t\t" + boost::lexical_cast<std::string>(prob_pos_strategy) + "\n";
	configs += "Fitness Function:\t" + boost::lexical_cast<std::string>(prob_fitness_function) + "\n";
	configs += "TimeNeuron Threshold:\t" + boost::lexical_cast<std::string>(prob_timeneuron_threshold) + "\n";
	configs += "Maximum Time:\t" + boost::lexical_cast<std::string>(prob_maximum_time) + "\n";

	configs += "Integration Step:\t" + boost::lexical_cast<std::string>(integrator_timestep) + "\n";
	configs += "Evolution Stuck:\t" + boost::lexical_cast<std::string>(evolution_stuck_threshold) + "\n";

	configs += "Generations:\t\t" + boost::lexical_cast<std::string>(algo_generations) + "\n";
	configs += "Crossover Rate:\t\t" + boost::lexical_cast<std::string>(algo_crossover/100.0) + "\n";
	configs += "Mutation Rate:\t\t" + boost::lexical_cast<std::string>(algo_mutation/100.0) + "\n";
	configs += "Elitism:\t\t" + boost::lexical_cast<std::string>(algo_elitism) + "\n";

	configs += "Islands:\t\t" + boost::lexical_cast<std::string>(islands) + "\n";
    configs += "Individuals:\t\t" + boost::lexical_cast<std::string>(individuals) + "\n";

    configs += "Vicinity Distance:\t" + boost::lexical_cast<std::string>(vicinity_distance) + "\n";
    configs += "Vicinity Speed:\t\t" + boost::lexical_cast<std::string>(vicinity_speed) + "\n";
    configs += "Vicinity Orientation:\t" + boost::lexical_cast<std::string>(vicinity_orientation) + "\n";
	configs += "-------------------------------------\n";
	cout << configs;
	///////////////////////////////////////////////////////////
		
		
		
		
/*		TODO
			add vicinity_distance
			and vicinity_orientation
			and vicinity_speed to these variables
			also add setter methods in the problem to set them!!!
*/			
			
	////////////////////////////////////////////////
	// Define the neural network
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
	ann_toolbox::multilayer_perceptron ann(ann_input_neurons, ann_hidden_neurons, ann_output_neurons);


	////////////////////////////////////////////////
	// Define the problem						positions, strategy, 				max_time, max_thrust
	problem::docking prob = problem::docking(&ann, prob_positions, prob_pos_strategy, prob_maximum_time, 0.1);
	prob.set_start_condition(start_cnd, 6);	
	prob.set_log_genome(true);

	prob.set_fitness_function(prob_fitness_function);
	prob.set_timeneuron_threshold(prob_timeneuron_threshold/100.0);
	
	prob.set_time_step(integrator_timestep);
	
	prob.set_vicinity_distance(vicinity_distance);
	prob.set_vicinity_speed(vicinity_speed);
	prob.set_vicinity_orientation(vicinity_orientation);
			
	algorithm::sga algo(algo_generations, 	// Generations
						algo_crossover/100.0,	// CR
						algo_mutation/100.0,	// Mutation	
						algo_elitism );/*, 	// Elitism
						0.0,
						  2,	//random
						  0); 	// no roulette selection*/
						
//	algorithm::de algo(20, 0.7, 0.5, 2);


	////////////////////////////////////////////////
	// Create the archipelag/islands
	cout << "Creating an archipelago...";
	archipelago arch = archipelago(prob, algo, islands, individuals);
	cout << "Created!";


	///////////////////////////////////////////////
	// Start evolution
	cout << "\rStarting evolution ...          ID: " << run_id << "            " << endl;	
	
	ofstream myfile;
	int i = 0, lasti = 0;
	double best_fitness = 99.9;	// minimizing!!
	double last_fitness = 99.9;
	
	//vector<individual> good_ones;

	pre_evolve = true;		// generate random positions at first
	// run until we are quite good
	while(best_fitness > -1.7/*i++ < 6/**/) { 
		cout << "\r                                                          "
			 << "                                                            ";
		cout << "\rGeneration #" << i << " ["; cout.flush();
		
		max_log_fitness	= 0.0;		
		arch.evolve();
		arch.join();
		i++;

		cout << "] best: " << arch.best().get_fitness() << ": " << best_fitness << ":" << last_fitness << "--" << i-lasti-1 << endl;// << "/" << arch[0].get_average_fitnes()?; // << "  lasti: " << (i-lasti);
			
		//////////////////////////////////////////
		// logging
		if(max_log_fitness < best_fitness) {
			best_fitness = max_log_fitness;	
			cout << "\r=== Best increased @ #" << i-1 << ": " << max_log_fitness << endl;

			// write to file
			std::string h = "id_" + run_id + "_bestrun.dat";
			myfile.open (h.c_str());
//			myfile << "ID: " << run_id << endl;
			myfile << configs << endl;//expected
			myfile << max_log_string << endl;
			myfile.close();	
			lasti = i-1;
			
			// save good ones
			//good_ones.push_back(arch.best());
		}
		if(max_log_fitness < last_fitness) {	
			last_fitness = max_log_fitness;
			std::string h = boost::lexical_cast<std::string>(i-1);
			while(h.length() < 5) h = "0" + h;
			std::string s = "id_" + run_id + "_genbest-" + h + ".dat";
			myfile.open (s.c_str());
//			myfile << "ID: " << run_id << endl;
			myfile << configs << endl;
			myfile << max_log_string << endl;
			myfile.close();	
			//////////////////////////////////////////
			lasti = i-1;
			
			// try to put it online!
			std::string cmd = "curl -H \"Expect: \" -F \"file=@";
			cmd += s += "\" -F \"submit=submit\" -F \"hostname=`hostname`\" http://juxi.net/projects/EvolvingDocking/upload.php";
			cmd += " -o tmp.out";
			int ret = system (cmd.c_str());	
			
		}
		cout.flush();		

		// randomize positions if we seem to be stuck
		if( (i - 1 - lasti) >= evolution_stuck_threshold ) {
			pre_evolve = true;
			lasti = i - 1;
			last_fitness = 0.0;
		}
	}	
	
	// finished
	cout << "==================== Best Overall: " << best_fitness << "\t(i=" << i << ")" << endl;

	return 0;
}
