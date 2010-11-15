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

#include "src/GOclasses/basic/migration/MigrationScheme.h"
#include "src/GOclasses/basic/migration/MigrationPolicy.h"
#include "src/GOclasses/basic/migration/ChooseBestMigrationSelectionPolicy.h"
#include "src/GOclasses/basic/topology/ring_topology.h"

#include "src/ann_toolbox/multilayer_perceptron.h"
#include "src/ann_toolbox/ctrnn.h"

#include "src/logger.h"
#include "src/config_toolbox/config_parser.h"

#include <numeric>

using namespace std;
using namespace pagmo;


my_logger logging;

extern string max_log_string = "";
extern double max_log_fitness = 99.0;
extern bool pre_evolve = false;


void usage(const char* name) {
	printf("PaGMO Evolving Docking Simulator\n"
		"usage: %s <config-file> <options>\n"
			"with options:\n"
			"  --load <file>    load a chromosome from <file> and perform a docking simulation run\n"
			"  --load-oc <file> load a <file> containing the optimal control strategy\n"
			"  --noise <val>    set the noise in the integrator to ]-val, val[\n"
			"  --help           display this help text\n"
		"\n",
		name
	);
	exit(0);
}

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

void run_chromosome(std::string file, problem::docking &prob) {
	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
	//	double start_cnd[] = { -1.615471, 0.0, 0.438620, 0.0, 0.0, 0.0 };	
	//  ann_toolbox::multilayer_perceptron ann(6, 10, 3);
	std::vector<double> v = load_chromosome_twodee(file.c_str());  // "chromosome.log"

	// problem::docking prob = problem::docking(&ann, 8, problem::docking::FIXED_POS, 25, 0.1);
	// prob.set_start_condition(start_cnd, 6);	
	// prob.set_timeneuron_threshold(.99);
	// prob.set_log_genome(true);
	// prob.set_fitness_function(99);	// 10  = no attitude

	// TESTING
	cout << "Created the problem!" << endl;	
	cout << "Calling the objfun_ ... " << endl;

	max_log_fitness = 0.0;
	
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

}


void run_optimal_control(std::string fname, problem::docking &prob) {
	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
	double start_cnd[] = { -2.0, 0.1, 2.0, 0.1, 0.0, 0.0 };
//	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };	
	
	// read oc file
	// the order is, :: time x vx z vz theta omega uL uR
	fstream file;
	file.open(fname.c_str(), ios::in);
	if ( file.fail() ) {
	    printf("Error opening %s - %s\n", fname.c_str(), strerror(errno));
		exit(0);
	}
	char h[512];
	std::string line;
	double dbl[9]; // elements in line
	std::vector<double> time, ul, ur;
	// add for
	while(! file.eof()) {
		//file.getline(h, 512);
		//line = h;
		//boost::algorithm::trim(line);
		//if((! line.empty()))) {
		//	read_lines.push_back(line);
		//}
	    for (int i = 0; i < 9; i++) {
	        file >> dbl[i];
	    }
		time.push_back(dbl[0]);
		ul.push_back(dbl[7]);
		ur.push_back(dbl[8]);
	}
    file.close();

	double sum_of_time_diffs = 0.0;
	double prev = time[0];
	std::vector<double>::iterator j = time.begin();
	for(j++; j != time.end(); ++j) {
		sum_of_time_diffs += (*j) - prev;
		prev = *j;
	}

//	cout << time.pop_back() << std::endl;
	time.pop_back(); ul.pop_back(); ur.pop_back();

	double integrator_timestep = sum_of_time_diffs / (time.size());
	
	cout << "TIME STEP --> " << integrator_timestep << "("<< time.size() << ")" << std::endl << std::endl;

	fprintf(stderr, "Control loaded from %s (%f, %f, %f)\n", fname.c_str(), time.back(), ul.back(), ur.back());
	
	// redo timeing!!! 
	// TODOD!!
	
	// TESTING
	cout << "Created the problem!" << endl;	
	cout << "Calling the objfun_ ... " << endl;

	max_log_fitness = 0.0;

	prob.set_time_step(integrator_timestep);
	prob.set_vicinity_speed(prob.get_vicinity_speed() + 0.05);
	prob.set_vicinity_distance(prob.get_vicinity_distance() + 0.17);
	
	prob.set_start_condition(start_cnd, 6);
//	max_log_fitness = prob.objfun_oc(time, ul, ur);
	max_log_fitness = prob.one_run_oc(max_log_string, ul, ur);	
	cout << "\r=== OC fitness: " << max_log_fitness << endl;	
	cout << max_log_string << endl;	

	ofstream myfile;
	myfile.open ("bestrun-oc.dat");
	myfile << max_log_string << endl;
	myfile.close();	
	max_log_fitness = 0;

}


int main(int argc, char *argv[]){
	// for evaluating twodee chromosome
	// evaluate_twodee();

	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
	double max_noise = 0.0;
	
	bool less_output = false;
	bool evaluate_chromosome = false, run_oc = false;
	char *config_filename;
	std::string filename;	// for laoding stuff
	
		
	///////////////////////////////////////
	// Parse command line arguments
 	if ( argc >= 2){
		for (int i = 1; i < argc; i++) {
			// if any parameter is help we show the help and stop
			if (strcmp(argv[i], "--help") == 0) usage(argv[0]);	
			
			// first parameter should be config file
			if (i == 1) config_filename = argv[i];
			// less switch to output less
			else if (strcmp(argv[i], "--less") == 0) less_output = true; 
			// load switch to load chromosome from a file
			else if (strcmp(argv[i], "--load") == 0) {
				if(run_oc) {
					cout << "ERROR: can not combine --load with --load-oc"<< endl;
					exit(0);
				}
				if(argc > i + 1) { 
					evaluate_chromosome = true;
					filename = argv[++i];
					//cout << "T:" << i << ", " << filename << ", " << argc << endl;
				} else { cout << "--load: missing a filename to load!" << endl; usage(argv[0]); }
			}
			// load-oc switch to load the AMPL oc from a file
			else if (strcmp(argv[i], "--load-oc") == 0) {
				if(evaluate_chromosome) {
					cout << "ERROR: can not combine --load with --load-oc"<< endl;
					exit(0);
				}
				if(argc > i + 1) { 
					run_oc = true;
					filename = argv[++i];
				} else { cout << "--load-oc: missing a filename to load!" << endl; usage(argv[0]); }
			}
			// change the noise level in the integrator
			else if(strcmp(argv[i], "--noise") == 0) {
				if(argc > i + 1) { 
					max_noise = boost::lexical_cast<double>(argv[++i]);
				} else { cout << "--noise: missing a noise value!" << endl; usage(argv[0]); }
			}
			// the rest show warning!
			else { printf("warning: unknown command line option '%s'\n", argv[i]); usage(argv[0]); }
		}
	} else config_filename = NULL;	
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
	int evolution_stuck_threshold = 500;config.read_parameter("STUCK_THRESHOLD", evolution_stuck_threshold);
	
	int algo_generations = 11;			config.read_parameter("GENERATIONS", algo_generations);
	int algo_crossover = 90;    		config.read_parameter("CROSSOVER", algo_crossover);
	int algo_mutation = 30;     		config.read_parameter("MUTATION_RATE", algo_mutation);
	int algo_elitism = 1;				config.read_parameter("ELITISM", algo_elitism);
                                		
	int islands = 1;					config.read_parameter("ISLANDS", islands);
	int individuals = 33;				config.read_parameter("INDIVIDUALS", individuals);

	double vicinity_distance = 0.1;		config.read_parameter("VICINITY_DISTANCE", vicinity_distance);
	double vicinity_speed = 0.1;		config.read_parameter("VICINITY_SPEED", vicinity_speed);
	double vicinity_orientation = 0.1;	config.read_parameter("VICINITY_ORIENTATION", vicinity_orientation);	
	
	bool upload_enabled = true;			config.read_parameter("ENABLE_UPLOAD", (int&)upload_enabled);	
	
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
	configs += "Maximum Time:\t\t" + boost::lexical_cast<std::string>(prob_maximum_time) + "\n";

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
	
	prob.set_max_noise(max_noise);
	
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



	if(evaluate_chromosome) {
		run_chromosome(filename, prob);
		exit(0);
	}
	
	if(run_oc) {
		run_optimal_control(filename, prob);
		exit(0);
	}
						
//	algorithm::de algo(20, 0.7, 0.5, 2);
	
	////////////////////////////////////////////////
	// Create the archipelag/islands
	cout << "Creating an archipelago...";
	ring_topology top;
	MigrationScheme migS(top);
//	MigrationPolicy migP;
					//probability, one individual per island, up to 100% of the population can be replaced (1.0)
	MigrationPolicy migP(0.2, ChooseBestMigrationSelectionPolicy(1), RandomMigrationReplacementPolicy(1.0));	
	Migration m(migS, migP);
	archipelago arch = archipelago(prob, algo, islands, individuals, m);
	cout << "Created!";


	///////////////////////////////////////////////
	// Start evolution
	cout << "\rStarting evolution ...          ID: " << run_id << "          " << endl;	
	
	ofstream myfile;
	int i = 0, lasti = 0;
	double best_fitness = 99.9;	// minimizing!!
	double last_fitness = 99.9;
	

	max_log_fitness	= 0.0;		
		
	pre_evolve = false;		// do NOT generate random positions at first
	// run until we are quite good
	while(best_fitness > -1.7 && i < 49999) { 
		if(!less_output) {
			cout << "\r                                                      "
			     << "                                                    ";
			cout << "\rGeneration #" << i << " ["; cout.flush();
		}

		max_log_fitness	= 0.0;		
		arch.evolve();
		arch.join();
		i++;

		
		if(!less_output) {
			cout << "] best: " << arch.best().get_fitness() << "~" << logging.best_value() << ": " << best_fitness << ":" << last_fitness << "--" << i-lasti-1 << endl;
			// << "/" << arch[0].get_average_fitnes()?; // << "  lasti: " << (i-lasti);
			cout << "TEST: #ofIslands: " << arch.size() << " #0: size" << arch[0].size() << " best: " << arch[0].best().get_fitness() << endl;
			cout << "TEST: Island" << arch[0] << endl;
		}else {
			cout << "\rG#" << i << "        "; cout.flush();
		}
		
		
		//////////////////////////////////////////
		// logging
		if(logging.best_value() < best_fitness) {	// max_log_fitness
			best_fitness = logging.best_value();	
			cout << "\r=== Best increased @ #" << i-1 << ": " << best_fitness << endl;
			
			// write to file
			std::string h = "id_" + run_id + "_bestrun.dat";
			myfile.open (h.c_str());
			myfile << configs << endl;//expected
			myfile << logging << endl;
			myfile.close();	
			lasti = i-1;
						
			// save good ones
			//good_ones.push_back(arch.best());
		}
		if(logging.best_value() < last_fitness) {	
			last_fitness = logging.best_value();
			std::string h = boost::lexical_cast<std::string>(i-1);
			while(h.length() < 5) h = "0" + h;
			std::string s = "id_" + run_id + "_genbest-" + h + ".dat";
			myfile.open (s.c_str());
			myfile << configs << endl;
			myfile << logging << endl;
			myfile.close();	
			//////////////////////////////////////////
			
			lasti = i-1;
			
			if (upload_enabled) {
				// try to put it online!
				std::string cmd = "curl -H \"Expect: \" -F \"file=@";
				cmd += s += "\" -F \"submit=submit\" -F \"hostname=`hostname`\" http://juxi.net/projects/EvolvingDocking/upload.php";
				cmd += " -o tmp.out";
			
				int ret = system (cmd.c_str());	
			}
			
		}
		cout.flush();		

		// randomize positions if we seem to be stuck
/*		if( (i - 1 - lasti) >= evolution_stuck_threshold ) {
			pre_evolve = true;
			lasti = i - 1;
			last_fitness = 0.0;
		}*/
	}	
	
	// finished
	cout << "==================== Best Overall: " << best_fitness << "\t(i=" << i << ")" << endl;	

	return 0;
}
