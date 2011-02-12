
#include <stdio.h>
#include <vector>

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

#include "src/archipelago.h"
#include "src/island.h"
#include "src/population.h"
#include "src/algorithm/sga.h"
#include "src/problem/docking.h"
#include "src/ann_toolbox/multilayer_perceptron.h"
#include "src/ann_toolbox/perceptron.h"

using namespace std;
using namespace pagmo;

extern std::string max_log_string = "";
extern double  max_log_fitness = 99.0;
extern bool pre_evolve = false;

typedef float float_type;


int main(int argc, char *argv[])
{
    float start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
    char *config_filename;
    if ( argc >= 2 ) config_filename = argv[1];
    else	config_filename = NULL;	
	
    /////////////////////////////////
    // Definitions
    const int ann_input_neurons  = 7;
    const int ann_hidden_neurons = 11;
    const int ann_output_neurons = 2;
	
    int prob_positions = 1;
    int prob_pos_strategy = problem::docking<float_type>::SPOKE_POS;
    int prob_fitness_function = 99;
    int prob_timeneuron_threshold = 99;
    //int prob_maximum_time = 25;
    int prob_maximum_time = 25;

    float integrator_timestep = 0.1f;
    int evolution_stuck_threshold = 11;
	
    int algo_generations = 11;
    int algo_crossover = 90;
    int algo_mutation = 30;
    int algo_elitism = 1;
                                		
    int islands = 1;
    //int individuals = 10;
    int individuals = 33;

    float vicinity_distance = 0.1;
    float vicinity_speed = 0.1;
    float vicinity_orientation = 0.1;
	
	
    time_t seconds = time (NULL);
    std::string run_id = boost::lexical_cast<std::string>(seconds);
    //////////////////////////////////////////
	
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
	
    // MultiLayer Perceptron

    cuda::info inf;
//    std::cout<<inf;

    ann_toolbox::multilayer_perceptron
	<float, ann_input_neurons, ann_hidden_neurons, ann_output_neurons,  adhoc_dimensions<128>,  adhoc_dimensions<128> >  
	ann(inf,"multilayer perceptron",  individuals, prob_positions);

    problem::docking<float_type>::integrator integ(inf, "rk integrator", individuals, prob_positions);
    problem::docking<float_type>::fitness_type fitt(inf, "fitness evaluator",  //problem::docking<float_type>::fitness_type::cristos_twodee_fitness2,
						    problem::docking<float_type>::fitness_type::minimal_distance,
						    individuals, prob_positions, 6, 2, vicinity_distance, vicinity_speed, vicinity_orientation, prob_maximum_time);
    ////////////////////////////////////////////////
    // Define the problem						positions, strategy, 				max_time, max_thrust
    problem::docking<float_type> prob = problem::docking<float_type>(&ann, &integ, &fitt, inf, prob_positions, prob_pos_strategy, prob_maximum_time, 0.1);

    prob.set_start_condition(start_cnd, 6);	
    prob.set_log_genome(true);

    prob.set_fitness_function(prob_fitness_function);
    prob.set_timeneuron_threshold(prob_timeneuron_threshold/100.0);
	
    prob.set_time_step(integrator_timestep);
	
    prob.set_vicinity_distance(vicinity_distance);
    prob.set_vicinity_speed(vicinity_speed);
    prob.set_vicinity_orientation(vicinity_orientation);


    std::cout<<"creating algorithms"<<std::endl;
    algorithm::sga algo(algo_generations, 	// Generations
			algo_crossover/100.0,	// CR
			algo_mutation/100.0,	// Mutation	
			algo_elitism );/*, 	// Elitism
					 0.0,
					 2,	//random
					 0); 	// no roulette selection*/


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
    float best_fitness = 99.9;	// minimizing!!
    float last_fitness = 99.9;
	

    pre_evolve = true;		// generate random positions at first
    //run until we are quite good
    while(best_fitness > -1.7) 
    { 
	cout << "\r                                                          "
	     << "                                                            ";
	cout << "\rGeneration #" << i << " ["; cout.flush();
		
	max_log_fitness	= 0.0;		
	arch.evolve();
	arch.join();
	i++;

	//cout << "] best: " << arch.best().get_fitness() << ": " 
	//cout << best_fitness << ":" << last_fitness << "--" << i-lasti-1 << endl;
	fitness_vector maxfit;
	decision_vector maxdec;
	prob.fittest(maxfit, maxdec);
	
	cout << "] best: " << maxfit.size() << ": " << best_fitness << ":" << last_fitness << "--" << i-lasti-1 << endl;
			
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
//	    int ret = system (cmd.c_str());	
			
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
