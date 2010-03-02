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
extern double max_log_fitness = 0.0;


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

int main(){
	double best_fitness = 1;	// minimizing!!

	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	
/*	// previous run // MultiLayerPerceptron
	
	//ann_toolbox::multilayer_perceptron ann(6, 13, 3);

	// CTRNN, 6 inputs, 10 hidden, 3 outputs
	ann_toolbox::ctrnn ann(6, 5, 3);
	ann.set_weights_lower_bound(-1.0);
	ann.set_weights_upper_bound( 1.0);	
	ann.set_bias_lower_bound(-1.0);
	ann.set_bias_upper_bound( 1.0);
	ann.set_tau_lower_bound(-1.0);	// from Christo's calculations
	ann.set_tau_upper_bound( 2.0);// my try
	

	//                                      	 positions, max_time, max_thrust
	problem::docking prob = problem::docking(&ann, 10, 20, 0.1);
	prob.set_start_condition(start_cnd, 6);
	prob.set_timeneuron_threshold(.95);
	prob.set_log_genome(true);

*/
	
	// Starting Conditions:  x,  vx,  y,   vy,theta,omega
//	double start_cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
/*	ann_toolbox::ctrnn ann(7, 10, 3);
	ann.set_weights_lower_bound(-10.0);
	ann.set_weights_upper_bound( 10.0);	
	ann.set_bias_lower_bound(-10.0);
	ann.set_bias_upper_bound( 10.0);
	ann.set_tau_lower_bound(-1.0);	// from Christo's calculations -1, 2.4
	ann.set_tau_upper_bound( 2.4);//
	
	double values[] = { 1.77348654585157605723, -2.15534058287446761426, -7.46429396927202315482, 6.35235258563676730148, 2.62178280970238386516, -5.82415515885258994189, -9.56716586723012341054, -10.00000000000000000000, 6.52265921961741046431, 8.43496117225652497496, -5.18393356492627876975, 1.99633444804050208532, 5.12377435579894235929, 7.51729002853047223454, -2.56569432493118965155, -0.11589706993630055631, 3.20067258311369062795, -2.49624249991262070481, 5.51296413719898747985, -4.35529432946614925015, 1.09965530120681953541, -7.53682914541451332013, 9.49318619006159991613, 10.00000000000000000000, 2.65215599186624828576, 8.18156530416435145980,
		2.65479466506654970814, -3.41795643549033734132, -3.26240740784545835140, -1.10403492177251716377, 0.42710388317434500038, -6.13287161375195832846, -9.40949768055042490289, 2.61339616205674296623, 2.67552341359323975212, -3.24967832852978100178, -0.03364285980529091447, -10.00000000000000000000, 9.60085189933925775563, 1.93972281986203665127, -7.48050394731502166223, 0.33957360846045858693, 8.96283154549937322031, -6.38635585736334387974, 8.36180467857472109472, -7.89602978631938334786, -7.93803650141234395932, 7.17515499490383579229, 9.94253844314291157502, 2.35321763106928294462, -6.57831048490049408173, 3.01182936726151151419, -9.16342742572635415854,
		9.86009308968574060827, 1.97035225207854014506, -9.37203829095588325515, -9.93027869426819087550, 2.42792071773500062193, -9.32225954494013997476, -9.88357339304244497669, -4.43837334157747775976, -1.49625375886967137973, -1.76172311373508594379, 5.25153799903402163096, -0.35509015214087541468, -8.96936314147959734555, 0.74192639104965918406, 8.63546625574836390626, -6.66118140383211265743, 3.42053559519670402622, 0.90723347322944158933, -10.00000000000000000000, -7.13950693640990330380, -6.47887554267428456001, 0.73633505206764660045, 5.22490210365019613903, -4.80283667027732263932, 4.85062043783160312671, 8.53320715572799137760, 9.73026868182836146559,
		-9.48097186596839591743, 4.91981397251320551334, -3.31429215496234652605, -3.35085847453116070227, 10.00000000000000000000, -0.58648501184484769233, -1.95218769974546768609, -5.54259759664241702382, 5.57416314509549071232, -6.14456709092262531868, -3.24934003061534637524, -3.23403320523136805775, -4.60910323845907576867, 8.17315965817054745912, 7.76909160122076158927, -7.64580732620715330938, -5.03315126373574539542, 0.73102266436047225184, -0.44322586448246947821, 1.78547737464080347181, 8.69238291586533762256, 1.96030725910767777975, -1.29703118142863393913, -0.88591333900215141206, -9.70839718884707814084, 6.11845587919173095059, 9.01672939185922039940,
		-10.00000000000000000000, 3.53046273582405323310, 0.48539857001714092899, 0.40247351504784423248, -1.30634099617724208642, 3.18501375814325449198, -8.37544938993101695246, -10.00000000000000000000, -4.66737153553889161373, 5.29746566408449570673, -4.63394259419013287982, -8.07347051045102404032, -10.00000000000000000000, 1.48653775927158426917, 5.34747818033508170998, 3.01187685431711082984, -0.35328543926630817662, -8.77378422059963902768, -8.41401867734951203204, 2.10699833348646370368, -7.71738252426889914659, -5.45603169070159754739, 1.18505315123450838399, -1.77378653027699351163, -2.23516432054870195500, 4.04334444041495011390, 6.28199450823707827851,
		-10.00000000000000000000, -10.00000000000000000000, 1.57463800761211114576, -10.00000000000000000000, -2.46039970097433435825, 7.73087157231923871592, -4.83567829461540910074, 8.97097054941689364682, 1.45607113190537296177, 5.28916369986834844497, -9.31243262688221129508, -9.38928869416600164755, 2.54350200721240327084, -10.00000000000000000000, -9.45669521489501363476, -6.94198124056382148694, -3.27112379950024845243, -4.49756769002140011082, 1.94606299681752492603, 4.15334242521274887849, -0.93581310362095004862, -3.81374169884754987692, -4.94652956407940536110, -3.80930390820916553452, 0.00267481555460480358, 1.72391792995429016244, -9.28445089333195738845,
		2.87428412792648346752, 7.84091897128721093679, 3.65060187312241879454, -5.68709116715559215294, 3.72724895681752066423, -7.21275669579252820540, 1.05250104727553384087, -10.00000000000000000000, 4.71345273838410872003, 8.37210761028372729697, 2.99222623550913935375, 6.46896368267613031122, -10.00000000000000000000, 2.54269313490240245201, 0.62989373263483350307, -8.00409989531443244459, 1.48713387525646556497, 3.60452577185338274290, 3.17313542330691866766, -9.44438256419828547905, -0.94906250685679527379, 8.91843183764445868178, -6.48417596953575348095, -6.26633176354223753179, -8.79409534045155538706, 3.43145892467087598732, -5.83764128811508165029,
		-8.49449951097679800682, 1.68107134516610612351, 1.10903077569647590117, 2.59927458173359227089, -6.17508307123625055368, 2.11141210651194333181, 6.51167588917347828215, -9.86182929745702097080, 2.81878375813575399178, 5.42766900392749906956, 3.27439224392200811664, 2.89049334001997548782, 4.53904549987180416792, 1.95341898666221602809, -4.15006894334019094117, -6.01819754331654355184, 3.78851017368936382468, 9.75717671794660290630, -7.40172410412432668636, -5.39278027706714802036, 9.77640137557894739473, 2.51107113748860522051, 8.23439494265433502562, -6.71275167598844824113, -2.25147281846072777611, 8.13631442057296716541, -6.47443677427188291063, 
		7.15658404159856154081, -2.35865202052897915408, -6.05528006556824216489, -1.46558175058077932995, -8.19074651305777479138, 1.99154352110351196004, -3.78132075218926066995, -0.82214304389817760566 };
	std::vector<double> v =  std::vector<double> (values, values + 223);
*/	

//	double start_cnd[] = { -1.615471, 0.0, 0.438620, 0.0, 0.0, 0.0 };	
//	ann_toolbox::multilayer_perceptron ann(7, 50, 3);
//	std::vector<double> v = load_chromosome_twodee("best.log");
/* 	TESTING
	cout << "Created the problem!" << endl;	
	cout << "Calling the objfun_ ... " << endl;
	max_log_fitness = 0;
	max_log_fitness = prob.objfun_(v);
	cout << "\r=== Best increased @#0" << ": " << max_log_fitness << endl;	
	cout << max_log_string << endl;	
*/
	


	ann_toolbox::multilayer_perceptron ann(4, 10, 2);

	//											 positions, strategy, max_time, max_thrust
	problem::docking prob = problem::docking(&ann, 9, problem::docking::SPOKE_POS/*FIXED_POS*/, 20, 0.1);
	prob.set_start_condition(start_cnd, 6);	
//	prob.set_timeneuron_threshold(.95);
	prob.set_log_genome(true);
	prob.set_fitness_function(10);	// 10  = no attitude
	
	algorithm::sga algo( 20, 	// Generations
				0.9,	// CR		the value drng has to be above to stop
				0.2,	// Mutation	
				1);		// Elitism
/*	algo.set_selection_type(0);		// no roulette selection (keep best!)*/

/*	algorithm::sga algo( 20, 	// Generations
						0.5,	// CR
						0.15,	// Mutation	
						1, 	// Elitism
						0.0,
						2,	//random
						0); 	// no roulette selection
						*/
						
//	algorithm::de algo(20, 0.7, 0.5, 2);

	cout << "Creating an archipelago...";
	archipelago arch = archipelago(prob, algo, 1, 20);	// 1 island, with x individuals
	cout << "Created!";


	ofstream myfile;
	cout << "\rStarting evolution ...                        " << endl;	
	int i = 0;
	while(best_fitness > -1.2/*i++ < 6/**/) { 
		arch.evolve();
		arch.join();
		i++;
		cout << "\rGeneration #" << i << ": best: " << arch.best().get_fitness() << "                     " << endl;
		if(max_log_fitness < best_fitness) {
			best_fitness = max_log_fitness;	
			cout << "\r=== Best increased @ #" << i << ": " << max_log_fitness << endl;
			myfile.open ("bestrun.dat");
			myfile << max_log_string << endl;
			myfile.close();		
		}
		cout.flush();		
	}	
	cout << "==================== Best Overall: " << best_fitness << "\t(i=" << i << ")" << endl;

	return 0;


//	cout << arch.best().get_decision_vector();
//	std::copy(arch.best().get_decision_vector().begin(), arch.best().get_decision_vector().end(), std::ostream_iterator<double>(cout, ", "));	


/*		std::string s = "gen-" + boost::lexical_cast<std::string>(i) + ".dat";
		myfile.open (s.c_str());
		myfile << max_log_string << endl;
		myfile.close();		*/

	
	
}
