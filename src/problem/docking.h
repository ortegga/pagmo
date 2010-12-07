#ifndef PAGMO_PROBLEM_DOCKING_H
#define PAGMO_PROBLEM_DOCKING_H

#include <string>
#include <vector>


#include "cuda_problem.h"

#include "../config.h"
#include "../population.h"
#include "../odeint/runge_kutta_4.h"
#include "../cuda/kernel.h"
#include "../cuda/fitness_tasks.h"
#include "../ann_toolbox/neural_network.h"


namespace pagmo 
{ 
    namespace problem 
    {	

	template <typename fty >
	    class docking : public cuda_problem{
	public:

	    typedef ann_toolbox::neural_network <fty, 7, 2 >  neural_network;
	    typedef hills_dynamical_system<fty, scale_functor<fty> > dynamic_system;
	    typedef odeint::ode_step_runge_kutta_4< fty, dynamic_system > integrator;
	    typedef fitness::evaluate_fitness_task<fty > fitness_type;


	docking(neural_network* ann_, 
		integrator * stepper,
		fitness_type * fitt, 
		cuda::info & inf_, 
		size_t random_positions,
		size_t in_pre_evo_strat = 1, fty max_time = 20, fty max_thr = 0.1) :
	    cuda_problem(ann_->get_number_of_weights()),
		ann(ann_),
		integrator_task(stepper),
		fitness_task(fitt),
		max_thrust(max_thr),
		max_docking_time(max_time),	
		inf(inf_),
		random_starting_positions(random_positions),
		pre_evolution_strategy(in_pre_evo_strat)
		{						
		    set_lb(	std::vector<double> (ann->get_number_of_weights(), -10.0) );
		    set_ub(	std::vector<double> (ann->get_number_of_weights(),  10.0) );
	  
		    this->random_start.clear();
		    log_genome = false;
		    this->time_neuron_threshold = .99;
		    this->needed_count_at_goal = 5;
		    this->vicinity_distance = vicinity_speed = 0.1;
		    this->vicinity_orientation = M_PI/8;	
		}
      
      
	    virtual base_ptr 	clone() const
	    { 
		return base_ptr(new docking<fty>(*this));
	    }
    
	    virtual std::string	id_object() const 
	    {
		return "Docking problem, using ANN to develop a robust controller"; 
	    }

	    void set_start_condition(size_t number) 
	    {
		if(number < random_start.size())
		    starting_condition = random_start[number];
		else
		    pagmo_throw(value_error, "wrong index for random start position");
	    }

	    void set_start_condition(fty *start_cnd, size_t size) 
	    {
		starting_condition = std::vector<fty> (start_cnd, start_cnd + size);
	    }

	    void set_start_condition(std::vector<fty> &start_cond) 
	    {
		starting_condition = start_cond;
	    }

	    void set_log_genome(bool b) 
	    {
		log_genome = b;
	    }

	    void set_timeneuron_threshold(fty t) 
	    {
		time_neuron_threshold = t;
	    }

	    void set_fitness_function(int f) 
	    {
		fitness_function = f;
	    }

	    void set_time_step(fty dt) 
	    {
		time_step = dt;
	    }

	    void set_vicinity_distance(fty d) 
	    {
		vicinity_distance = d;
	    }
	    void set_vicinity_speed(fty d) 
	    {
		vicinity_speed = d;
	    }

	    void set_vicinity_orientation(fty d) 
	    {
		vicinity_orientation = d;
	    }

	    void initialize_tasks()
	    {
		//Ann controls most of the data
		//integrator reuses the datasets for ann
		// fitness does the same
		ann->prepare_outputs();

		integrator_task->add_association(ann, neural_network::param_outputs, integrator::param_o);
		integrator_task->add_association(ann, neural_network::param_inputs, integrator::param_x);
		integrator_task->prepare_outputs();

		fitness_task->add_association(ann, neural_network::param_outputs, fitness_type::param_outputs);
		fitness_task->add_association(integrator_task, integrator::param_x, fitness_type::param_inputs);
		fitness_task->prepare_outputs();
		
		generate_starting_positions();
	    }

	    void objfun_impl(population & pop) const
	    {
		population::size_type size = pop.size();
		std::vector<fty> inputs;


		////////////////////////////////////////////////////////////////////////////
		//Load inputs
		for (population::size_type s = 0; s < size; ++s)
		{
		    population::individual_type indiv = pop.get_individual(s);
		    decision_vector wi = indiv.cur_x;

		    //necessary if we're working with floats while pagmo works with doubles
		    std::vector<fty> wv;
		    wv.insert(wv.begin(),wi.begin(),wi.end());
		    ann->set_weights(s,wv);

		    for(size_t i = 0;i < random_start.size();i++) 
		    {	
			inputs = random_start[i];
			if (ann->get_number_of_inputs() == 7)
			{
			    fty initial_distance = inputs[0] * inputs[0] + inputs[1] * inputs[1];
			    if (inputs.size() != ann->get_number_of_inputs())
				inputs.push_back(initial_distance);
			    fitness_task->set_initial_distance(s, i, initial_distance);
			}
			if(!ann->set_inputs(s,i,inputs))
			{
	    		    CUDA_LOG_ERR("docking"," failed to set inputs ",inputs.size());
			    return ;
			}
		    }
		}
		if (!integrator_task->execute_associations())
		{
		    CUDA_LOG_ERR("docking", " failed to complete docking run ",0);
		    return;
		}
		if (!fitness_task->execute_associations())
		{
		    CUDA_LOG_ERR("docking", " failed to complete docking run ",0);
		    return;
		}
		////////////////////////////////////////////////////////////////////////////////
		//Execute the kernels

		for(fty t = 0.0;t < max_docking_time ;t += time_step) 
		{
		    std::cout<<std::endl<<"t="<<t<<std::endl;
		    if(!ann->launch())
		    {
			return ;
		    }
		    integrator_task->set_params(t, time_step, max_thrust);
		    if(!integrator_task->launch())
		    {
			return;
		    }
		    fitness_task->set_time(t+time_step);
		    if (!fitness_task->launch())
		    {
			return;
		    }
		    std::cout<<"-----"<<std::endl;
		}
		
		/////////////////////////////////////////////////////////////////////////
		//Extract new x and new fitnesses

		bool first = true;
		for (population::size_type s = 0; s < size; ++s)
		{
		    population::individual_type indiv = pop.get_individual(s);

		    //necessary if we're working with floats while pagmo works with doubles
		    std::vector<fty> wv;
		    ann->get_weights(s,wv);
		    indiv.cur_x.clear();
		    indiv.cur_x.insert(indiv.cur_x.begin(),wv.begin(),wv.end());

		    std::vector<fty> out;
		    fty result = 0; 
		    std::cout<<"fty result = 0"<<std::endl;
		    for(size_t i = 0;i < random_start.size();i++) 
		    {	
			out.clear();
			if(!fitness_task->get_fitness(s, i, out))
			{
			    std::cout<<"failed to retrieve fitness results"<<std::endl;
			    return;
			}
			std::cout<<"out"<<out.size()<<std::endl;
			result += out[0];
		    }
		    result /= random_start.size();
		    indiv.cur_f[0] = result;
		    if ( first || base::compare_fitness(indiv.cur_f, max_fit)) {
			max_fit = indiv.cur_f;
			max_dec = indiv.cur_x;
			first = false;
		    }
		}
		std::cout<<max_fit<<std::endl;
		//std::string logg;
		//f[0] = all_run(logg);	
		//std::cout<<"Fitness value: "<<f[0]<<std::endl;
	    }	    

	    /// Generate starting positions for the run of the individuals.
	    void generate_starting_positions() const
	    {
		if (random_start.empty())
		{
		    if(random_starting_positions >= 1) 
		    {
			fty cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
			random_start.push_back(std::vector<fty> (cnd, cnd + ann->get_number_of_inputs()));
		    }

		    if(random_starting_positions >= 2) 
		    {
			fty cnd[] = { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
			random_start.push_back(std::vector<fty> (cnd, cnd + ann->get_number_of_inputs()));
		    }
		
		    if(random_starting_positions >= 3) 
		    {
			fty cnd[] = { -1.0, 0.0, -1.0, 0.0, 0.0, 0.0 };		
			random_start.push_back(std::vector<fty> (cnd, cnd + ann->get_number_of_inputs()));
		    }
		}
	    }
				
	    /// Constants for 
	    enum 
	    {
		FIXED_POS = 1,
		SPOKE_POS = 2,
		RAND_POS  = 3,
		CLOUD_POS = 4,
		SPOKE_POS_HALF 	= 20,
		DONUT_FACING = 33,
		FULL_GRID = 99,
		SPOKE_8_POS 	= 200
	    };
		
	private:


	    mutable std::vector< std::vector<fty> > random_start;
	    mutable std::vector<fty>	starting_condition;

	    // Reference to the neural network representation
	    mutable neural_network *ann;
	    mutable integrator * integrator_task;
	    mutable fitness_type * fitness_task;
		
	    // Variables/Constants for the ODE
	    fty nu, max_thrust, mR, max_docking_time;
	    fty time_neuron_threshold;
		
	    // control variables
	    bool log_genome;					// is the genome logged in the log string 
	    size_t needed_count_at_goal;		// how long does the s/c need to stay within the target area before the optimization stops
	    size_t random_starting_positions;	// how many random starting positions exist/need to be generated
	    size_t pre_evolution_strategy;		// which strategy for the generation of the random numbers is used
	    size_t fitness_function;			// how to calculate the fitness
	
	    fty vicinity_distance;			// the size of the vicinity around the origin that we take as close enough
	    fty vicinity_speed;				// the maximum speed around the origin that we take as small enough
	    fty vicinity_orientation;		// the needed orientation around the origin that we take as good enough
		
	    fty time_step; 					// for integrator		
	    cuda::info & inf;
	};	
    }
}
#endif
