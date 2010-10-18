#ifndef PAGMO_PROBLEM_DOCKING_H
#define PAGMO_PROBLEM_DOCKING_H

#include <string>
#include <vector>


#include "base.h"
//#include "docking_positions.h"

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

    template <typename float_type>
      class __PAGMO_VISIBLE docking : public base {
    public:

      typedef hills_dynamical_system<float_type, scale_functor<float_type> > dynamic_system;
      typedef fitness::evaluate_fitness_task<float_type, nop_functor<float_type> > fitness_task_type;
      typedef odeint::ode_step_runge_kutta_4< float_type, dynamic_system > integrator;
      typedef ann_toolbox::neural_network <float_type, 7, 2>  neural_network;

    docking(neural_network* ann_, 
	    integrator * stepper,
	    cuda::info & inf_, 
	    size_t random_positions, 
	    size_t in_pre_evo_strat = 1, float_type max_time = 20, 
	    float_type max_thr = 0.1) :
      base(ann_->get_number_of_weights()),
	ann(ann_),
	integrator_task(stepper),
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
	return base_ptr(new docking<float_type>(*this));
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

      void set_start_condition(float_type *start_cnd, size_t size) 
      {
	starting_condition = std::vector<float_type> (start_cnd, start_cnd + size);
      }

      void set_start_condition(std::vector<float_type> &start_cond) 
      {
	starting_condition = start_cond;
      }

      void set_log_genome(bool b) 
      {
	log_genome = b;
      }

      void set_timeneuron_threshold(float_type t) 
      {
	time_neuron_threshold = t;
      }

      void set_fitness_function(int f) 
      {
	fitness_function = f;
      }

      void set_time_step(float_type dt) 
      {
	time_step = dt;
      }

      void set_vicinity_distance(float_type d) 
      {
	vicinity_distance = d;
      }
      void set_vicinity_speed(float_type d) 
      {
	vicinity_speed = d;
      }

      void set_vicinity_orientation(float_type d) 
      {
	vicinity_orientation = d;
      }

      void objfun_impl(fitness_vector &f, const decision_vector &x) const
      {
	generate_starting_positions();
	static int cnt = 0;
	if(x.size() != ann->get_number_of_weights()) {
	  pagmo_throw(value_error, "wrong number of weights in the chromosome");
	}
	//float_type average = 0.0;	
	size_t i;

	// Prepare output vectors
	ann->prepare_outputs();
	std::vector<float_type> v;
	v.insert(v.begin(),x.begin(),x.end());
      
	for(i = 0;i < random_start.size();i++) 
	  {		
	    ann->set_weights(i,v);
	    starting_condition = random_start[i];
	    ann->set_inputs(i,starting_condition);
	  }
	std::string logg;
	f[0] = one_run(logg);	
	std::cout<<"Fitness value: "<<f[0]<<std::endl;
      }


      /// Generate starting positions for the run of the individuals.
      void generate_starting_positions() const
      {
	if (random_start.empty())
	  {
	    if(random_starting_positions >= 1) 
	      {
		float_type cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
		random_start.push_back(std::vector<float_type> (cnd, cnd + ann->get_number_of_inputs()));
	      }

	    if(random_starting_positions >= 2) 
	      {
		float_type cnd[] = { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
		random_start.push_back(std::vector<float_type> (cnd, cnd + ann->get_number_of_inputs()));
	      }
		
	    if(random_starting_positions >= 3) 
	      {
		float_type cnd[] = { -1.0, 0.0, -1.0, 0.0, 0.0, 0.0 };		
		random_start.push_back(std::vector<float_type> (cnd, cnd + ann->get_number_of_inputs()));
	      }
	  }
      }
				
      /// Constants for 
      enum {
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

      float_type one_run(std::string &log) const 
      {	
	float_type retval = 0.0;
	float_type dt = time_step, t = 0.0;
	size_t size = random_start.size();
	std::cout <<" random starts " << size<< std::endl;

	fitness_task_type fitness_task (inf, fitness_task_type::minimal_distance , 
					ann->get_number_of_outputs(),  vicinity_distance,  
					vicinity_speed, max_docking_time );
	fitness_task.prepare_outputs();
	std::cout <<"docking problem configuration" <<max_docking_time<<std::endl;

	for(t = 0.0;t < max_docking_time ;t += dt) 
	  {
	    std::cout << "increment: "<<t <<std::endl;
	    if (!ann->launch())
	      {
		//Log an error
		std::cout <<"NNet launch fail" <<std::endl;
		return false;
	      }

	    //Scale inputs
	    std::vector<float_type> nnetouts;

	    //Integrator / dynamic system
	    integrator_task->set_params(t, dt, max_thrust);
	    for (unsigned int k = 0; k < size; ++k)
	      {
		//Possibly unnecessary steps here
		ann->get_outputs(k, nnetouts);
		if(nnetouts.size() > 2) nnetouts.pop_back();	//delete last
		integrator_task->set_dynamical_inputs(k, nnetouts); 
		std::vector<float_type> tmp = random_start[k];
		tmp.pop_back();
		if (!integrator_task->set_inputs(k, tmp))
		  {
		    std::cout <<"size mismatch ["<<tmp.size()<<"/"<<size<<"]"<<std::endl;
		  }
	      }
	    integrator_task->launch();
	    // evaluate the fitness

	    fitness_task.set_time(t+dt);
	    for (unsigned int k = 0; k < size; ++ k)
	      {
		integrator_task->get_outputs(k, nnetouts);
		fitness_task.set_inputs(k, nnetouts);
		std::vector<float_type> tmp = random_start[k];
		std::vector<float_type> initdis;
		initdis.push_back(sqrt(tmp[0]*tmp[0] + tmp[2] * tmp [2]));
		fitness_task.set_initial_distance(k, initdis);
	      }
	    fitness_task.launch();
	  
	    for (unsigned int k = 0; k < size; ++k)
	      {
		fitness_task.get_outputs(k, nnetouts);
		retval 	+= nnetouts[0];
	      }
	    std::cout << "End increment: "<< (-retval) <<std::endl;
	  }
	return -retval / size;
      }

      mutable std::vector< std::vector<float_type> > random_start;
      mutable std::vector<float_type>	starting_condition;

      // Reference to the neural network representation
      mutable neural_network *ann;
      mutable integrator * integrator_task;
		
      // Variables/Constants for the ODE
      float_type nu, max_thrust, mR, max_docking_time;
      float_type time_neuron_threshold;
		
      // control variables
      bool log_genome;					// is the genome logged in the log string 
      size_t needed_count_at_goal;		// how long does the s/c need to stay within the target area before the optimization stops
      size_t random_starting_positions;	// how many random starting positions exist/need to be generated
      size_t pre_evolution_strategy;		// which strategy for the generation of the random numbers is used
      size_t fitness_function;			// how to calculate the fitness
	
      float_type vicinity_distance;			// the size of the vicinity around the origin that we take as close enough
      float_type vicinity_speed;				// the maximum speed around the origin that we take as small enough
      float_type vicinity_orientation;		// the needed orientation around the origin that we take as good enough
		
      float_type time_step; 					// for integrator		
      cuda::info & inf;
    };	

    /*    template <>
      class docking<float>;

    template <>
    class docking<double>;*/
  }
}
#endif
