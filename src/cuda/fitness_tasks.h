#ifndef __CUDA_PAGMO_FITNESS_TASKS__
#define __CUDA_PAGMO_FITNESS_TASKS__

#include "cudatask.h"
#include "kernel.h"


using namespace cuda;


namespace pagmo
{
    namespace fitness
    {
	template <typename ty, typename preprocessor>
	    class evaluate_fitness_task : public task <ty>
	{ 
	public:
	    enum fitness_type
	    {
		//Fitness types
		minimal_distance = 0,
		minimal_distance_speed_theta,
		minimal_distance_simple,
		no_attitude_fitness,
		cristos_twodee_fitness1,
		cristos_twodee_fitness2,
		cristos_twodee_fitness3
	    };

	evaluate_fitness_task(info & inf, fitness_type type , size_t individuals, 
			      size_t taskCount, ty vicinity_distance, 
			      ty vicinity_speed, ty max_docking_time ) : 
	    task<ty>::task(inf, individuals, taskCount, 1), m_fitness_type(type), //<TODO> not sure that the task size is 1
		m_inputs (6), m_outputs (3), 
		m_vicinity_distance(vicinity_distance), 
		m_vicinity_speed(vicinity_speed), 
		m_max_docking_time(max_docking_time),
		m_tdt(0)
		{}

	    enum
	    {
		param_states = 0,
		param_outputs = 1, 
		param_init_distance = 2
	    };

	    void set_time(ty t)
	    {
		m_tdt = t;
	    }


	    virtual bool set_initial_distance(int taskid, const std::vector<ty> & distance)
	    {
		if (distance.size() == 1)
		{
		    return task<ty>::set_inputs (taskid, param_init_distance, distance, 1);
		}
		return false;
	    }

	    virtual bool set_inputs(int taskid, const std::vector<ty> & inputs)
	    {
		if (inputs.size() == m_inputs)
		{
		    return task<ty>::set_inputs (taskid, param_states, inputs, m_inputs);
		}
		return false;
	    }

	    virtual bool get_outputs( int taskid, std::vector<ty> & outputs)
	    {
		return task<ty>::get_outputs (taskid, param_outputs, outputs);
	    }

	    virtual bool prepare_outputs()
	    {
		return task<ty>::prepare_dataset(param_outputs, m_outputs);
	    }

	    virtual bool launch() 
	    {

		dataset<ty> * pState = this->get_dataset(param_states);
		dataset<ty> * pOutData = this->get_dataset(param_outputs);
		dataset<ty> * pInitDistance = this->get_dataset(param_init_distance);

		if (!(pState && pOutData && pInitDistance))
		{
		    std::cout <<" Could not find a dataset"<<std::endl;
		    std::cout <<pState << " "<<pOutData << " "<<pInitDistance<<std::endl;
		    return false;
		}

		block_complete_dimensions dims(&this->m_info, this->get_profile());

		switch (m_fitness_type)
		{
		case  minimal_distance:
		    cu_compute_fitness_mindis<ty, preprocessor>(*pState->get_data(),*pOutData->get_data(), pOutData->get_task_size(), &dims); 
		    break;				
		    /*case  minimal_distance_speed_theta:
		      cu_compute_fitness_mindis_theta<ty, preprocessor>(*pState->get_data(),*pOutData->get_data(), width, g, b);
		      break;
		      case  minimal_distance_simple:
		      cu_compute_fitness_mindis_simple<ty, preprocessor>(*pState->get_data(),*pOutData->get_data(),*pInitDistance->get_data(), width, g, b);
		      break;
		      case no_attitude_fitness:
		      cu_compute_fitness_mindis_noatt<ty, preprocessor>(*pState->get_data() ,*pOutData->get_data() , *pInitDistance->get_data(), 
		      m_vicinity_distance, m_vicinity_speed,  m_max_docking_time, m_tdt, width, g, b);
		      break;
		      case cristos_twodee_fitness1:
		      cu_compute_fitness_twodee1<ty, preprocessor>(*pState->get_data(),*pOutData->get_data(),*pInitDistance->get_data(), 
		      m_max_docking_time, m_tdt, width, g, b);
		      break;
		      case  cristos_twodee_fitness2:
		      cu_compute_fitness_twodee2<ty, preprocessor>(*pState->get_data(),*pOutData->get_data(),*pInitDistance->get_data(), 
		      m_vicinity_distance,m_vicinity_speed, vic_orientation, m_max_docking_time, m_tdt, width, g, b);
		      break;*/
		    /*	  case cristos_twodee_fitness3:
		    //TODO orientation?
		    cu_compute_fitness_twodee3<ty, preprocessor>(*pState->get_data(),*pOutData->get_data(),*pInitDistance->get_data(), m_vicinity_distance,
		    m_vicinity_speed, vic_orientation, m_max_docking_time, m_tdt, width, g, b); 
		    break;*/
		default:
		    return false;
		};
		return true;
	    }
	protected:

	    size_t m_fitness_type;
	    unsigned int  m_inputs;
	    unsigned int  m_outputs;
	    ty m_vicinity_distance;
	    ty m_vicinity_speed;
	    ty m_max_docking_time;
	    ty m_tdt;

	};
    }
}


#endif
