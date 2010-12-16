#ifndef __CUDA_PAGMO_FITNESS_TASKS__
#define __CUDA_PAGMO_FITNESS_TASKS__

#include "cudatask.h"
#include "kernel.h"
#include "kernel_dims.h"


using namespace cuda;


namespace pagmo
{
    namespace fitness
    {
	template <typename ty, typename pre_exec = nop_functor<ty> , typename post_exec = nop_functor<ty> >
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

	evaluate_fitness_task(info & inf, const std::string & name, fitness_type type , size_t individuals, 
			      size_t taskCount, ty vicinity_distance, 
			      ty vicinity_speed, ty max_docking_time ) : 
	task<ty>::task(inf, name, individuals, taskCount, 1), m_fitness_type(type), //<TODO> not sure that the task size is 1
		m_inputs (6), m_outputs (3), m_fitness(4), 
		m_vicinity_distance(vicinity_distance), 
		m_vicinity_speed(vicinity_speed), 
		m_max_docking_time(max_docking_time),
		m_tdt(0)
		{
		    
		    this->set_shared_chunk(0, 0 , m_inputs + m_outputs);
		    this->set_global_chunk(0, 0 , m_inputs + m_outputs + m_fitness);
		    this->m_dims = kernel_dimensions::ptr( new block_complete_dimensions (&this->m_info, this->get_profile(), this->m_name));	    
		}

	    enum
	    {
		param_inputs = 0,
		param_outputs = 1, 
		param_init_distance = 2,
		param_fitness = 3
	    };

	    void set_time(ty t)
	    {
		m_tdt = t;
	    }


	    //<TODO> use point inputs instead
	    virtual bool set_initial_distance(size_t id, size_t pt, const ty & distance)
	    {
		std::vector<ty> d; d.push_back(distance);
		return this->set_initial_distance (id, pt, d);
	    }

	    virtual bool set_initial_distance(size_t id, size_t pt, const std::vector<ty> & distance)
	    {
		if (distance.size() == 1)
		{
		    return task<ty>::set_inputs (id, pt, param_init_distance, distance, 1);
		}
		return false;
	    }

	    virtual bool set_inputs(size_t id, size_t pt, const std::vector<ty> & inputs)
	    {
		if (inputs.size() == m_inputs)
		{
		    return task<ty>::set_inputs (id, pt, param_inputs, inputs, m_inputs);
		}
		return false;
	    }

	    virtual bool set_outputs(size_t id, size_t pt, const std::vector<ty> & outputs)
	    {
		if (outputs.size() == m_outputs)
		{
		    return task<ty>::set_inputs (id, pt, param_outputs, outputs, m_outputs);
		}
		return false;
	    }

	    virtual bool get_fitness( size_t id, size_t pt, std::vector<ty> & fitness)
	    {
		return task<ty>::get_outputs (id, pt, param_fitness, fitness);
	    }

	    virtual bool prepare_outputs()
	    {
		return task<ty>::prepare_dataset(param_fitness, 1);
	    }

	    virtual bool launch()
	    {

		typename dataset<ty>::ptr pState = this->get_dataset(param_inputs);
		typename dataset<ty>::ptr pOutData = this->get_dataset(param_outputs);
		typename dataset<ty>::ptr pFitness = this->get_dataset(param_fitness);
		typename dataset<ty>::ptr pInitDistance = this->get_dataset(param_init_distance);

		if (!(pState && pOutData && pFitness && pInitDistance))
		{

		    CUDA_LOG_ERR(this->m_name, " Could not find a dataset ", 0);
		    CUDA_LOG_ERR(this->m_name, " state " , pState);
		    CUDA_LOG_ERR(this->m_name, " outdata ",  pOutData);
		    CUDA_LOG_ERR(this->m_name, " fitness ",  pFitness);
		    CUDA_LOG_ERR(this->m_name, " initial distance ",  pInitDistance);
		    return false;
		}

		block_complete_dimensions dims(&this->m_info, this->get_profile(), this->m_name);

		cudaError_t err = cudaSuccess;
		switch (m_fitness_type)
		{
		case  minimal_distance:
		    err = cu_compute_fitness_mindis<ty, pre_exec, post_exec>(*pState->get_data(),*pOutData->get_data(), *pFitness->get_data(), 
									     *pInitDistance->get_data(), pOutData->get_task_size(), this->m_dims.get()); 
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

		if (err != cudaSuccess)
		{
		    CUDA_LOG_ERR(this->m_name, " launch fail ", err);
		    return false;
		}
		return true;
	    }
	protected:

	    const size_t m_fitness_type;
	    const size_t  m_inputs;
	    const size_t  m_outputs;
	    const size_t  m_fitness;
	    ty m_vicinity_distance;
	    ty m_vicinity_speed;
	    ty m_max_docking_time;
	    ty m_tdt;
  	    kernel_dimensions::ptr m_dims;

	};
    }
}


#endif
