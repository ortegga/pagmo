

#ifndef __PAGMO_RUNGE_KUTTA_4__
#define __PAGMO_RUNGE_KUTTA_4__

#include "../cuda/cudatask.h"
#include "../cuda/kernel.h"

//Todo finish integrator code
namespace pagmo
{
  namespace odeint
  {

    template <typename ty, typename system >
      class ode_step_runge_kutta_4 : public cuda::task<ty>
      {
      public:
    
	ode_step_runge_kutta_4 (cuda::info & inf, unsigned int task_count_) : cuda::task<ty>(inf, task_count_), 
	  m_param_t(0),  m_param_dt(0)
	  {
	  }

	enum
	{
	  param_x = 0,
	  param_o
	};


	virtual bool set_inputs(int taskid, const std::vector<ty> & inputs)
	{
	  if (inputs.size() == get_size())
	    {
	      return cuda::task<ty>::set_inputs (taskid, this->param_x, inputs);
	    }
	  return false;
	}

	virtual bool set_dynamical_inputs(int taskid, const std::vector<ty> & inputs)
	{
	  if (inputs.size() == 2)
	    {
	      return cuda::task<ty>::set_inputs (taskid, this->param_o, inputs);
	    }
	  return false;
	}

	virtual bool get_outputs( int taskid, std::vector<ty> & outputs)
	{
	  return cuda::task<ty>::get_outputs (taskid, this->param_x, outputs); //<TODO>This might be wrong
	}

	virtual bool prepare_outputs()
	{
	  return true;
	}
	unsigned int get_size()
	{
	  return task_data_size;
	}

	void set_params(ty t, ty dt, ty scale_limits)
	{
	  m_param_t = t;
	  m_param_dt = dt;
	  m_param_scale_limits = scale_limits;
	}

	bool launch()
	{
	  using namespace cuda;
	  //InParams
	  dataset<ty> * pX = this->get_dataset(this->param_x);
	  dataset<ty> * pO = this->get_dataset(this->param_o);

	  if (!pX || !pO)
	    {
	      std::cout <<" Could not find a dataset ("<<pX<<") "<<std::endl;
	      return false;
	    }
	  
	  //TODO handle shared memory and different individuals
	  size_t shared_data_size = 0; //pInput->get_task_byte_size() + pWeights->get_task_byte_size();
	  size_t global_data_size = *pX->get_byte_size() + pO->get_byte_size();

	  block_complete_dimensions dims (this->m_inf, pX->get_tasksize(), this->m_task_count, shared_data_size, global_data_size);

	  dim3 blocksize(pX->get_tasksize(),1,1);

	  dim3 gridsize(this->m_task_count,1,1);

	  runge_kutta_integrate<ty, system>(*pX->get_data()  , *pO->get_data(), m_param_t, m_param_dt,pX->get_tasksize(), 
					    m_param_scale_limits, task_data_size, blocksize, gridsize);

	  return true;

	}

      protected:

	//<TODO>Probably not a great idea
	ty m_param_t;
	ty m_param_dt;
	ty m_param_scale_limits;


	enum
	{
	  task_data_size = 6
	};
      };

  }
}
#endif
