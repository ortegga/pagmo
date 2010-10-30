

#ifndef __PAGMO_RUNGE_KUTTA_4__
#define __PAGMO_RUNGE_KUTTA_4__


#include "integrator.h"
#include "../cuda/kernel.h"
#include "../cuda/kernel_dims.h"
#include "../cuda/logger.h"

namespace pagmo
{
  namespace odeint
  {

    template <typename ty, typename system >
      class ode_step_runge_kutta_4 : public integrator<ty, 6, system>
      {
      public:

	typedef integrator<ty, 6, system > base;
    
	ode_step_runge_kutta_4 (cuda::info & inf, size_t individuals, size_t task_count_) : base(inf, individuals, task_count_)
	  {
	  }

	bool launch()
	{
	  using namespace cuda;
	  //InParams
	  dataset<ty> * pX = this->get_dataset(this->param_x);
	  dataset<ty> * pO = this->get_dataset(this->param_o);

	  if (!pX || !pO)
	    {
	      CUDA_LOG_ERR("Could not find a dataset", 0);
	      return false;
	    }
	  
	  //TODO handle shared memory and different individuals
	  size_t shared_data_size = pX->get_byte_size() + pO->get_byte_size();
	  size_t global_data_size = shared_data_size;

	  block_complete_dimensions dims (&this->m_info, this->get_profile());

	  runge_kutta_integrate<ty, system>(*pX->get_data()  , *pO->get_data(), this->m_param_t, this->m_param_dt,this->m_param_scale_limits, &dims);

	  return true;

	}

      };

  }
}
#endif
