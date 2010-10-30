#ifndef __PAGMO_CUDA_EULER_INT__
#define __PAGMO_CUDA_EULER_INT__

#include "integrator.h"
#include "../cuda/kernel.h"
#include "../cuda/kernel_dims.h"
#include "../cuda/logger.h"

namespace pagmo
{
  namespace odeint
  {

    template <typename ty, typename system >
      class euler_integrator : public integrator<ty, 1, system>
      {
      public:

	typedef typename integrator<ty, 1, system> base;
    
	euler_integrator (cuda::info & inf, size_t individuals, size_t task_count_) : base(inf, individuals, task_count_)
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
	  size_t shared_data_size = pX->get_byte_size() + pO->get_byte_size(); //pInput->get_task_byte_size() + pWeights->get_task_byte_size();
	  size_t global_data_size = shared_data_size;

	  block_complete_dimensions dims (this->m_inf, pX->get_tasksize(), this->m_task_count, shared_data_size, global_data_size);

	  euler_integrate<ty, system>(*pX->get_data()  , *pO->get_data(), m_param_t, m_param_dt,pX->get_tasksize(), 
					    m_param_scale_limits, task_data_size, &dims);

	  return true;

	}

      };

  }
}



#endif
