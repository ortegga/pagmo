

#ifndef __PAGMO_RUNGE_KUTTA_4__
#define __PAGMO_RUNGE_KUTTA_4__


#include "integrator.h"
#include "../cuda/kernel.h"
#include "../cuda/dataset.h"
#include "../cuda/kernel_dims.h"
#include "../cuda/logger.h"

namespace pagmo
{
    namespace odeint
    {
	
	template <typename ty, typename system, typename pre_exec=nop_functor<ty>, typename post_exec=nop_functor<ty> >
	    class ode_step_runge_kutta_4 : public integrator<ty, 6, system>
	{
	public:
	
	typedef integrator<ty, 6, system > base;
	
	ode_step_runge_kutta_4 (cuda::info & inf, const std::string & name, size_t individuals, size_t task_count_) : base(inf, name, individuals, task_count_)
	{
	    this->set_shared_chunk(0, 0 , 24 + 6 + 2);//inputs + outputs for all individuals for all points plus 6 for each k of the rk method
	    this->set_global_chunk(0, 0 , 6 + 2);
	    
	}

	bool launch()
	{
	    using namespace cuda;
	    //InParams
	    typename dataset<ty>::ptr pX = this->get_dataset(this->param_x);
	    typename dataset<ty>::ptr pO = this->get_dataset(this->param_o);

	    if (!pX || !pO)
	    {
		CUDA_LOG_ERR(this->m_name, " Could not find a dataset ", 0);
		CUDA_LOG_ERR(this->m_name, " inputs " , pX);
		CUDA_LOG_ERR(this->m_name, " outputs ",  pO);
		return false;
	    }
	  
	    block_complete_dimensions dims (&this->m_info, this->get_profile(), this->m_name);

	    CUDA_LOG_WARN(this->m_name, "block_complete_dimensions", &dims);
	    cudaError_t err =  runge_kutta_integrate<ty, system, scale_functor<ty>, pre_exec, post_exec>(*pX->get_data()  , *pO->get_data(), this->m_param_t, 
										      this->m_param_dt,this->m_param_scale_limits, &dims);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(this->m_name, " Launch fail ", err);
		return false;
	    }
	    return true;
		
	}

	};

    }
}
#endif
