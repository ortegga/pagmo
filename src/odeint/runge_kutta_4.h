

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
	
	template <typename ty, typename system, size_t in_, size_t out_, size_t system_params, typename kernel_dims1 = block_complete_dimensions, 
	    typename pre_exec_o = nop_functor<ty>, typename pre_exec=nop_functor<ty>, typename post_exec=nop_functor<ty> >
	    class ode_step_runge_kutta_4 : public integrator<ty, system, in_, out_, system_params>
	{
	public:
	
	typedef integrator<ty, system , in_, out_, system_params> base;
	
	ode_step_runge_kutta_4 (cuda::info & inf, const std::string & name, size_t individuals, size_t task_count_) : base(inf, name, individuals, task_count_)
	{
	    this->set_shared_chunk(0, 0 , (4*in_ + in_ + out_) * sizeof(ty) );//inputs + outputs for all individuals for all points plus 6 for each k of the rk method
	    this->set_global_chunk(0, 0 , (in_ + out_) * sizeof(ty) );
	    this->m_dims = kernel_dimensions::ptr( new kernel_dims1 (&this->m_info, this->get_profile(), this->m_name));	    
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
	    cudaError_t err =  runge_kutta_integrate<ty, system, in_, system_params, pre_exec_o, pre_exec, post_exec>(pX->get_data(), pO->get_data(), 
													      this->m_param_t, this->m_param_dt,
													      this->m_param_scale_limits, 
													      this->m_dims.get());
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(this->m_name, " Launch fail ", err);
		return false;
	    }
	    return true;
		
	}
	protected:
	kernel_dimensions::ptr m_dims;

	};

    }
}
#endif
