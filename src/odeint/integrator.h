#ifndef __PAGMO_CUDA_INTEGRATOR__
#define __PAGMO_CUDA_INTEGRATOR__

#include "../cuda/cudainfo.h"
#include "../cuda/cudatask.h"
#include "../cuda/kernel.h"


namespace pagmo
{
    namespace odeint
    {

	using namespace cuda;
	template <typename ty, size_t order, typename system>
	    class integrator : public task<ty>
	{
	public:
	    integrator (cuda::info & inf, const std::string & name, size_t individuals, size_t task_count_) : 
	    cuda::task<ty>(inf, name, individuals, task_count_, 1), 
		m_param_t(0),  m_param_dt(0), m_param_scale_limits(0)
	    {
	    }
	    virtual ~integrator ()
	    {
	  
	    }

	    enum
	    {
		param_x = 0,
		param_o
	    };

	    virtual bool set_inputs(size_t individual, size_t taskid, const std::vector<ty> & inputs)
	    {
		if (inputs.size() == get_size())
		{
		    return task<ty>::set_inputs (individual, taskid, this->param_x, inputs, get_size());
		}
		return false;
	    }

	    virtual bool set_dynamical_inputs(size_t individual, size_t taskid, const std::vector<ty> & inputs)
	    {
		if (inputs.size() == 2)//<TODO> incomplete type
		{
		    return cuda::task<ty>::set_inputs (individual, taskid, this->param_o, inputs, 2);
		}
		return false;
	    }

	    virtual bool get_outputs( size_t individual, size_t taskid, std::vector<ty> & outputs)
	    {
		return task<ty>::get_outputs (individual, taskid, this->param_x, outputs);
	    }
     
	    virtual bool prepare_outputs()
	    {
		return true;
	    }
	    unsigned int get_size()
	    {
		return order;
	    }

	    void set_params(ty t, ty dt, ty scale_limits)
	    {
		m_param_t = t;
		m_param_dt = dt;
		m_param_scale_limits = scale_limits;
	    }

	    virtual bool launch() = 0;
	protected:

	    ty m_param_t;
	    ty m_param_dt;
	    ty m_param_scale_limits;	
	};
    }
}


#endif
