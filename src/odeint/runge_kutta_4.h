

#ifndef __PAGMO_RUNGE_KUTTA_4__
#define __PAGMO_RUNGE_KUTTA_4__

#include "ode_tasks.h"
#include "../cuda/tasklet.h"


//Todo finish integrator code
namespace odeint
{

  template <typename ty>
    class hills_system  : public cuda::tasklet<ty>
    {
    public:
    typedef typename odeint::hills_eq_task <ty> task;
    hills_system (unsigned int size, cuda::task<ty> * task_) : cuda::tasklet<ty>(task_), m_size(size)
      {

      }
      unsigned int get_size()
      {
	return m_size;
      }
      void set_size(unsigned int size)
      {
	m_size = size;
      }

      bool set_inputs(std::vector<ty> & inputs)
      {
	if (inputs.size() == get_size())
	  {
    	    return this->set_inputs (hills_eq_task<ty>::param_state, inputs);
	  }
      }
    protected:
      unsigned int m_size;

    };

  template <typename ty, typename system >
    class ode_step_runge_kutta_4 : public cuda::tasklet<ty>
  {
  public:
    typedef typename odeint::runge_kutta_4_task <ty,system> task;
    
    ode_step_runge_kutta_4 (unsigned int size, cuda::task<ty> * task_) : cuda::tasklet<ty>(task_), m_size(size)
    {

    }
   
    virtual bool prepare_outputs()
    {
      return m_system_task->prepare_outputs() && 
	this->prepare_dataset( this->m_task->param_x, get_size()) && 
	this->prepare_dataset( this->m_task->param_dx_dt, get_size()) &&
	this->prepare_dataset( this->m_task->param_dx_dm, get_size()) &&
	this->prepare_dataset( this->m_task->param_x_t, get_size());
    }
    unsigned int get_size()
    {
      return m_size;
    }
    void set_size(unsigned int size)
    {
      m_size = size;
    }
    void set_params(ty t, ty dt)
    {
      this->m_task->set_params( t,  dt);
    }

  protected:
    unsigned int m_size;
    system * m_system_task;
  };
}
#endif
