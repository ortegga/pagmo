#ifndef __PAGMO_RUNGE_KUTTA_4_INTEG__
#define __PAGMO_RUNGE_KUTTA_4_INTEG__

#include "../cuda/cudatask.h"
#include "../cuda/cudainfo.h"
#include "../cuda/layer.h"

namespace odeint
{


  /*<todo> add reference to docking problem here. 
  /But we need to know what a docking problem will
  /look like</todo>*/
  template <typename cuda_type>
    class hills_eq_task : public cuda::task<cuda_type>
    {
    public:
    hills_eq_task(cuda::info & in, int task_count) : 
      cuda::task<cuda_type> (in, task_count), m_param_t(cuda_type(0))
	{}

      enum 
      {
	param_state = 0,
	param_dx_dt,
	param_outputs
      };

      void set_param(cuda_type t)
      {
	m_param_t = t;
      }

      bool launch()
      {
	using namespace cuda;
	dataset<cuda_type> * pX = this->get_dataset(param_state);
	dataset<cuda_type> * pDxdt = this->get_dataset(param_dx_dt);
	dataset<cuda_type> * pOutputs = this->get_dataset(param_outputs);

	if (!(pX && pDxdt && pOutputs))
	  {
	    std::cout <<" Could not find a dataset"<<std::endl;
	    //Raise error that something was not initialised
	    return false;
	  }
	//each thread block contains O number of threads
	dim3 blocksize1(1,1,1);

	//The number of neural networks to simulate
	dim3 gridsize1(pX->get_tasksize()*this->m_task_count,1,1);

	cu_hills_equation<cuda_type >(*pX->get_data(), *pDxdt->get_data(), 
				*pOutputs->get_data(), m_param_t,  
				gridsize1, blocksize1);

	return true;
      }
    protected:
      cuda_type m_param_t;
    };
    
  template <typename cuda_type, typename sub_task_type>
    class runge_kutta_4_task : public cuda::task<cuda_type>
    {
    public:

      enum
      {
	param_x = 0,
	param_dx_dt,
	param_t,
	param_dt,
	param_dx_dm, 
	param_x_t
      };
    runge_kutta_4_task(cuda::info & inf, unsigned int task_count_, sub_task_type * sub_task_) 
      :cuda::task<cuda_type> (inf, task_count_), m_sub_task(sub_task_),
	m_param_t(cuda_type(0)), m_param_dt(cuda_type(0))
	{
	}


      bool launch()
      {
	using namespace cuda;
	//InParams
	dataset<cuda_type> * pX = this->get_dataset(param_x);
	dataset<cuda_type> * pDxdt = this->get_dataset(param_dx_dt);

	//OutParams

	dataset<cuda_type> * pDxdm = this->get_dataset(param_dx_dm);
	dataset<cuda_type> * pXt = this->get_dataset(param_x_t);

	if (!(pX && pDxdt && pDxdm && pXt && m_sub_task))
	  {
	    std::cout <<" Could not find a dataset"<<std::endl;
	    //Raise error that something was not initialised
	    return false;
	  }

	if (!launch_sub_task(pX,pDxdt,m_param_t))
	  {
	    return false;
	  }

	cuda_type dh = cuda_type(0.5) * m_param_dt;
	cuda_type th = m_param_t + dh;

	dim3 blocksize(1,1,1);

	dim3 gridsize(pXt->get_tasksize()*this->m_task_count,1,1);
	cu_assign_sum<cuda_type>( *pXt->get_data(), *pX->get_data() , *pDxdt->get_data() , blocksize, gridsize);

	if (!launch_sub_task(pXt,pDxdt,th))
	  {
	    return false;
	  }

	cu_assign_sum<cuda_type>( *pXt->get_data() , *pX->get_data() , *pDxdt->get_data() , dh , blocksize, gridsize);

	if (!launch_sub_task(pXt,pDxdm,th))
	  {
	    return false;
	  }

	cu_assign_sum_increment<cuda_type>( *pXt->get_data(), *pX->get_data() , *pDxdm->get_data() ,
					    *pDxdt->get_data(), m_param_dt  , blocksize, gridsize);

	if (!launch_sub_task(pXt,pDxdt,m_param_t+m_param_dt))
	  {
	    return false;
	  }

	dim3 blocksize1(pX->get_tasksize()*this->m_task_count,1,1);
	cu_increment_sum_sum<cuda_type>( *pX->get_data() , *pDxdt->get_data() ,
					 pDxdt->get_data() , *pDxdm->get_data(),
					 m_param_dt /  cuda_type( 6.0 ) , cuda_type(2.0),
					  blocksize1, gridsize);
	return true;

      }


      void set_params(cuda_type t, cuda_type dt)
      {
	m_param_t = t;
	m_param_dt = dt;
      }


    protected:


      bool launch_sub_task(cuda::dataset<cuda_type> *state, cuda::dataset<cuda_type>*deriv, cuda_type t)
      {
	m_sub_task->assign_data(sub_task_type::param_state,state,true);
	m_sub_task->assign_data(sub_task_type::param_dx_dt,deriv,true);
	m_sub_task->set_param(t);

	if (!m_sub_task->launch())
	  {
	    return false;
	  }
	//use shared pointers instead
	m_sub_task->assign_data(sub_task_type::param_state,NULL,true);
	m_sub_task->assign_data(sub_task_type::param_dx_dt,NULL,true);
	return true;
      }
	sub_task_type * m_sub_task;

	//<TODO>Probably not a great idea
	cuda_type m_param_t;
	cuda_type m_param_dt;
    };
}


#endif
