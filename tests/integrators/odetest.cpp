
#include <stdio.h>
#include <vector>


#include "../src/cuda/cudainfo.h"
#include "../src/cuda/cudatimer.h"
#include "../src/cuda/cudatask.h"
#include "../src/odeint/runge_kutta_4.h" 


using namespace cuda;
using namespace pagmo;

#define CUDA_TY float


typedef hills_dynamical_system<CUDA_TY > dynamic_system;
typedef odeint::ode_step_runge_kutta_4< CUDA_TY, dynamic_system , 7, 2, 2, adhoc_dimensions<128> > rk_integrator;


int load_subtask( int individualid, int taskid, rk_integrator * pNet);
int print_subtask(int individualid, int taskid, rk_integrator * pNet);

int main( int argc, char **argv )
{

  info * pInf = new cuda::info();
  info & inf = *pInf;
  std::cout << inf << std::endl;

  unsigned int taskCount = 50;
  unsigned int individuals = 30;

  //task might need to know the complete type of the subtasks.

  rk_integrator integ(inf, "runge kutta integrator", individuals, taskCount);
  //Timer scope
  {
    scoped_timer tim("runge_kutta cumulative timer");
    for (size_t j=0; j< individuals; j++)
      {
	for (size_t i=0; i < taskCount; i++)
	  {
	    load_subtask(j, i, & integ);
	  }
      }

      std::cout<<"Launch started "<<std::endl;
      //Timer scope
    {
      scoped_timer launchTimer("rk integrator launch timer");
      for (int i=0; i < 10; i++)
      {
	  integ.set_params(i * 0.1f, 0.1f,0);
	  if(!integ.launch())
	  {
	      std::cout<<"launch fail"<<std::endl;
	  }
      }
    }
  }

  for (unsigned int j=0; j < individuals; j++)
    {
      for (unsigned int i=0; i < taskCount; i++)
	{
	  print_subtask(j, i, & integ);
	}
    }
  delete pInf;
}


int load_subtask(int individualid, int taskid, rk_integrator * pInt)
{

  CUDA_TY x = 1.0f + taskid;

  std::vector<CUDA_TY> X (pInt->get_size(), x);
  std::vector<CUDA_TY> D (pInt->get_system_size(), x);
  pInt->set_inputs(data_item::point_data(0,individualid, taskid), X);
  pInt->set_dynamical_inputs(data_item::point_data(0,individualid, taskid), D);

  if(!individualid && !taskid)
    pInt->prepare_outputs();
  return 0;
};


int print_subtask(int individualid, int taskid, rk_integrator * pInt)
{

  std::vector<CUDA_TY> O;
  O.clear();
  
  if(!pInt->get_outputs(data_item::point_data(0,individualid, taskid), O))
    {
      std::cout<<" Error: could not get data"<<std::endl;
      return 0;
    }

  int i=0;
  for(std::vector<CUDA_TY>::iterator iter = O.begin(); i < 6 && iter != O.end(); ++iter, ++i)
    {
      std::cout<<*iter<<" ";
    }

    std::cout<<std::endl;

  return 0;
};

