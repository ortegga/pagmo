
#include <stdio.h>
#include <vector>


#include "../src/cuda/cudainfo.h"
#include "../src/cuda/cudatimer.h"
#include "../src/cuda/cudatask.h"
#include "../src/cuda/fitness_tasks.h" 


using namespace cuda;
using namespace pagmo;

#define CUDA_TY float

typedef fitness::evaluate_fitness_task<CUDA_TY, adhoc_dimensions<256> > fitness_type;


int load_subtask( int individualid, int taskid, fitness_type * pFit);
int print_subtask(int individualid, int taskid, fitness_type * pFit);

int main( int argc, char **argv )
{

  info * pInf = new cuda::info();
  info & inf = *pInf;
  std::cout << inf << std::endl;

  unsigned int taskCount = 50;
  unsigned int individuals = 30;

  //task might need to know the complete type of the subtasks.

  fitness_type fit(inf, "fitness evaluator", fitness_type::minimal_distance, individuals, taskCount, 
		   0.1, 0.1, 0.1, 1);
  //Timer scope
  {
    scoped_timer tim("fitness evaluator cumulative timer");
    for (size_t j=0; j< individuals; j++)
      {
	for (size_t i=0; i < taskCount; i++)
	  {
	    load_subtask(j, i, & fit);
	  }
      }

    std::cout<<"Launch started "<<std::endl;
    //Timer scope
    {
      scoped_timer launchTimer("fitness evaluator launch timer");
      if(!fit.launch())
	{
	  std::cout<<"launch fail"<<std::endl;
	}
    }
  }

  for (unsigned int j=0; j < individuals; j++)
    {
      for (unsigned int i=0; i < taskCount; i++)
	{
	  print_subtask(j, i, & fit);
	}
    }
  delete pInf;
}


int load_subtask(int individualid, int taskid, fitness_type * pFit)
{

  CUDA_TY x = 1.0f + taskid;

  std::vector<CUDA_TY> X (pFit->get_inputs(), x);
  std::vector<CUDA_TY> Y (pFit->get_outputs(), x + 0.1f);
  pFit->set_inputs(data_item::point_data(0,individualid, taskid), X);
  pFit->set_outputs(data_item::point_data(0,individualid, taskid), Y);
  pFit->set_initial_distance(data_item::point_data(0,individualid, taskid), x);

  if(!individualid && !taskid)
    std::cout<<pFit->prepare_outputs()<<std::endl;
  return 0;
};


int print_subtask(int individualid, int taskid, fitness_type * pFit)
{

  std::vector<CUDA_TY> O;
  O.clear();
  
  if(!pFit->get_fitness(data_item::point_data(0,individualid, taskid), O))
    {
      std::cout<<" Error: could not get data"<<std::endl;
      return 0;
    }

  for(std::vector<CUDA_TY>::iterator iter = O.begin(); iter != O.end(); ++iter)
    {
      std::cout<<*iter<<" ";
    }

  std::cout<<std::endl;

  return 0;
};

