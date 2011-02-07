
#include <stdio.h>
#include <vector>


#include "../src/cuda/cudainfo.h"
#include "../src/cuda/cudatimer.h"
#include "../src/cuda/cudatask.h"
#include "../src/cuda/nnet.h"
//#include "src/odeint/runge_kutta_4.h"

#include "../src/ann_toolbox/multilayer_perceptron.h"
#include "../src/ann_toolbox/perceptron.h"


using namespace cuda;

#define CUDA_TY float


typedef ann_toolbox::multilayer_perceptron<CUDA_TY, 2, 3, 2, adhoc_dimensions<256>,  adhoc_dimensions<256> >  multilayer_nnet;
//typedef ann_toolbox::perceptron<CUDA_TY, 2, 2, adhoc_dimensions<256> >  multilayer_nnet;

int load_subtask( int individualid, int taskid, multilayer_nnet * pNet);
int print_subtask(int individualid, int taskid, multilayer_nnet * pNet);

int main( int argc, char **argv )
{

  info * pInf = new cuda::info();
  info & inf = *pInf;
  std::cout << inf << std::endl;

  unsigned int taskCount = 30;
  unsigned int individuals = 40;

  //task might need to know the complete type of the subtasks.

  multilayer_nnet perc(inf, "test network", individuals, taskCount);
  //Timer scope
  {
    scoped_timer tim("multilayer nnet cumulative timer");
    for (size_t j=0; j< individuals; j++)
      {
	for (size_t i=0; i < taskCount; i++)
	  {
	    load_subtask(j, i, & perc);
	  }
      }

    std::cout<<"Launch started "<<std::endl;
    //Timer scope
    {
      scoped_timer launchTimer("nnet launch timer");
      if(!perc.launch())
	{
	  std::cout<<"launch fail"<<std::endl;
	}
    }
  }

  for (unsigned int j=0; j< individuals; j++)
    {
      for (unsigned int i=0; i < taskCount; i++)
	{
	  print_subtask(j, i, & perc);
	}
    }
  delete pInf;
}
int load_subtask(int individualid, int taskid, multilayer_nnet * pNet)
{

  CUDA_TY x = 1.0f + taskid;

  std::vector<CUDA_TY> X (pNet->get_number_of_inputs(), x);
  std::vector<CUDA_TY> W;

  if (taskid == 0)
    {
      for (unsigned int j=0; j< pNet->get_number_of_weights(); j++)
	{
	  W.push_back(j + 1 + individualid);
	}
      //std::cout<<"Setting weights for individual "<<individualid<<std::endl;
      //std::cout<<
      pNet->set_weights(data_item::individual_data(0,individualid), W);
	  //<<std::endl;//" "<<taskid<<" "<<individualid<<std::endl;
    }

//  std::cout<<
  pNet->set_inputs(data_item::point_data(0,individualid, taskid), X);
  //<<" "<<taskid<<" "<<individualid<<std::endl;

  if(!individualid && !taskid)
//    std::cout<<
      pNet->prepare_outputs();
	    //<<std::endl;
  return 0;
};


int print_subtask(int individualid, int taskid, multilayer_nnet * pNet)
{

  std::vector<CUDA_TY> O;

  //std::cout <<std::endl<<individualid<<"/"<<taskid<<" outputs: ";

  O.clear();
  
  if(!pNet->get_outputs(data_item::point_data(0,individualid, taskid), O))
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

