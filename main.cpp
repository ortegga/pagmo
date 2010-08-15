
#include <stdio.h>
#include <vector>


#include "src/cuda/cudainfo.h"
#include "src/cuda/cudatimer.h"
#include "src/cuda/cudatask.h"
#include "src/cuda/layer.h"
//#include "src/odeint/runge_kutta_4.h"
#include "src/ann_toolbox/multilayer_perceptron.h"


using namespace cuda;

#define CUDA_TY float

int load_subtask( int taskid, ann_toolbox::neural_network<CUDA_TY> * pNet);
int print_subtask( int taskid, ann_toolbox::neural_network<CUDA_TY> * pNet);

int main( int argc, char **argv )
{

  using namespace cuda;

  info inf;
  std::cout << inf << std::endl;

  unsigned int taskCount = 200;

  //task might need to know the complete type of the subtasks.

  unsigned int I = 4, H = 4, O = 4;
  ann_toolbox::multilayer_perceptron<CUDA_TY, sigmoid>::task  task_(inf, taskCount);
  ann_toolbox::multilayer_perceptron<CUDA_TY, sigmoid> perc(I, H, &task_, O);
  //Timer scope
  {
    scoped_timer tim("multilayer nnet cumulative timer");
    for (unsigned int i=0; i < taskCount; i++)
      {
	load_subtask(i, & perc);
      }
    //Timer scope
    {
      scoped_timer launchTimer("nnet launch timer");
      if(!task_.launch())
	{
	  std::cout<<"launch fail"<<std::endl;
	}
    }
  }

  for (unsigned int i=0; i < taskCount; i++)
  {
    print_subtask(i, & perc);
  }
}
int load_subtask(int taskid, ann_toolbox::neural_network<CUDA_TY> * pNet)
{

  CUDA_TY x = 1.0f + taskid;

  std::vector<CUDA_TY> X (pNet->get_number_of_inputs(), x);
  std::vector<CUDA_TY> W;

  std::cout<<"weight count"<<pNet->get_number_of_weights()<<std::endl;
  for (unsigned int j=0; j< pNet->get_number_of_weights(); j++)
    {
      W.push_back(j + 1 + taskid);
    }

  std::cout<<"Prepare task"<<std::endl;
  pNet->set_task(taskid);
  std::cout<<"Prepare weights"<<std::endl;
  pNet->set_weights(W);
  std::cout<<"Prepare inputs"<<std::endl;
  pNet->set_inputs(X);
  std::cout<<"Prepare outputs"<<std::endl;
  pNet->prepare_outputs();
  std::cout<<"done"<<std::endl;
  return 0;
};


int print_subtask(int taskid, ann_toolbox::neural_network<CUDA_TY> * pNet)
{

  std::vector<CUDA_TY> O;
  pNet->set_task(taskid);
  if(!pNet->get_outputs(O))
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

