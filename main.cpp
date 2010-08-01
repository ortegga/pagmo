
#include <stdio.h>
#include <vector>


#include "src/cuda/cudainfo.h"
#include "src/cuda/cudatimer.h"
#include "src/cuda/cudatask.h"
#include "src/cuda/cudaty.h"

#include "src/ann_toolbox/multilayer_perceptron.h"


int load_subtask( int taskid, ann_toolbox::neural_network * pNet);
int print_subtask( int taskid, ann_toolbox::neural_network * pNet);

int main( int argc, char **argv )
{
  CudaInfo info;
  std::cout << info << std::endl;

  int taskCount = 200;
  MultilayerPerceptronTask task (info, taskCount);
  int I = 4, H = 4, O = 4;
  ann_toolbox::multilayer_perceptron perc(I, H, & task, O);
  //Timer scope
  {
    ScopedCudaTimer timer("multilayer nnet cumulative timer");
    for (int i=0; i < taskCount; i++)
      {
	load_subtask(i, & perc);
      }
    //Timer scope
    {
      ScopedCudaTimer launchTimer("nnet launch timer");
      task.Launch();
    }
  }

  for (int i=0; i < taskCount; i++)
  {
    print_subtask(i, & perc);
  }
}
int load_subtask(int taskid, ann_toolbox::neural_network * pNet)
{

  CUDA_TY x = 1.0f + taskid;

  std::vector<CUDA_TY> X (pNet->get_number_of_inputs(), x);
  std::vector<CUDA_TY> W;

  std::cout<<"weight count"<<pNet->get_number_of_weights()<<std::endl;
  for (int j=0; j< pNet->get_number_of_weights(); j++)
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


int print_subtask(int taskid, ann_toolbox::neural_network * pNet)
{

  std::vector<CUDA_TY> O;
  pNet->set_task(taskid);
  pNet->get_outputs(O);

  for(std::vector<CUDA_TY>::iterator iter = O.begin(); iter != O.end(); ++iter)
  {
    std::cout<<*iter<<" ";
  }

  std::cout<<std::endl;

  return 0;
};

