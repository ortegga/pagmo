
#include <stdio.h>
#include <vector>
#include "src/cudainfo.h"
#include "src/ann_toolbox/cudaty.h"
#include "src/ann_toolbox/multilayer_perceptron.h"
#include "src/ann_toolbox/perceptron.h"
#include "src/ann_toolbox/elman_network.h"

int main( int argc, char **argv )
{

  CudaInfo info;
  std::cout<<info;

  /*int i=2, h=2, o=2;
  std::vector<CUDA_TY> X (i, 1.0f);


  //int size = (i + 1)*h + (h + 1)*o;
    int size = (i + h + 1)*h + o*(h + 1);
  //   int size = (i + 1)*o;

    std::cout<<" weight size = "<<size<<std::endl;

  //std::vector<CUDA_TY> W(size, 2.0f);
  std::vector<CUDA_TY> W;


  for (int j=1; j<=size; j++)
    {
      W.push_back((CUDA_TY)j);
    }


  //ann_toolbox::multilayer_perceptron net(i,h,o, W);
  //ann_toolbox::perceptron net(i,o, W);
  ann_toolbox::elman_network net(i,h,o,W);


  net.set_weights(W);

  std::vector<CUDA_TY> Z = net.compute_outputs(X);

  for(std::vector<CUDA_TY>::iterator iter = Z.begin(); iter != Z.end(); ++iter)
    {
      std::cout<<*iter<<" ";
    }

  Z = net.compute_outputs(X);

  for(std::vector<CUDA_TY>::iterator iter = Z.begin(); iter != Z.end(); ++iter)
    {
      std::cout<<*iter<<" ";
    }
  */
  return 0;
};

