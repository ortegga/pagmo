
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "src/ann_toolbox/cuda/external_dependency.h"

extern int doit();

int main( int argc, char **argv )
{
  CHECK_CUDA_ERROR();
  return doit();
};

