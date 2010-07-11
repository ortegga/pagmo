#ifndef __PAGMO_ANN_LAYER__
#define __PAGMO_ANN_LAYER__

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaty.h"

void cuComputeLayer(CUDA_TY *X, CUDA_TY *W,  CUDA_TY *Y, int width, 
		    dim3 gridsize, dim3 blocksize);

void cuComputeLayerWithSegment(CUDA_TY *X, CUDA_TY *W,  CUDA_TY *Y, int width, int seg, 
		    dim3 gridsize, dim3 blocksize);


#endif 
