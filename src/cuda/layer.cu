//-*-c++-*-
//Basic kernel to compute the result of a layer's operation
#include "layer.h"
#include "stdio.h"
#include "cudaty.h"

#define SIGMOID(X) (1.0f/(1 + exp(-(X))));
//#define SIGMOID(X) (X)


__global__ void cu_computelayer(CUDA_TY *X, CUDA_TY *W,  CUDA_TY *Y, int width) 
{

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int offset = bx*blockDim.x;
  int itemid = offset + tx;
  int woffset = itemid * width;
  
  CUDA_TY value = W[woffset + width];
  for (int i=0; i < width; ++i)
  {
    value += X[offset + i]*W[woffset + i];
  }
  Y[itemid] = SIGMOID( value );
};

__global__ void cu_computelayerWithSegment(CUDA_TY *X,  CUDA_TY *W,  CUDA_TY *Y, int width, int seg) 
{

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  /*The order of weights is as follows:
   1) the weights between X and Y
   2) the bias for Y
   3) the weights for the memory component*/
  int offset = tx*(width+1);

  CUDA_TY value = W[offset + seg];
  for (int i=0; i < seg; ++i)
  {
    value += X[i]*W[offset + i];
  }

  for (int i=seg; i < width; ++i)
  {
    value += X[i]*W[offset +  i  + 1];
  }

  Y[tx] = SIGMOID( value );
};


#define input_to_hidden_weights(idx)	m_weights[(idx)]
#define hidden_to_hidden_weights(idx)	m_weights[dI * dH + (idx)]
#define hidden_bias(idx)	        m_weights[dI * dH + dH * dH + (idx)]
#define hidden_taus(idx)		m_weights[dI * dH + dH * dH + dH + (idx)]
#define hidden_to_output_weights(idx)	m_weights[dI * dH + dH * dH + dH + dH + (idx)]
#define output_bias(idx)		m_weights[dI * dH + dH * dH + dH + dH + dH * dO + (idx)]



__global__ void cu_computeCtrnnLayer(CUDA_TY *X,  CUDA_TY *W,  CUDA_TY *Y, int width, int seg) 
{

}

void cuComputeLayer(CUDA_TY *X, CUDA_TY *W,  CUDA_TY *Y, int width, 
		    dim3 gridsize, dim3 blocksize)
{
  cu_computelayer<<<gridsize, blocksize>>>(X, W, Y, width);
}

void cuComputeLayerWithSegment(CUDA_TY *X, CUDA_TY *W,  CUDA_TY *Y, int width, int seg,
		    dim3 gridsize, dim3 blocksize)
{
  cu_computelayerWithSegment<<<gridsize, blocksize>>>(X, W, Y, width,seg);
}


