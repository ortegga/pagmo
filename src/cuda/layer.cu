//-*-c++-*-
//Basic kernel to compute the result of a layer's operation
#include "layer.h"
#include "stdio.h"

///////////////////////////////////////////////////////////
template <typename cuda_type, int activation_type>
__device__ cuda_type activate (cuda_type val)
{
  return val;
}

template <>
__device__ float activate<float, sigmoid> (float val)
{
  return 1.0f/(1 + exp(-val));
}

template <>
__device__ double activate<double, sigmoid> (double val)
{
  return 1.0f/(1 + exp(-val));
}

template <>
__device__ float activate<float, linear> (float val)
{
  return val > 0.0f ? 1.0f : 0.0f;
}

template <>
__device__ double activate<double, linear> (double val)
{
  return val > 0.0f ? 1.0f : 0.0f;
}

///////////////////////////////////////////////////////////
template <typename cuda_type, int activ_type >
__global__ void cu_compute_layer_kernel(cuda_type *X, cuda_type *W,  
				cuda_type *Y, int width) 
{

  unsigned int bx = blockIdx.x;
  unsigned int tx = threadIdx.x;

  unsigned int offset = bx*blockDim.x;
  unsigned int itemid = offset + tx;
  unsigned int woffset = itemid * width;
  
  cuda_type value = W[woffset + width];
  for (int i=0; i < width; ++i)
  {
    value += X[offset + i]*W[woffset + i];
  }
  Y[itemid] = activate <cuda_type, activ_type>( value );
};


template <>
void cu_compute_layer<float, sigmoid>(float *X, float *W,  
		    float *Y, int width, 
		    dim3 gridsize, dim3 blocksize)
{
  cu_compute_layer_kernel<float, sigmoid><<<gridsize, blocksize>>>(X, W, Y, width);
}

template <>
void cu_compute_layer<float, linear>(float *X, float *W,  
		    float *Y, int width, 
		    dim3 gridsize, dim3 blocksize)
{
  cu_compute_layer_kernel<float, linear><<<gridsize, blocksize>>>(X, W, Y, width);
}

///////////////////////////////////////////////////////////
template <typename cuda_type, int activ_type>
__global__ void cu_compute_layer_with_segments_kernel(cuda_type *X,  cuda_type *W,  cuda_type *Y, int width, int seg) 
{

  unsigned int bx = blockIdx.x, by = blockIdx.y;
  unsigned int tx = threadIdx.x, ty = threadIdx.y;

  /*The order of weights is as follows:
   1) the weights between X and Y
   2) the bias for Y
   3) the weights for the memory component*/
  unsigned int offset = tx*(width+1);

  cuda_type value = W[offset + seg];
  for (unsigned int i=0; i < seg; ++i)
  {
    value += X[i]*W[offset + i];
  }

  for (unsigned int i=seg; i < width; ++i)
  {
    value += X[i]*W[offset +  i  + 1];
  }

  Y[tx] = activate<cuda_type, activ_type>( value );
};


template <typename cuda_type, int activ_type>
void cu_compute_layer_with_segments(cuda_type *X, cuda_type *W,  
				    cuda_type *Y, int width, int seg,
				    dim3 gridsize, dim3 blocksize)
{
  cu_compute_layer_with_segments_kernel<cuda_type, activ_type><<<gridsize, blocksize>>>(X, W, Y, width, seg);
}

///////////////////////////////////////////////////////////////////////
// computes y += alpha * x1
template <typename cuda_type>
__global__ void cu_increment_kernel(cuda_type *Y,  cuda_type *X,  cuda_type alpha, int width) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] += alpha * X[idx];
}

template <>
void cu_increment(float *Y, float *X,  
		  float alpha, int width,
		  dim3 gridsize, dim3 blocksize)
{
  cu_increment_kernel<float><<<gridsize, blocksize>>>(Y, X, alpha, width);
}

///////////////////////////////////////////////////////////////////////
// computes y = x1 - x2
template <typename cuda_type>
__global__ void cu_assign_diff_kernel(cuda_type *Y,  cuda_type *X1,  cuda_type * X2, int width) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = X1[idx] - X2[idx];
}

template <>
void cu_assign_diff(float *Y, float *X1, float*X2, 
			       int width, dim3 gridsize, dim3 blocksize)
{
  cu_assign_diff_kernel<float><<<gridsize, blocksize>>>(Y, X1, X2, width);
}

///////////////////////////////////////////////////////////////////////
// computes y = x1 + alpha * x2
template <typename cuda_type>
__global__ void cu_assign_sum_kernel(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, cuda_type alpha, int width) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = X1[idx] + alpha * X2[idx];
}

template <>
void cu_assign_sum(float *Y,  float *X1, float* X2,
		   float alpha, int width, dim3 gridsize, dim3 blocksize) 
{
  cu_assign_sum_kernel<float><<<gridsize, blocksize>>>(Y, X1, X2, alpha, width);
}

///////////////////////////////////////////////////////////////////////
// computes y = alpha1 * ( x1 + x2 + beta*x3 )
template <typename cuda_type>
__global__ void cu_increment_sum_sum_kernel(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
					    cuda_type* X3, cuda_type alpha, cuda_type beta, int width) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = alpha*(X1[idx] + X2[idx] + beta*X3[idx]);
}

template <>
void cu_increment_sum_sum(float *Y,  float *X1,  float* X2, 
			  float* X3, float alpha, float beta, 
			  int width, dim3 gridsize, dim3 blocksize) 
{
  cu_increment_sum_sum_kernel<float><<<gridsize,blocksize>>>(Y, X1, X2, X3, alpha, beta, width);
}

///////////////////////////////////////////////////////////////////////
// computes y = x1 + alpha * x2 ; x2 += x3
template <typename cuda_type>
__global__ void cu_assign_sum_increment_kernel(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
					       cuda_type* X3, cuda_type alpha, int width) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = X1[idx] + alpha*X2[idx];
  X2[idx] += X3[idx];
}

template <>
void cu_assign_sum_increment(float *Y,  float *X1,  float* X2, 
			     float* X3, float alpha, int width,
			     dim3 gridsize, dim3 blocksize) 
{
  cu_assign_sum_increment_kernel<float><<<gridsize,blocksize>>>(Y,X1,X2,X3,alpha,width);
}

///////////////////////////////////////////////////////////////////////
// hills equation


template <typename cuda_type>
__global__ void cu_hills_equation_kernel(cuda_type *S,  cuda_type *D,  cuda_type* O, 
					       cuda_type t) 
{

  const cuda_type nu = 0.08, mR = (1.5 * 0.5);	

  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  unsigned int sstride = 6*idx;
  unsigned int ostride = 2*idx;

  cuda_type x = S[sstride];
  cuda_type vx = S[++sstride];
  cuda_type y = S[++sstride];
  cuda_type vy = S[++sstride];
  cuda_type theta = S[++sstride];	
  cuda_type omega = S[++sstride];
	
  cuda_type distance = sqrt(x * x + y * y);

  if(theta < -M_PI) theta += 2 * M_PI;
  if(theta > M_PI) theta -= 2 * M_PI;
	
  cuda_type ul = O[ostride];
  cuda_type ur = O[++ostride];
       
  D[sstride] = (ul - ur) * 1/mR;
  D[--sstride] = omega;
  D[--sstride] = -2 * nu * vx + (ul + ur) * sin(theta);
  D[--sstride] = vy;
  D[--sstride] = 2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta);
  D[--sstride] = vx;
  }

template <>
void cu_hills_equation( float *S , float *D , 
			  float*O, float t,
			  dim3 gridsize, dim3 blocksize) 
{
  cu_hills_equation_kernel<float><<<gridsize,blocksize>>>(S,D,O,t);
}

 //////////////////////////////////////////////////////////////////////////////////
 //
 /////////////////////////////////////////////////////////////////////////////////

/*template <typename cuda_type, int activ_type>
void cuComputeLayer(cuda_type *X, cuda_type *W,  
		    cuda_type *Y, int width, 
		    dim3 gridsize, dim3 blocksize)
{
  cu_computelayer<cuda_type, activ_type><<<gridsize, blocksize>>>(X, W, Y, width);
  }*/
