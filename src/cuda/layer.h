#ifndef __PAGMO_ANN_LAYER__
#define __PAGMO_ANN_LAYER__

#include <cuda.h>
#include <cuda_runtime.h>

const int linear = 0;
const int sigmoid = 1;

//////////////////////////////////////////////////////////////////////////////
template <typename cuda_type, int activ_type>
void cu_compute_layer(cuda_type *X, cuda_type *W,  cuda_type *Y, int width, 
dim3 gridsize, dim3 blocksize);

template <>
void cu_compute_layer<float, sigmoid>(float *X, float *W,  float *Y, int width, 
		    dim3 gridsize, dim3 blocksize);

//////////////////////////////////////////////////////////////////////////////

template <typename cuda_type, int activ_type>
void cu_compute_layer_with_segments(cuda_type *X, cuda_type *W,  cuda_type *Y, int width, int seg, 
		    dim3 gridsize, dim3 blocksize);

//////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////

template <typename cuda_type>
void cu_increment(cuda_type *Y, cuda_type *X,  
		  cuda_type alpha, int width,
		  dim3 gridsize, dim3 blocksize);
template <>
void cu_increment(float *Y, float *X,  
		  float alpha, int width,
		  dim3 gridsize, dim3 blocksize);
/////////////////////////////////////////////////////////////////////////
template <typename cuda_type>
void cu_assign_diff(cuda_type *Y, cuda_type *X1, cuda_type*X2, 
		    int width, dim3 gridsize, dim3 blocksize);
template <>
void cu_assign_diff(float *Y, float *X1, float*X2, 
		    int width, dim3 gridsize, dim3 blocksize);

/////////////////////////////////////////////////////////////////////////
template <typename cuda_type>
void cu_assign_sum(cuda_type *Y,  cuda_type *X1, cuda_type* X2,
		   cuda_type alpha, int width, dim3 gridsize, dim3 blocksize);
template <>
void cu_assign_sum(float *Y,  float *X1, float* X2,
		   float alpha, int width, dim3 gridsize, dim3 blocksize);

/////////////////////////////////////////////////////////////////////////
template <typename cuda_type>
void cu_increment_sum_sum(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
			  cuda_type* X3, cuda_type alpha, cuda_type beta, 
			  int width, dim3 gridsize, dim3 blocksize);
template <>
void cu_increment_sum_sum(float *Y,  float *X1,  float* X2, 
			  float* X3, float alpha, float beta, 
			  int width, dim3 gridsize, dim3 blocksize);

/////////////////////////////////////////////////////////////////////////
template <typename cuda_type>
void cu_assign_sum_increment(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
			     cuda_type* X3, cuda_type alpha, int width,
			     dim3 gridsize, dim3 blocksize);
template <>
void cu_assign_sum_increment(float *Y,  float *X1,  float* X2, 
			     float* X3, float alpha, int width,
			     dim3 gridsize, dim3 blocksize);




template <typename cuda_type>
void cu_hills_equation( cuda_type *S , cuda_type *D , 
			  cuda_type*O, cuda_type t,
			  dim3 gridsize, dim3 blocksize);
template <>
void cu_hills_equation( float *S , float *D , 
			  float*O, float t,
			  dim3 gridsize, dim3 blocksize);

#endif 
