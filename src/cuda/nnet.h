#ifndef __PAGMO_CUDA_NNET_KERNELS__
#define __PAGMO_CUDA_NNET_KERNELS__

#include "common.h"

////////////////////////////////////////////////////////////////////////////////////////////
// activation types

template <typename cuda_type>
struct sigmoid_functor;


template <typename cuda_type>
struct linear_functor;


////////////////////////////////////////////////////////////////////////////////////////////
// neural networks

namespace cuda
{
    class kernel_dimensions;
}


template <typename cuda_type, typename pre_exec, typename activ_type>
    __host__   cudaError_t cu_compute_layer(cuda_type *X, cuda_type *W,  cuda_type *Y, 
				     size_t inputs, 
				     cuda::kernel_dimensions * dimensions_);

template <>
__host__ cudaError_t cu_compute_layer<float, nop_functor<float>, linear_functor<float> >(float *X, float *W,  float *Y, 
							      size_t inputs, 
							      cuda::kernel_dimensions * dimensions_);
template <>
__host__ cudaError_t cu_compute_layer<float, nop_functor<float>, sigmoid_functor<float> >(float *X, float *W,  float *Y, 
							       size_t inputs, 
							       cuda::kernel_dimensions * dimensions_);

template <>
__host__ cudaError_t cu_compute_layer<double, nop_functor<float>, linear_functor<double> >(double *X, double *W, double *Y, 
								size_t inputs, 
								cuda::kernel_dimensions * dimensions_);

template <>
__host__ cudaError_t cu_compute_layer<double, nop_functor<float>, sigmoid_functor<double> >(double *X, double *W,  double *Y, 
								 size_t inputs, 
								 cuda::kernel_dimensions * dimensions_);


#endif
