#ifndef __PAGMO_CUDA_NNET_KERNELS__
#define __PAGMO_CUDA_NNET_KERNELS__

#include "common.h"

////////////////////////////////////////////////////////////////////////////////////////////
// activation types

template <typename cty>
struct sigmoid_functor;


template <typename cty>
struct linear_functor;


////////////////////////////////////////////////////////////////////////////////////////////
// neural networks

namespace cuda
{
    class kernel_dimensions;
}


template <typename cty, typename pre_exec, typename activ_type>
    __host__   cudaError_t cu_compute_layer(cty *X, cty *W,  cty *Y, 
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
