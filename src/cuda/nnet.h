#ifndef __PAGMO_CUDA_NNET_KERNELS__
#define __PAGMO_CUDA_NNET_KERNELS__

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


template <typename cuda_type, typename activ_type>
__host__   void cu_compute_layer(cuda_type *X, cuda_type *W,  cuda_type *Y, 
			size_t inputs, 
				 cuda::kernel_dimensions * dimensions_);

template <>
__host__ void cu_compute_layer<float, linear_functor<float> >(float *X, float *W,  float *Y, 
						     size_t inputs, 
						     cuda::kernel_dimensions * dimensions_);
template <>
__host__ void cu_compute_layer<float, sigmoid_functor<float> >(float *X, float *W,  float *Y, 
						      size_t inputs, 
						     cuda::kernel_dimensions * dimensions_);

template <>
__host__ void cu_compute_layer<double, linear_functor<double> >(double *X, double *W, double *Y, 
						       size_t inputs, 
						       cuda::kernel_dimensions * dimensions_);

template <>
__host__ void cu_compute_layer<double, sigmoid_functor<double> >(double *X, double *W,  double *Y, 
							size_t inputs, 
							cuda::kernel_dimensions * dimensions_);


#endif
