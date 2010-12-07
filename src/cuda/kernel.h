#ifndef __PAGMO_CUDA_KERNEL__
#define __PAGMO_CUDA_KERNEL__

#include "kernel_dims.h"
#include "common.h"

////////////////////////////////////////////////////////////////////////////////////////////


template <typename ty, typename preprocesor, typename pre_exec = nop_functor<ty>, typename post_exec = nop_functor<ty> >
    struct hills_dynamical_system ;

typedef hills_dynamical_system <float, scale_functor<float>, nop_functor<float>, nop_functor<float> > hills_dynamical_sys_float;
typedef hills_dynamical_system <double, scale_functor<double>, nop_functor<float>, nop_functor<double> > hills_dynamical_sys_double;



////////////////////////////////////////////////////////////////////////////////////////////

template <typename cuda_type, typename dynamicalsystem, typename pre_exec, typename post_exec>
    cudaError_t runge_kutta_integrate (cuda_type  * X , cuda_type * O, cuda_type t , cuda_type dt , 
				cuda_type max_val, cuda::kernel_dimensions * dimensions_);

template <>
cudaError_t runge_kutta_integrate <float, hills_dynamical_sys_float, nop_functor<float>, nop_functor<float> > 
    (float  * X , float * O, 
     float t , float dt , float max_val, 
     cuda::kernel_dimensions * dimensions_);

template <>
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double, nop_functor<float>, nop_functor<float> > 
    (double  * X , double * O, double t ,
     double dt , double max_val, 
     cuda::kernel_dimensions * dimensions_);


////////////////////////////////////////////////////////////////////////////////////////////



template <typename ty, typename pre_exec, typename post_exec>
cudaError_t cu_compute_fitness_mindis (ty *S , ty *O, ty *F, ty *I, size_t width, cuda::kernel_dimensions * dims_  );

template <>
cudaError_t cu_compute_fitness_mindis<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, 
									       float *I, size_t width, cuda::kernel_dimensions * dims_  );

template <>
cudaError_t cu_compute_fitness_mindis<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, 
										  double *I, size_t width, cuda::kernel_dimensions * dims_  );

#endif
