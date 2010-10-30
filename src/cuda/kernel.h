#ifndef __PAGMO_CUDA_KERNEL__
#define __PAGMO_CUDA_KERNEL__

#include "kernel_dims.h"


template <typename cuda_type>
struct nop_functor;


template <typename cuda_type>
struct scale_functor;

////////////////////////////////////////////////////////////////////////////////////////////


template <typename cuda_type, typename preprocessor>
struct hills_dynamical_system ;



////////////////////////////////////////////////////////////////////////////////////////////

template <typename cuda_type, typename dynamicalsystem>
  void runge_kutta_integrate (cuda_type  * X , cuda_type * O, cuda_type t , cuda_type dt , 
			      cuda_type max_val, cuda::kernel_dimensions * dimensions_);

template <>
void runge_kutta_integrate <float, hills_dynamical_system <float, scale_functor<float> > > (float  * X , float * O, 
											    float t , float dt , float max_val, 
											    cuda::kernel_dimensions * dimensions_);

template <>
void runge_kutta_integrate <double, hills_dynamical_system <double, scale_functor<double> > > (double  * X , double * O, double t ,
											       double dt , double max_val, 
											       cuda::kernel_dimensions * dimensions_);


////////////////////////////////////////////////////////////////////////////////////////////

template <typename ty, typename preprocessor>
void cu_compute_fitness_mindis(ty *S , ty *O, ty width, cuda::kernel_dimensions * dimensions_ );

template <>
void cu_compute_fitness_mindis<float, nop_functor<float> >(float *S , float *O, float width, cuda::kernel_dimensions * dimensions_  );
template <>
void cu_compute_fitness_mindis<double, nop_functor<double> >(double *S , double *O, double width, cuda::kernel_dimensions * dimensions_  );

#endif
