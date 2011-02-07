#ifndef __PAGMO_CUDA_KERNEL__
#define __PAGMO_CUDA_KERNEL__

#include "kernel_dims.h"
#include "common.h"

////////////////////////////////////////////////////////////////////////////////////////////


template <typename ty, typename preprocesor = nop_functor<ty>, typename pre_exec = nop_functor<ty>, typename post_exec = nop_functor<ty> >
    struct hills_dynamical_system ;

typedef hills_dynamical_system <float, nop_functor<float>, nop_functor<float>, nop_functor<float> > hills_dynamical_sys_float;
typedef hills_dynamical_system <double, nop_functor<double>, nop_functor<float>, nop_functor<double> > hills_dynamical_sys_double;



////////////////////////////////////////////////////////////////////////////////////////////

template <typename cuda_type, typename dynamicalsystem, size_t order, size_t system_params, typename o_pre_exec, typename pre_exec, typename post_exec>
    cudaError_t runge_kutta_integrate (cuda_type  * X , cuda_type * O, cuda_type t , cuda_type dt , 
				       cuda_type max_val, 
				       cuda::kernel_dimensions * dimensions_);

template <>
cudaError_t runge_kutta_integrate <float, hills_dynamical_sys_float, 6, 2, 
    scale_functor<float>, nop_functor<float>, nop_functor<float>  > 
    (float  * X , float * O, 
     float t , float dt , float max_val, 
     cuda::kernel_dimensions * dimensions_);

template <>
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double, 6, 2, scale_functor<double>, nop_functor<double>, nop_functor<double> > 
    (double  * X , double * O, double t ,
     double dt , double max_val, 
     cuda::kernel_dimensions * dimensions_);

template <>
cudaError_t runge_kutta_integrate <float, hills_dynamical_sys_float, 6, 2, 
    nop_functor<float>, nop_functor<float>, nop_functor<float>  > 
    (float  * X , float * O, 
     float t , float dt , float max_val, 
     cuda::kernel_dimensions * dimensions_);

template <>
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double, 6, 2, 
    nop_functor<double>, nop_functor<double>, nop_functor<double> > 
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


////////////////////////////////////////////////////////////////////////////////////////////
template <typename ty, typename pre_exec, typename post_exec>
cudaError_t cu_compute_fitness_mindis_theta(ty *S, ty *O, ty *F, ty *I, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_mindis_theta<float, nop_functor<float>, nop_functor<float> >(float *S, float *O, float *F, 
									       float *I, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_mindis_theta<double, nop_functor<double>, nop_functor<double> >(double *S, double *O, double *F, 
									       double *I, size_t width, cuda::kernel_dimensions * dims_ );


////////////////////////////////////////////////////////////////////////////////////////////

template <typename ty, typename pre_exec, typename post_exec>
cudaError_t cu_compute_fitness_mindis_simple(ty *S, ty *O, ty *F, ty *D, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_mindis_simple<float, nop_functor<float> , nop_functor<float> >(float *S, float *O, float *F, 
									       float *I, size_t width, cuda::kernel_dimensions * dims_ );


template <>
cudaError_t cu_compute_fitness_mindis_simple<double, nop_functor<double> , nop_functor<double> >(double *S, double *O, double *F, 
									       double *I, size_t width, cuda::kernel_dimensions * dims_ );





////////////////////////////////////////////////////////////////////////////////////////////

template <typename ty, typename pre_exec, typename post_exec>
cudaError_t cu_compute_fitness_mindis_noatt(ty *S, ty *O, ty *F, ty *I,  ty vicinity_distance, ty vicinity_speed,  ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_mindis_noatt<float, nop_functor<float>, nop_functor<float> > (float *S, float *O, float *F, float *I, float vicinity_distance, float vicinity_speed,  float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_ );


template <>
cudaError_t cu_compute_fitness_mindis_noatt<double, nop_functor<double>, nop_functor<double> > (double *S, double *O, double *F, double *I, double vicinity_distance, double vicinity_speed,  double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_ );

////////////////////////////////////////////////////////////////////////////////////////////

template <typename ty, typename pre_exec, typename post_exec>
cudaError_t cu_compute_fitness_twodee1(ty *S, ty *O, ty *F, ty *I, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_twodee1<float, nop_functor<float>, nop_functor<float> >(float *S, float *O, float *F, float *I, float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_twodee1<double, nop_functor<double>, nop_functor<double> >(double *S, double *O, double *F, double *I, double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_ );


////////////////////////////////////////////////////////////////////////////////////////////

template <typename ty, typename pre_exec, typename post_exec>
cudaError_t cu_compute_fitness_twodee2(ty *S , ty *O, ty *F, ty *I, ty vicinity_distance, ty vicinity_speed, ty vic_orientation, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_twodee2<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, float vicinity_distance, float vicinity_speed, float vic_orientation, float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_ );

template <>
cudaError_t cu_compute_fitness_twodee2<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, double vicinity_distance, double vicinity_speed, double vic_orientation, double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_ );




////////////////////////////////////////////////////////////////////////////////////////////

template <typename ty, typename pre_exec, typename post_exec>
    cudaError_t cu_compute_fitness_twodee3(ty *S, ty *O, ty* F, ty *I, ty vicinity_distance, ty vicinity_speed, ty vic_orientation, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_ ); 

template <>
cudaError_t cu_compute_fitness_twodee3<float, nop_functor<float>, nop_functor<float> >(float *S, float *O, float *F, float *I, float vicinity_distance, float vicinity_speed, float vic_orientation, float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_ ); 

template <>
cudaError_t cu_compute_fitness_twodee3<double, nop_functor<double>, nop_functor<double> >(double *S, double *O, double *F, double *I, double vicinity_distance, double vicinity_speed, double vic_orientation, double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_ ); 


////////////////////////////////////////////////////////////////////////////////////////////

#endif
