//-*-c++-*-
//Basic kernel to compute the result of a layer's operation

#ifndef __PAGMO_CUDA_KERNELS__
#define  __PAGMO_CUDA_KERNELS__

#include "stdio.h"
#include "kernel.h"
#include "common.cu"


///////////////////////////////////////////////////////////////////////////////////////////////
// computes y += alpha * x1
//Move this device code back to a struct so that we can use a template pitch
template <typename fty, size_t size>
__device__ __forceinline__ void increment(fty *Y,  fty *X,  const fty alpha) 
{
    const size_t pitch = 1;
    const size_t limits  = size * pitch;

    #pragma unroll
    for (int i=0; i<limits; i+= pitch)
    {
	Y[i] += alpha * X[i];
    }
}

// computes y = x1 - x2
template <typename fty, size_t size>
__device__ __forceinline__ void assign_diff(fty *Y,  fty *X1,  fty * X2) 
{
    const size_t pitch = 1;
    const size_t limits  = size * pitch;

    #pragma unroll
    for (int i=0; i<limits; i += pitch)
    {
	Y[i] = X1[i] - X2[i];
    }
}

// computes y = x1 + alpha * x2
template <typename fty, size_t size>
__device__ __forceinline__ void assign_sum(fty *Y,  fty *X1,  
					   fty* X2, const fty alpha) 
{
    const size_t pitch = 1;
    const size_t limits  = size * pitch;

    #pragma unroll
    for (int i=0; i<limits; i+= pitch)
    {
	Y[i] = X1[i] + alpha * X2[i];
    }
}

// computes y = alpha1 * ( x1 + x2 + beta*x3 )
template <typename fty, size_t size>
__device__ __forceinline__ void increment_sum_sum(fty *Y,  fty *X1,  fty* X2, 
						  fty* X3, const fty alpha, 
						  const fty beta) 
{
    const size_t pitch = 1;
    const size_t limits  = size * pitch;

    #pragma unroll
    for (int i=0; i<limits; i+= pitch)
    {
	Y[i] += alpha*(X1[i] + X2[i] + beta*X3[i]);
    }
}

// computes y = x1 + alpha * x2 ; x2 += x3
template <typename fty, size_t size>
__device__ __forceinline__ void assign_sum_increment(fty *Y,  fty *X1,  fty* X2, 
						     fty* X3, fty alpha) 
{
    const size_t pitch = 1;
    const size_t limits  = size * pitch;

    #pragma unroll
    for (int i=0; i<limits; i+= pitch)
    {
	Y[i] = X1[i] + alpha*X2[i];
	X2[i] += X3[i];
    }
}


///////////////////////////////////////////////////////////////////////
// line equation

template <typename fty, typename pre_exec, typename post_exec>
struct line_system 
{
    enum { size = 1 };
    enum { control_params = 0};

    __device__ void operator () (fty * S, fty*D, pre_exec pre = pre_exec(), post_exec post = post_exec() )
	{
	    fty x = pre(S[0]);
	    D[0] = post(x);    
	}
};

///////////////////////////////////////////////////////////////////////
// hills equation

template <typename fty, typename preprocessor, typename pre_exec, typename post_exec>
struct hills_dynamical_system 
{
    enum { size = 6 };
    enum { control_params = 2};

    /*typedef typename pre_exec_ pre_exec;
    typedef typename post_exec_ post_exec;
    typedef typename preprocessor_ preprocessor;*/

    __device__ void operator () (fty *S,  fty *D,  fty* O, 
				 fty t, preprocessor prec = preprocessor (),
				 pre_exec pre = pre_exec(), 
				 post_exec post = post_exec())
	{

	    const fty nu = 0.08, mR = (1.5 * 0.5);

	    const size_t inc  = 1;
	    size_t offset = inc;

	    fty x = pre(S[0]);
	    fty vx = pre(S[1]);
	    fty y = pre(S[2]);
	    fty vy = pre(S[3]);
	    fty theta = pre(S[4]);	
	    fty omega = pre(S[5]);
	
	    fty distance = sqrt(x * x + y * y);

	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;
	
	    fty ul = prec (O[0]);
	    fty ur = prec (O[1]);
       
	    D[0] = post(vx);
	    D[1] = post(2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta));
	    D[2] = post(vy);
	    D[3] = post(-2 * nu * vx + (ul + ur) * sin(theta));
	    D[4] = post(omega);
	    D[5] = (ul - ur) * 1/mR;
	}
};


/////////////////////////////////////////////////////////////////////////
// runge kutta integrator
//

//assume task size = 1
extern __shared__ char rk_shared_mem [];

//(double *, double *, double, double, double, size_t, size_t, size_t, size_t, size_t, size_t)
template <typename fty, typename DynamicSystem, size_t order, size_t system_params, typename o_pre_exec, typename pre_exec, typename post_exec>

__global__ void cu_runge_kutta_integrate (fty  * X , fty * O, const fty t , const fty dt ,  
					  const fty max_val,
					  size_t task_size,
					  size_t tasks_per_block, 
					  size_t individuals, size_t points, 
					  DynamicSystem system = DynamicSystem(),
					  pre_exec pre = pre_exec(),
					  post_exec post = post_exec())
{



    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);

    //0. load shared memory with inputs and outputs. 
    size_t step = BLOCK_TASKS(tasks_per_block) * order;
    size_t offset = BLOCK_TASKS(tasks_per_block) * system_params;

    fty * Os = (fty *) rk_shared_mem; 

    fty * Xs = & ((fty *) rk_shared_mem)[offset];// inputs come after the weights
    offset += step;
    fty * DxDt = & ((fty *) rk_shared_mem)[offset];
    offset += step;
    fty * Xt = & ((fty *) rk_shared_mem)[offset];
    offset += step;
    fty * Dxt = & ((fty *) rk_shared_mem)[offset];
    offset += step;
    fty * Dxm = & ((fty *) rk_shared_mem)[offset];

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);
    size_t individ = GLOBAL_INDIV_ID(tasks_per_block, task_size, points);



    copy_to_shared_mem1<fty, o_pre_exec>(Os, O, BLOCK_TASKS(tasks_per_block) * system_params, max_val);
    copy_to_shared_mem<fty, pre_exec> (Xs, X, step);    // probably verified

    __syncthreads();


    //<TODO> handle last block
    int idx = BLOCK_ADJUSTED_TX(tasks_per_block);//GLOBAL_RAW_TX();
    int ridx = GLOBAL_RAW_TX();

    if (IS_VALID_FOR_BLOCK(tasks_per_block,task_size,points))
    {

	const fty  dh = fty( 0.5 ) * dt;
	const fty th = t + dh;
	const fty val2 = fty( 2.0 );

	size_t inc1 = order*idx;
	size_t inc2 = system_params*idx;

	// k1
	system(&Xs[inc1], &DxDt[inc1], &Os[inc2], t);
	assign_sum<fty, order>( &Xt[inc1] , &Xs[inc1] , &DxDt[inc1] , dh );
	
	//k2
	system( &Xt[inc1] , &Dxt[inc1] , &Os[inc2], th );
	assign_sum<fty, order>( &Xt[inc1], &Xs[inc1], &Dxt[inc1] , dh );


	//k3
	system( &Xt[inc1] , &Dxm[inc1] , &Os[inc2], th );
	assign_sum_increment<fty, order>( &Xt[inc1], &Xs[inc1], &Dxm[inc1], &Dxt[inc1], dt );
	
	//k4
	system( &Xt[inc1] , &Dxt[inc1] , &Os[inc2], fty( t + dt ));

	increment_sum_sum<fty, order>( &Xs[inc1], &DxDt[inc1], &Dxt[inc1],  &Dxm[inc1], dt /  fty( 6.0 ) , val2 );
    }
    __syncthreads();

    copy_to_global_mem<fty, post_exec> (X, Xs, step);

    __syncthreads();

}


static void print_parameters(const char * name, cuda::kernel_dimensions * dims_)
{
/*    printf("%s with \n grid size = %d\n block size = %d\n shared mem = %d\n tasks per block = %d\n", 
	   name, dims_->get_grid_dims().x, dims_->get_block_dims().x, dims_->get_shared_mem_size(), dims_->get_tasks_per_block());
    printf("individuals = %d\n points = %d\n task size = %d\n ", 
    dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());*/

}

//////////////////////////////////////////////////////////////////////////////////////////////
// kernel interface functions


template <typename fty, typename dynamicalsystem, size_t order, size_t system_params, typename o_pre_exec, typename pre_exec, typename post_exec>
cudaError_t runge_kutta_integrate (fty  * X , fty * O, fty t , fty dt , fty max_val, cuda::kernel_dimensions * dims_)
{
    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <fty, dynamicalsystem, order, system_params, pre_exec, post_exec >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X , O, t , dt, max_val, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());

    cudaThreadSynchronize();
    return cudaGetLastError();
} 

template <>
cudaError_t runge_kutta_integrate <float, hills_dynamical_sys_float , 7, 2, 
				   scale_functor<float>, nop_functor<float> , nop_functor<float> >
                                                                                           (float  * X , float * O, float t , float dt , 
											    float max_val, 
											    cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <float, hills_dynamical_sys_float ,  7, 2, scale_functor<float>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X , O, t , dt, max_val,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());

    cudaThreadSynchronize();
    return cudaGetLastError();
} 


template <>
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double , 7, 2, 
				   scale_functor<double>, nop_functor<double>, nop_functor<double> > (double  * X , double * O, double t , double dt , 
												      double max_val, 
												      cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <double, hills_dynamical_sys_double ,7, 2, scale_functor<double>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X , O, t , dt, max_val, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());

    cudaThreadSynchronize();
    return cudaGetLastError();
} 


template <>
cudaError_t runge_kutta_integrate <float, hills_dynamical_sys_float , 7, 2, 
				   nop_functor<float>, nop_functor<float> , nop_functor<float> >
                                                                                           (float  * X , float * O, float t , float dt , 
											    float max_val, 
											    cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate3n", dims_);
    cu_runge_kutta_integrate <float, hills_dynamical_sys_float ,  7, 2, nop_functor<float>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X , O, t , dt, max_val,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());

    cudaThreadSynchronize();
    return cudaGetLastError();
} 


template <>
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double , 7, 2, 
				   nop_functor<double>, nop_functor<double>, nop_functor<double> > (double  * X , double * O, double t , double dt , 
												      double max_val, 
												      cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <double, hills_dynamical_sys_double ,7, 2, nop_functor<double>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X , O, t , dt, max_val, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());

    cudaThreadSynchronize();
    return cudaGetLastError();
} 


///////////////////////////////////////////////////////////////////////
// fitness kernels

extern __shared__ char rk_mindis_kernel_mem [];
template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_mindis_kernel(ty *S , ty *O, ty *F, ty *I, size_t width, 
						 size_t task_size,
						 size_t tasks_per_block, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);

    //0. load shared memory with inputs and outputs. 
    

    ty * Ss = (ty *) rk_mindis_kernel_mem; 

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);

    size_t offset = BLOCK_ADJUSTED_TX(tasks_per_block)*width;

    copy_to_shared_mem<ty, pre_exec>(Ss, S, block_points * width);

    __syncthreads();

    if (IS_VALID_FOR_BLOCK(tasks_per_block, task_size, points))
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)

	F[globaloffset*4] =  1/( 1 + distance );
	F[globaloffset*4 + 1] =   distance;
	F[globaloffset*4 + 2] =  speed;
	F[globaloffset*4 + 3] =  Ss[offset + 4];//theta
    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis(ty *S , ty *O, ty *F, ty *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_mindis<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

/////////////////////////////////////////////////////////////////////////////



extern __shared__ char mindis_theta_mem [];
template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_mindis_theta_kernel(ty *S , ty *O, ty *F, ty *I, size_t width, 
						 size_t task_size,
						 size_t tasks_per_block, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);


    //0. load shared memory with inputs and outputs. 
    

    ty * Ss = (ty *) mindis_theta_mem; 

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);

    size_t offset = BLOCK_ADJUSTED_TX(tasks_per_block)*width;

    copy_to_shared_mem<ty, pre_exec>(Ss, S, block_points * width);

    __syncthreads();


    //<TODO> handle last block
    if (IS_VALID_FOR_BLOCK(tasks_per_block, task_size, points))
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)
	ty theta = Ss[offset + 4];
	// keep theta between -180 and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;	

	F[globaloffset*4] =  1/( (1+distance) * (1+speed) * (1+fabs(theta)) );
	F[globaloffset*4 + 1] =   distance;
	F[globaloffset*4 + 2] =  speed;
	F[globaloffset*4 + 3] =  Ss[offset + 4];

    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis_theta(ty *S , ty *O, ty *F, ty *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_theta_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis_theta<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_theta_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_mindis_theta<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_theta_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

/////////////////////////////////////////////////////////////////////////////

extern __shared__ char fitness_mindis_simple_mem [];
template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_mindis_simple_kernel(ty *S , ty *O, ty *F, ty *I, size_t width, 
						 size_t task_size,
						 size_t tasks_per_block, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);


    //0. load shared memory with inputs and outputs. 
    

    ty * Ss = (ty *) mindis_theta_mem; 

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);

    size_t offset = BLOCK_ADJUSTED_TX(tasks_per_block)*width;

    copy_to_shared_mem<ty, pre_exec>(Ss, S, block_points * width);

    __syncthreads();


    //<TODO> handle last block
    if (IS_VALID_FOR_BLOCK(tasks_per_block, task_size, points))
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)
	ty theta = Ss[offset + 4];
	// keep theta between -180 and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;	

	F[globaloffset*4] =  0.0;
	if(distance < I[globaloffset]) 
	{
	    F[globaloffset*4] = 1/( (1+distance));	
	}
	F[globaloffset*4 + 1] =   distance;
	F[globaloffset*4 + 2] =  speed;
	F[globaloffset*4 + 3] =  Ss[offset + 4];

    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis_simple(ty *S , ty *O, ty *F, ty *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_simple_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis_simple<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_simple_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_mindis_simple<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_simple_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


/////////////////////////////////////////////////////////////////////////////

extern __shared__ char fitness_mindis_noatt_mem [];
template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_mindis_noatt_kernel(ty *S , ty *O, ty *F, ty *I, ty vicinity_distance, ty vicinity_speed,  
					    ty max_docking_time, ty tdt, size_t width, 
						 size_t task_size,
						 size_t tasks_per_block, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);


    //0. load shared memory with inputs and outputs. 
    

    ty * Ss = (ty *) mindis_theta_mem; 

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);

    size_t offset = BLOCK_ADJUSTED_TX(tasks_per_block)*width;

    copy_to_shared_mem<ty, pre_exec>(Ss, S, block_points * width);

    __syncthreads();


    //<TODO> handle last block
    if (IS_VALID_FOR_BLOCK(tasks_per_block, task_size, points))
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)
	ty theta = Ss[offset + 4];
	// keep theta between -180 and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;	

	F[globaloffset*4] =  0.0;
	if(distance < I[globaloffset] / 2) 
	{
	    F[globaloffset*4] = 1.0/((1 + distance) * (1 + speed));				
	    if(distance < vicinity_distance && speed < 0.1)
		F[globaloffset*4] += F[globaloffset*4] * (max_docking_time - tdt)/max_docking_time;
	}
	F[globaloffset*4 + 1] =   distance;
	F[globaloffset*4 + 2] =  speed;
	F[globaloffset*4 + 3] =  Ss[offset + 4];

    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis_noatt(ty *S , ty *O, ty *F, ty *I, ty vicinity_distance, ty vicinity_speed,  
					    ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_noatt_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, 
	 vicinity_distance, vicinity_speed, max_docking_time, tdt, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis_noatt<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, float vicinity_distance, float vicinity_speed,  float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_noatt_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, 
	 vicinity_distance, vicinity_speed,  max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>

cudaError_t cu_compute_fitness_mindis_noatt<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, double vicinity_distance, double vicinity_speed,  double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_noatt_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, vicinity_distance, vicinity_speed,  
	 max_docking_time,  tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

/////////////////////////////////////////////////////////////////////////////

extern __shared__ char fitness_twodee1_mem [];
template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_twodee1_kernel(ty *S , ty *O, ty *F, ty *I, ty max_docking_time, ty tdt, size_t width, 
						 size_t task_size,
						 size_t tasks_per_block, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);


    //0. load shared memory with inputs and outputs. 
    

    ty * Ss = (ty *) mindis_theta_mem; 

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);

    size_t offset = BLOCK_ADJUSTED_TX(tasks_per_block)*width;

    copy_to_shared_mem<ty, pre_exec>(Ss, S, block_points * width);

    __syncthreads();


    //<TODO> handle last block
    if (IS_VALID_FOR_BLOCK(tasks_per_block, task_size, points))
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)
	ty theta = Ss[offset + 4];
	// keep theta between -180 and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;	

	ty timeBonus = (max_docking_time - tdt)/max_docking_time;
	F[globaloffset*4] = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
	if (I[globaloffset] > distance/2) 
	{
	    if(F[globaloffset*4] > 0.87)
		F[globaloffset*4] += F[globaloffset*4] * timeBonus;	
	} 
	else
	    F[globaloffset*4] = 0;
	F[globaloffset*4 + 1] =   distance;
	F[globaloffset*4 + 2] =  speed;
	F[globaloffset*4 + 3] =  Ss[offset + 4];

    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_twodee1(ty *S , ty *O, ty *F, ty *I, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee1_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, 
	 max_docking_time, tdt, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_twodee1<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee1_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, 
	 max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_twodee1<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee1_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, 
	 max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

/////////////////////////////////////////////////////////////////////////////

extern __shared__ char compute_fitness_twodee2_mem [];
template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_twodee2_kernel(ty *S , ty *O, ty *F, ty *I, 	 ty vicinity_distance, ty vicinity_speed, 
						  ty vicinity_orientation, ty max_docking_time, ty tdt, size_t width, 
						 size_t task_size,
						 size_t tasks_per_block, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);


    //0. load shared memory with inputs and outputs. 
    

    ty * Ss = (ty *) mindis_theta_mem; 

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);

    size_t offset = BLOCK_ADJUSTED_TX(tasks_per_block)*width;

    copy_to_shared_mem<ty, pre_exec>(Ss, S, block_points * width);

    __syncthreads();


    //<TODO> handle last block
    if (IS_VALID_FOR_BLOCK(tasks_per_block, task_size, points))
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)
	ty theta = Ss[offset + 4];
	// keep theta between -180 and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;	

	ty timeBonus = (max_docking_time - tdt)/max_docking_time;
	ty alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
	if (I[globaloffset] > distance/2) {
	    if (distance < vicinity_distance && fabs(theta) < vicinity_orientation && speed < vicinity_speed)
		F[globaloffset*4] = alpha + alpha * timeBonus;	
	    else
		F[globaloffset*4] = alpha;
	} else
	    F[globaloffset*4] = 0;

	F[globaloffset*4 + 1] =   distance;
	F[globaloffset*4 + 2] =  speed;
	F[globaloffset*4 + 3] =  Ss[offset + 4];

    }
    __syncthreads();
}

template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_twodee2(ty *S , ty *O, ty *F, ty *I, ty vicinity_distance, ty vicinity_speed, ty vic_orientation, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee2_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, 
	 vicinity_distance, vicinity_speed, 
	 vic_orientation, max_docking_time, tdt, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_twodee2<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, float vicinity_distance, float vicinity_speed, float vic_orientation, float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee2_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, 
	 vicinity_distance, vicinity_speed, 
	 vic_orientation, max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_twodee2<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, double vicinity_distance, 
											  double vicinity_speed, double vic_orientation, double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee2_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, vicinity_distance, vicinity_speed, 
	 vic_orientation, max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

/////////////////////////////////////////////////////////////////////////////




extern __shared__ char fitness_twodee3_kernel_mem [];
template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_twodee3_kernel(ty *S , ty *O, ty *F, ty *I, 
						  ty vicinity_distance, ty vicinity_speed, 
						  ty vicinity_orientation, ty max_docking_time, ty tdt, 
						  size_t width, size_t task_size,
						  size_t tasks_per_block, 
						  size_t individuals, size_t points, 
						  pre_exec prep = pre_exec(), 
						  post_exec post = post_exec() )
{

    size_t block_points = BLOCK_POINTS(tasks_per_block, task_size);


    //0. load shared memory with inputs and outputs. 
    

    ty * Ss = (ty *) mindis_theta_mem; 

    size_t globaloffset = GLOBAL_ADJUSTED_TX(tasks_per_block);

    size_t offset = BLOCK_ADJUSTED_TX(tasks_per_block)*width;

    copy_to_shared_mem<ty, pre_exec>(Ss, S, block_points * width);

    __syncthreads();


    //<TODO> handle last block
    if (IS_VALID_FOR_BLOCK(tasks_per_block, task_size, points))
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)
	ty theta = Ss[offset + 4];
	// keep theta between -180 and +180°
	if(theta < -M_PI) theta += 2 * M_PI;
	if(theta > M_PI) theta -= 2 * M_PI;	

	ty timeBonus = (max_docking_time - tdt)/max_docking_time;
	ty alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
	if (I[globaloffset] > distance/2) {
	    if (distance < vicinity_distance && fabs(theta) < vicinity_orientation && speed < vicinity_speed)
		F[globaloffset*4] = 1 + timeBonus;	
	    else
		F[globaloffset*4] = alpha;
	} else
	    F[globaloffset*4] = 0;

	F[globaloffset*4 + 1] =   distance;
	F[globaloffset*4 + 2] =  speed;
	F[globaloffset*4 + 3] =  Ss[offset + 4];

    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_twodee3(ty *S , ty *O, ty *F, ty *I, ty vicinity_distance, ty vicinity_speed, 
				       ty vic_orientation, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee3_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, 
	 vicinity_distance, vicinity_speed,vic_orientation, 
	 max_docking_time, tdt, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_twodee3<float, nop_functor<float>, nop_functor<float> >(float *S , float *O, float *F, float *I, float vicinity_distance, 
											    float vicinity_speed, float vic_orientation, float max_docking_time, 
											    float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee3_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, 
	 vicinity_distance, vicinity_speed,vic_orientation, 
	 max_docking_time, tdt, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_twodee3<double, nop_functor<double>, nop_functor<double> >(double *S , double *O, double *F, double *I, 
											       double vicinity_distance,double vicinity_speed, 
											       double vic_orientation, double max_docking_time, 
											       double tdt,size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_twodee3_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(S , O, F, I, 
	 vicinity_distance, vicinity_speed,vic_orientation, 
	 max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}



/////////////////////////////////////////////////////////////////////////////
//Misc kernels




#endif 

