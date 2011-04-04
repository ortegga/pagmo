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
__device__ __forceinline__ void increment(fty *Y,  fty *X,  const fty alpha, const size_t pitch = 1) 
{
    const size_t limits  = size * pitch;

    #pragma unroll
    for (int i=0; i<limits; i+= pitch)
    {
	Y[i] += alpha * X[i];
    }
}

// computes y = x1 - x2
template <typename fty, size_t size>
__device__ __forceinline__ void assign_diff(fty *Y,  fty *X1,  fty * X2, const size_t pitch = 1) 
{
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
					   fty* X2, const fty alpha, const size_t pitch = 1 ) 
{
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
						  const fty beta, const size_t pitch = 1) 
{
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
						     fty* X3, fty alpha, const size_t pitch = 1) 
{
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

    __device__ void operator () (fty *S,  fty *D,  fty* O, 
				 fty t, const size_t pitch = 1, preprocessor prec = preprocessor (),
				 pre_exec pre = pre_exec(), 
				 post_exec post = post_exec())
	{

	    const fty nu = 0.08, mR = (1.5 * 0.5);
	    size_t inc = 0;
	    fty x = pre(S[0]);
	    inc += pitch;
	    fty vx = pre(S[inc]);
	    inc += pitch; 
	    fty y = pre(S[inc]);
	    inc += pitch; 
	    fty vy = pre(S[inc]);
	    inc += pitch; 
	    fty theta = pre(S[inc]);	
	    inc += pitch; 
	    fty omega = pre(S[inc]);
	
	    fty distance = sqrt(x * x + y * y);

	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;
	
	    fty ul = prec (O[0]);
	    fty ur = prec (O[pitch]);
       
	    inc = 0;
	    D[0] = post(vx);
	    inc += pitch; 
	    D[inc] = post(2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta));
	    inc += pitch; 
	    D[inc] = post(vy);
	    inc += pitch; 
	    D[inc] = post(-2 * nu * vx + (ul + ur) * sin(theta));
	    inc += pitch; 
	    D[inc] = post(omega);
	    inc += pitch; 
	    D[inc] = (ul - ur) * 1/mR;
	}
};


/////////////////////////////////////////////////////////////////////////
// runge kutta integrator
//

//assume task size = 1
extern __shared__ char rk_shared_mem [];

//(double *, double *, double, double, double, size_t, size_t, size_t, size_t, size_t, size_t)
template <typename fty, typename DynamicSystem, size_t order, size_t system_params, typename o_pre_exec, typename pre_exec, typename post_exec>

__global__ void cu_runge_kutta_integrate (cudaPitchedPtr X , cudaPitchedPtr O, const fty t , const fty dt ,  
					  const fty max_val,
					  size_t task_size,
					  size_t block_points, 
					  size_t individuals, size_t points, 
					  DynamicSystem system = DynamicSystem(),
					  pre_exec pre = pre_exec(),
					  post_exec post = post_exec())
{



    //0. load shared memory with inputs and outputs. 
    size_t step = block_points * order;
    size_t offset = block_points * system_params;

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

    copy_to_shared_mem1<fty, o_pre_exec>(Os, O.ptr, O.pitch, O.ysize, block_points, max_val);
    copy_to_shared_mem<fty, pre_exec> (Xs, X.ptr, X.pitch, X.ysize, block_points);

    __syncthreads();


    for(int i=0; i < block_points; i+= blockDim.x)// task id
    {
	size_t taskid = i + threadIdx.x;//block's task id
	if (taskid  < block_points)
	{
	    const fty  dh = fty( 0.5 ) * dt;
	    const fty th = t + dh;
	    const fty val2 = fty( 2.0 );

	    // k1
	    system(&Xs[taskid], &DxDt[taskid], &Os[taskid], t, block_points);
	    assign_sum<fty, order>( &Xt[taskid], &Xs[taskid], &DxDt[taskid], dh, block_points );
	
	    //k2
	    system( &Xt[taskid] , &Dxt[taskid] , &Os[taskid], th, block_points );
	    assign_sum<fty, order>( &Xt[taskid], &Xs[taskid], &Dxt[taskid], dh, block_points );


	    //k3
	    system( &Xt[taskid] , &Dxm[taskid] , &Os[taskid], th, block_points);
	    assign_sum_increment<fty, order>( &Xt[taskid], &Xs[taskid], &Dxm[taskid], &Dxt[taskid], dt, block_points);
	
	    //k4
	    system( &Xt[taskid] , &Dxt[taskid] , &Os[taskid], fty( t + dt ), block_points);

	    increment_sum_sum<fty, order>( &Xs[taskid], &DxDt[taskid], &Dxt[taskid],  &Dxm[taskid], dt /  fty( 6.0 ), val2, block_points);
	}
    }

    copy_to_global_mem<fty, post_exec> (X.ptr, X.pitch, X.ysize, Xs, block_points);

    __syncthreads();

}


static void print_parameters(const char * name, cuda::kernel_dimensions * dims_)
{
    /*printf("%s with \n grid size = %d\n block size = %d\n shared mem = %d\n tasks per block = %d\n", 
	   name, dims_->get_grid_dims().x, dims_->get_block_dims().x, dims_->get_shared_mem_size(), dims_->get_tasks_per_block());
    printf("individuals = %d\n points = %d\n task size = %d\n ", 
    dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());*/

}

//////////////////////////////////////////////////////////////////////////////////////////////
// kernel interface functions


template <typename fty, typename dynamicalsystem, size_t order, size_t system_params, typename o_pre_exec, typename pre_exec, typename post_exec>
cudaError_t runge_kutta_integrate (cudaPitchedPtr * X , cudaPitchedPtr * O, fty t , fty dt , fty max_val, cuda::kernel_dimensions * dims_)
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
                                                                                           (cudaPitchedPtr * X , cudaPitchedPtr * O, float t , float dt , 
											    float max_val, 
											    cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <float, hills_dynamical_sys_float ,  7, 2, scale_functor<float>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X , *O, t , dt, max_val,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());

    cudaThreadSynchronize();
    return cudaGetLastError();
} 


template <>
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double , 7, 2, 
				   scale_functor<double>, nop_functor<double>, nop_functor<double> > (cudaPitchedPtr * X , cudaPitchedPtr * O, double t , double dt , 
												      double max_val, 
												      cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <double, hills_dynamical_sys_double ,7, 2, scale_functor<double>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X , *O, t , dt, max_val, 
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
                                                                                           (cudaPitchedPtr * X , cudaPitchedPtr * O, float t , float dt , 
											    float max_val, 
											    cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <float, hills_dynamical_sys_float ,  7, 2, nop_functor<float>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X , *O, t , dt, max_val,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());

    cudaThreadSynchronize();
    return cudaGetLastError();
} 


template <>
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double , 7, 2, 
				   nop_functor<double>, nop_functor<double>, nop_functor<double> > (cudaPitchedPtr * X , cudaPitchedPtr * O, double t , double dt , 
												      double max_val, 
												      cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <double, hills_dynamical_sys_double ,7, 2, nop_functor<double>, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X , *O, t , dt, max_val, 
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
__global__ void cu_compute_fitness_mindis_kernel(cudaPitchedPtr S , cudaPitchedPtr O, cudaPitchedPtr F, cudaPitchedPtr I, size_t width, 
						 size_t task_size,
						 size_t block_points, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{


    //0. load shared memory with inputs and outputs. 

    ty * Ss = (ty *) rk_mindis_kernel_mem; 

    copy_to_shared_mem<ty, pre_exec>(Ss, S.ptr, S.pitch, S.ysize, block_points);

    __syncthreads();

    for (int i=0; i < block_points; i+= blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < block_points)
	{
	    ty distance = sqrt(Ss[col] * Ss[col] + Ss[2*block_points + col] * Ss[2*block_points + col]);
	    ty speed    = sqrt(Ss[block_points + col] * Ss[block_points + col] + Ss[3*block_points + col] * Ss[3*block_points + col]);		// sqrt(vx^2 + vy^2)

	    ty * Fr = (ty *)F.ptr;
	    size_t row_size = F.pitch / sizeof(ty);
	    size_t offset =block_points*blockIdx.x + col; 
	    
	    Fr[             offset] =  1/( 1 + distance );
	    Fr[row_size  +  offset] =  distance;
	    Fr[2*row_size + offset] =  speed;
	    Fr[3*row_size + offset] =  Ss[4*block_points + threadIdx.x];//theta
	}
	
    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis", dims_);
    cu_compute_fitness_mindis_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(*S, *O, F, I, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis<float, nop_functor<float>, nop_functor<float> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis", dims_);
    cu_compute_fitness_mindis_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_mindis<double, nop_functor<double>, nop_functor<double> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis", dims_);
    cu_compute_fitness_mindis_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, width,
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
__global__ void cu_compute_fitness_mindis_theta_kernel(cudaPitchedPtr S, cudaPitchedPtr O, cudaPitchedPtr F, cudaPitchedPtr I, size_t width, 
						 size_t task_size,
						 size_t block_points, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{


    //0. load shared memory with inputs and outputs. 

    ty * Ss = (ty *) mindis_theta_mem; 

    copy_to_shared_mem<ty, pre_exec>(Ss, S.ptr, S.pitch, S.ysize, block_points);

    __syncthreads();

    for (int i=0; i < block_points; i+= blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < block_points)
	{
	    ty distance = sqrt(Ss[col] * Ss[col] + Ss[2*block_points + col] * Ss[2*block_points + col]);
	    ty speed    = sqrt(Ss[block_points + col] * Ss[block_points + col] + Ss[3*block_points + col] * Ss[3*block_points + col]);		// sqrt(vx^2 + vy^2)
	    ty theta = Ss[4*block_points + col];
	    
	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;	

	    ty * Fr = (ty *)F.ptr;
	    size_t row_size = F.pitch / sizeof(ty);
	    size_t offset =block_points*blockIdx.x + col; 
	    
	    Fr[             offset] =  1/( (1+distance) * (1+speed) * (1+fabs(theta)) );
	    Fr[row_size  +  offset] =  distance;
	    Fr[2*row_size + offset] =  speed;
	    Fr[3*row_size + offset] =  Ss[4*block_points + threadIdx.x];//theta
	}
	
    }
    __syncthreads();

}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis_theta(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_theta", dims_);
    cu_compute_fitness_mindis_theta_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(*S , *O, *F, *I, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis_theta<float, nop_functor<float>, nop_functor<float> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_theta", dims_);
    cu_compute_fitness_mindis_theta_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_mindis_theta<double, nop_functor<double>, nop_functor<double> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_theta", dims_);
    cu_compute_fitness_mindis_theta_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, width,
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
__global__ void cu_compute_fitness_mindis_simple_kernel(cudaPitchedPtr S , cudaPitchedPtr O, cudaPitchedPtr F, cudaPitchedPtr I, size_t width, 
						 size_t task_size,
						 size_t block_points, 
						 size_t individuals, size_t points, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    //0. load shared memory with inputs and outputs. 

    ty * Ss = (ty *) fitness_mindis_simple_mem; 
    ty * Ir = (ty * ) I.ptr;

    copy_to_shared_mem<ty, pre_exec>(Ss, S.ptr, S.pitch, S.ysize, block_points);

    __syncthreads();

    for (int i=0; i < block_points; i+= blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < block_points)
	{
	    ty distance = sqrt(Ss[col] * Ss[col] + Ss[2*block_points + col] * Ss[2*block_points + col]);
	    ty speed    = sqrt(Ss[block_points + col] * Ss[block_points + col] + Ss[3*block_points + col] * Ss[3*block_points + col]);		// sqrt(vx^2 + vy^2)
	    ty theta = Ss[4*block_points + col];
	    
	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;	

	    ty * Fr = (ty *)F.ptr;
	    size_t row_size = F.pitch / sizeof(ty);
	    size_t offset =block_points*blockIdx.x + col; 
	    
	    Fr[             offset] =  0.0;
	    if (distance < Ir[offset] )
	    {
		Fr[ offset] = 1/( (1+distance) * (1+speed) * (1+fabs(theta)) );
	    }

	    Fr[row_size  +  offset] =  distance;
	    Fr[2*row_size + offset] =  speed;
	    Fr[3*row_size + offset] =  Ss[4*block_points + threadIdx.x];//theta
	}
	
    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis_simple(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_simple", dims_);
    cu_compute_fitness_mindis_simple_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(*S , *O, *F, *I, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis_simple<float, nop_functor<float>, nop_functor<float> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_simple", dims_);
    cu_compute_fitness_mindis_simple_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_mindis_simple<double, nop_functor<double>, nop_functor<double> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_simple", dims_);
    cu_compute_fitness_mindis_simple_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, width,
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
__global__ void cu_compute_fitness_mindis_noatt_kernel(cudaPitchedPtr S , cudaPitchedPtr O, cudaPitchedPtr F, cudaPitchedPtr I, 
						       ty vicinity_distance, ty vicinity_speed,  
						       ty max_docking_time, ty tdt, size_t width, 
						       size_t task_size,
						       size_t block_points, 
						       size_t individuals, size_t points, 
						       pre_exec prep = pre_exec(), 
						       post_exec post = post_exec() )
{


    //0. load shared memory with inputs and outputs. 

    ty * Ss = (ty *) fitness_mindis_noatt_mem; 
    ty * Ir = (ty * ) I.ptr;

    copy_to_shared_mem<ty, pre_exec>(Ss, S.ptr, S.pitch, S.ysize, block_points);

    __syncthreads();

    for (int i=0; i < block_points; i+= blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < block_points)
	{
	    ty distance = sqrt(Ss[col] * Ss[col] + Ss[2*block_points + col] * Ss[2*block_points + col]);
	    ty speed    = sqrt(Ss[block_points + col] * Ss[block_points + col] + Ss[3*block_points + col] * Ss[3*block_points + col]);		// sqrt(vx^2 + vy^2)
	    ty theta = Ss[4*block_points + col];
	    
	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;	

	    ty * Fr = (ty *)F.ptr;
	    size_t row_size = F.pitch / sizeof(ty);
	    size_t offset =block_points*blockIdx.x + col; 
	    
	    Fr[             offset] =  0.0;
	    if (distance < Ir[offset] )
	    {
		Fr[ offset] = 1.0/((1 + distance) * (1 + speed));
		if(distance < vicinity_distance && speed < 0.1)
		    Fr[offset] += Fr[offset] * (max_docking_time - tdt)/max_docking_time;
	    }

	    Fr[row_size  +  offset] =  distance;
	    Fr[2*row_size + offset] =  speed;
	    Fr[3*row_size + offset] =  Ss[4*block_points + threadIdx.x];//theta
	}
	
    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis_noatt(cudaPitchedPtr *S, cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, ty vicinity_distance, ty vicinity_speed,  
					    ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_noatt", dims_);
    cu_compute_fitness_mindis_noatt_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(*S, *O, *F, *I, 
	 vicinity_distance, vicinity_speed, max_docking_time, tdt, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_mindis_noatt<float, nop_functor<float>, nop_functor<float> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, float vicinity_distance, float vicinity_speed,  float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_noatt", dims_);
    cu_compute_fitness_mindis_noatt_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, 
	 vicinity_distance, vicinity_speed,  max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>

cudaError_t cu_compute_fitness_mindis_noatt<double, nop_functor<double>, nop_functor<double> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, double vicinity_distance, double vicinity_speed,  double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_mindis_noatt", dims_);
    cu_compute_fitness_mindis_noatt_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, vicinity_distance, vicinity_speed,  
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
__global__ void cu_compute_fitness_twodee1_kernel(cudaPitchedPtr S , cudaPitchedPtr O, cudaPitchedPtr F, cudaPitchedPtr I, 
						  ty max_docking_time, ty tdt, size_t width, 
						  size_t task_size,
						  size_t block_points, 
						  size_t individuals, size_t points, 
						  pre_exec prep = pre_exec(), 
						  post_exec post = post_exec() )
{

    //0. load shared memory with inputs and outputs. 

    ty * Ss = (ty *) fitness_twodee1_mem; 
    ty * Ir = (ty * ) I.ptr;

    copy_to_shared_mem<ty, pre_exec>(Ss, S.ptr, S.pitch, S.ysize, block_points);

    __syncthreads();

    for (int i=0; i < block_points; i+= blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < block_points)
	{
	    ty distance = sqrt(Ss[col] * Ss[col] + Ss[2*block_points + col] * Ss[2*block_points + col]);
	    ty speed    = sqrt(Ss[block_points + col] * Ss[block_points + col] + Ss[3*block_points + col] * Ss[3*block_points + col]);		// sqrt(vx^2 + vy^2)
	    ty theta = Ss[4*block_points + col];
	    
	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;	

	    ty * Fr = (ty *)F.ptr;
	    size_t row_size = F.pitch / sizeof(ty);
	    size_t offset =block_points*blockIdx.x + col; 
	    ty timeBonus = (max_docking_time - tdt)/max_docking_time;
	    Fr[offset] = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
	    if (Ir[offset] > distance/2) 
	    {
		if(Fr[offset] > 0.87)
		    Fr[offset] += Fr[offset] * timeBonus;	
	    } 
	    else
		Fr[offset] = 0;

	    Fr[row_size  +  offset] =  distance;
	    Fr[2*row_size + offset] =  speed;
	    Fr[3*row_size + offset] =  Ss[4*block_points + threadIdx.x];//theta
	}
	
    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_twodee1(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee1", dims_);
    cu_compute_fitness_twodee1_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(*S, *O, *F, *I, 
	 max_docking_time, tdt, width, 
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_fitness_twodee1<float, nop_functor<float>, nop_functor<float> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee1", dims_);
    cu_compute_fitness_twodee1_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, 
	 max_docking_time, tdt, width,
	 dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}


template <>
cudaError_t cu_compute_fitness_twodee1<double, nop_functor<double>, nop_functor<double> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee1", dims_);
    cu_compute_fitness_twodee1_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, 
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
__global__ void cu_compute_fitness_twodee2_kernel(cudaPitchedPtr S, cudaPitchedPtr O, cudaPitchedPtr F, cudaPitchedPtr I,
						  ty vicinity_distance, ty vicinity_speed, 
						  ty vicinity_orientation, ty max_docking_time, ty tdt, size_t width, 
						  size_t task_size,
						  size_t block_points, 
						  size_t individuals, size_t points, 
						  pre_exec prep = pre_exec(), 
						  post_exec post = post_exec() )
{

    //0. load shared memory with inputs and outputs. 

    ty * Ss = (ty *) compute_fitness_twodee2_mem; 
    ty * Ir = (ty * ) I.ptr;

    copy_to_shared_mem<ty, pre_exec>(Ss, S.ptr, S.pitch, S.ysize, block_points);

    __syncthreads();

    for (int i=0; i < block_points; i+= blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < block_points)
	{
	    ty distance = sqrt(Ss[col] * Ss[col] + Ss[2*block_points + col] * Ss[2*block_points + col]);
	    ty speed    = sqrt(Ss[block_points + col] * Ss[block_points + col] + Ss[3*block_points + col] * Ss[3*block_points + col]);		// sqrt(vx^2 + vy^2)
	    ty theta = Ss[4*block_points + col];
	    
	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;	

	    ty * Fr = (ty *)F.ptr;
	    size_t row_size = F.pitch / sizeof(ty);
	    size_t offset =block_points*blockIdx.x + col; 
	    ty timeBonus = (max_docking_time - tdt)/max_docking_time;
	    ty alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));

	    if (Ir[offset] > distance/2) 
	    {
		if (distance < vicinity_distance && fabs(theta) < vicinity_orientation && speed < vicinity_speed)
		    Fr[offset] = alpha + alpha * timeBonus;	
		else
		    Fr[offset] = alpha;
	    } 
	    else
		Fr[offset] = 0;
	    Fr[row_size  +  offset] =  distance;
	    Fr[2*row_size + offset] =  speed;
	    Fr[3*row_size + offset] =  Ss[4*block_points + threadIdx.x];//theta
	}
	
    }
    __syncthreads();
}

template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_twodee2(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, ty vicinity_distance, ty vicinity_speed, ty vic_orientation, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee2", dims_);
    cu_compute_fitness_twodee2_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(*S, *O, *F, *I, 
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
cudaError_t cu_compute_fitness_twodee2<float, nop_functor<float>, nop_functor<float> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, float vicinity_distance, float vicinity_speed, float vic_orientation, float max_docking_time, float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee2", dims_);
    cu_compute_fitness_twodee2_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, 
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
cudaError_t cu_compute_fitness_twodee2<double, nop_functor<double>, nop_functor<double> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, double vicinity_distance, 
											  double vicinity_speed, double vic_orientation, double max_docking_time, double tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee2", dims_);
    cu_compute_fitness_twodee2_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, vicinity_distance, vicinity_speed, 
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
__global__ void cu_compute_fitness_twodee3_kernel(cudaPitchedPtr S , cudaPitchedPtr O, cudaPitchedPtr F, cudaPitchedPtr I, 
						  ty vicinity_distance, ty vicinity_speed, 
						  ty vicinity_orientation, ty max_docking_time, ty tdt, 
						  size_t width, size_t task_size,
						  size_t block_points, 
						  size_t individuals, size_t points, 
						  pre_exec prep = pre_exec(), 
						  post_exec post = post_exec() )
{

    //0. load shared memory with inputs and outputs. 

    ty * Ss = (ty *) fitness_twodee3_kernel_mem; 
    ty * Ir = (ty * ) I.ptr;

    copy_to_shared_mem<ty, pre_exec>(Ss, S.ptr, S.pitch, S.ysize, block_points);

    __syncthreads();

    for (int i=0; i < block_points; i+= blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < block_points)
	{
	    ty distance = sqrt(Ss[col] * Ss[col] + Ss[2*block_points + col] * Ss[2*block_points + col]);
	    ty speed    = sqrt(Ss[block_points + col] * Ss[block_points + col] + Ss[3*block_points + col] * Ss[3*block_points + col]);		// sqrt(vx^2 + vy^2)
	    ty theta = Ss[4*block_points + col];
	    
	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;	

	    ty * Fr = (ty *)F.ptr;
	    size_t row_size = F.pitch / sizeof(ty);
	    size_t offset =block_points*blockIdx.x + col; 
	    ty timeBonus = (max_docking_time - tdt)/max_docking_time;
	    ty alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));

	    if (Ir[offset] > distance/2) 
	    {
		if (distance < vicinity_distance && fabs(theta) < vicinity_orientation && speed < vicinity_speed)
		    Fr[offset] = 1 + timeBonus;	
		else
		    Fr[offset] = alpha;
	    } 
	    else
		Fr[offset] = 0;

	    Fr[row_size  +  offset] =  distance;
	    Fr[2*row_size + offset] =  speed;
	    Fr[3*row_size + offset] =  Ss[4*block_points + threadIdx.x];//theta
	}
	
    }
    __syncthreads();
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_twodee3(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, ty vicinity_distance, ty vicinity_speed, 
				       ty vic_orientation, ty max_docking_time, ty tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee3", dims_);
    cu_compute_fitness_twodee3_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(*S, *O, *F, *I, 
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
cudaError_t cu_compute_fitness_twodee3<float, nop_functor<float>, nop_functor<float> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, float vicinity_distance, 
											    float vicinity_speed, float vic_orientation, float max_docking_time, 
											    float tdt, size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee3", dims_);
    cu_compute_fitness_twodee3_kernel<float, nop_functor<float>, nop_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, 
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
cudaError_t cu_compute_fitness_twodee3<double, nop_functor<double>, nop_functor<double> >(cudaPitchedPtr *S , cudaPitchedPtr *O, cudaPitchedPtr *F, cudaPitchedPtr *I, 
											       double vicinity_distance,double vicinity_speed, 
											       double vic_orientation, double max_docking_time, 
											       double tdt,size_t width, cuda::kernel_dimensions * dims_  )
{
    print_parameters("cu_compute_fitness_twodee3", dims_);
    cu_compute_fitness_twodee3_kernel<double, nop_functor<double>, nop_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*S, *O, *F, *I, 
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
/*
extern __shared__ char transpose_mem [];
__global__ void cu_transpose_kernel(float *O, float *I, size_t count, size_t size )
{


    float * Ss = (float *) transpose_mem; 

    //Fix transpose function. The dimensions are fine. Theres something else wrong.
    transp_to_shared_mem<float, nop_functor<float> >(Ss, I, count, size);
    //copy_to_shared_mem<float, nop_functor<float> >(Ss, I, (count * size) / gridDim.x );
    __syncthreads();
    copy_to_global_mem<float, nop_functor<float> >(O, Ss, (count * size) / gridDim.x );
    //O[blockIdx.x*blockDim.x + threadIdx.x] =blockIdx.x*blockDim.x + threadIdx.x;

    __syncthreads();
}


cudaError_t transpose(float *O, float *I,  size_t count, size_t size, dim3 g, dim3 b)
{
    size_t s = (sizeof(float) * count * size) / g.x;
    printf("\nthreads %d blocks %d shared memory %d\n", b.x, g.x, s);
    cu_transpose_kernel <<<g, b, s >>>(O, I, count, size);
    cudaThreadSynchronize();
    return cudaGetLastError();
    }*/


#endif 

