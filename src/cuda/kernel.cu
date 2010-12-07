//-*-c++-*-
//Basic kernel to compute the result of a layer's operation

#ifndef __PAGMO_CUDA_KERNELS__
#define  __PAGMO_CUDA_KERNELS__

#include "stdio.h"
#include "kernel.h"
#include "common.cu"


///////////////////////////////////////////////////////////////////////////////////////////////
// computes y += alpha * x1
template <typename cuda_type, size_t size>
__device__ __forceinline__ void increment(cuda_type *Y,  cuda_type *X,  const cuda_type alpha) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] += alpha * X[idx];
    }
}

// computes y = x1 - x2
template <typename cuda_type, size_t size>
__device__ __forceinline__ void assign_diff(cuda_type *Y,  cuda_type *X1,  cuda_type * X2) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] = X1[idx] - X2[idx];
    }
}

// computes y = x1 + alpha * x2
template <typename cuda_type, size_t size>
__device__ __forceinline__ void assign_sum(cuda_type *Y,  cuda_type *X1,  
					   cuda_type* X2, const cuda_type alpha) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] = X1[idx] + alpha * X2[idx];
    }
}

// computes y = alpha1 * ( x1 + x2 + beta*x3 )
template <typename cuda_type, size_t size>
__device__ __forceinline__ void increment_sum_sum(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
						  cuda_type* X3, const cuda_type alpha, 
						  const cuda_type beta) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] = alpha*(X1[idx] + X2[idx] + beta*X3[idx]);
    }
}

// computes y = x1 + alpha * x2 ; x2 += x3
template <typename cuda_type, size_t size>
__device__ __forceinline__ void assign_sum_increment(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
						     cuda_type* X3, cuda_type alpha) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] = X1[idx] + alpha*X2[idx];
	X2[idx] += X3[idx];
    }
}


///////////////////////////////////////////////////////////////////////
// hills equation

template <typename cuda_type, typename preprocessor, typename pre_exec, typename post_exec>
struct hills_dynamical_system 
{
    //static size_t get_size () {return 2;} 
    enum { size = 2};

    __device__ void operator () (cuda_type *S,  cuda_type *D,  cuda_type* O, 
				 cuda_type t, cuda_type max_val, preprocessor prec = preprocessor (),
				 pre_exec pre = pre_exec(), 
				 post_exec post = post_exec())
	{

	    const cuda_type nu = 0.08, mR = (1.5 * 0.5);	
	    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	    unsigned int sstride = 6*idx;
	    unsigned int ostride = 2*idx;

	    cuda_type x = pre(S[sstride]);
	    cuda_type vx = pre(S[++sstride]);
	    cuda_type y = pre(S[++sstride]);
	    cuda_type vy = pre(S[++sstride]);
	    cuda_type theta = pre(S[++sstride]);	
	    cuda_type omega = pre(S[++sstride]);
	
	    cuda_type distance = sqrt(x * x + y * y);

	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;
	
	    cuda_type ul = prec (O[ostride], max_val);
	    cuda_type ur = prec (O[++ostride], max_val);
       
	    D[sstride] = (ul - ur) * 1/mR;
	    D[--sstride] = post(omega);
	    D[--sstride] = post(-2 * nu * vx + (ul + ur) * sin(theta));
	    D[--sstride] = post(vy);
	    D[--sstride] = post(2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta));
	    D[--sstride] = post(vx);
	}
};


/////////////////////////////////////////////////////////////////////////
// runge kutta integrator
//


/*template <typename cuda_type, typename pre_exec, typename activ_type >
__global__ void cu_compute_layer_kernel(cuda_type *X, cuda_type *W,  
					cuda_type *Y, size_t inputs, size_t outputs,
					size_t tasks_per_block, 
					size_t individuals, size_t points, 
					pre_exec pre = pre_exec(),
					activ_type activator = activ_type()) 
{

    size_t netsize = (inputs + 1)*outputs;
    size_t block_individuals = tasks_per_block / (points * outputs);


    //0. load shared memory with inputs and weights. 
    cuda_type * Ws = (cuda_type *) compute_layer_shared_mem; 
    load_to_shared<cuda_type>(Ws, W, block_individuals*netsize);

    cuda_type * Xs = & ((cuda_type *) compute_layer_shared_mem)[block_individuals*netsize];// inputs come after the weights
    load_to_shared<cuda_type>(Xs, X, inputs*block_individuals*points);

    __syncthreads();

    //Add check for last block that will be running less than normal threads
    if (threadIdx.x < tasks_per_block)
    {
	size_t tx = blockIdx.x * tasks_per_block + threadIdx.x;
	size_t taskid = tx / outputs;
	size_t yid = tx % outputs;
	//size_t individ = tx / (outputs*points);
	size_t sha_individ = threadIdx.x / (outputs*points);
	size_t sha_taskid = threadIdx.x / outputs;   
	//1. load in the bias
	cuda_type value = pre(Ws[netsize*sha_individ + (inputs + 1)*yid + inputs]);

	//2. Add the weights * inputs.
	for (int i=0; i < inputs; ++i)
	{
	    value += pre(Xs[inputs*sha_taskid+i])*pre(Ws[ netsize*sha_individ + (inputs + 1)*yid + i]);
	}

	//3. save to output
	Y[taskid*outputs+yid] = activator ( value );     
    }
    };*/



extern __shared__ char rk_shared_mem [];

template <typename cuda_type, typename DynamicSystem, typename pre_exec, typename post_exec>

__global__ void cu_runge_kutta_integrate (cuda_type  * X , cuda_type * O, cuda_type t , cuda_type dt ,  
					  const cuda_type max_val,
					  size_t outputs,
					  size_t tasks_per_block, 
					  size_t individuals, size_t points, 
					  DynamicSystem system = DynamicSystem(),
					  pre_exec pre = pre_exec(),
					  post_exec post = post_exec())
{

    //<TODO> handle last block
    //int idx = blockIdx.x*blockDim.x + threadIdx.x;
    /*const size_t datasize = 6;
    if (threadIdx.x < tasks_per_block)
    {

	cuda_type dxdt [ datasize ] ;
	cuda_type  dh = cuda_type( 0.5 ) * dt;
	cuda_type th = t + dh;

	const cuda_type val2 = cuda_type( 2.0 );

	system(X, dxdt, O, t, max_val);

	cuda_type xt [datasize];

	assign_sum<cuda_type, datasize>( xt , X , dxdt , dh );

	cuda_type dxt [datasize];

	system( xt , dxt , O, th , max_val);

	assign_sum<cuda_type, datasize>( xt, X, dxt , dh );

	cuda_type dxm [datasize];
	system( xt , dxm , O, th , max_val);

	assign_sum_increment<cuda_type, datasize>( xt, X, dxm,dxt, dt );

	system( xt , dxt , O, cuda_type( t + dt ) , max_val);

	increment_sum_sum<cuda_type, datasize>( X, dxdt, dxt,  dxm, 
	dt /  cuda_type( 6.0 ) , val2 );
	}*/

}


static void print_parameters(const char * name, cuda::kernel_dimensions * dims_)
{
    printf("%x\n", dims_);
    printf("%s with \n grid size = %d\n block size = %d\n shared mem = %d\n tasks per block = %d\n", 
	   name, dims_->get_grid_dims().x, dims_->get_block_dims().x, dims_->get_shared_mem_size(), dims_->get_tasks_per_block());

}

//////////////////////////////////////////////////////////////////////////////////////////////
// kernel interface functions


template <typename cuda_type, typename dynamicalsystem, typename pre_exec, typename post_exec>
cudaError_t runge_kutta_integrate (cuda_type  * X , cuda_type * O, cuda_type t , cuda_type dt , cuda_type max_val, cuda::kernel_dimensions * dims_)
{
    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <cuda_type, dynamicalsystem, pre_exec, post_exec >
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
cudaError_t runge_kutta_integrate <float, hills_dynamical_sys_float , nop_functor<float> , nop_functor<float> >
                                                                                           (float  * X , float * O, float t , float dt , 
											    float max_val, cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <float, hills_dynamical_sys_float , nop_functor<double>, nop_functor<double> >
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
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double , nop_functor<double>, nop_functor<double> > 
                                                                                         (double  * X , double * O, double t , double dt , 
											       double max_val, cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <double, hills_dynamical_sys_double ,nop_functor<double>, nop_functor<double> >
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

template <typename ty, typename pre_exec, typename post_exec >
__global__ void cu_compute_fitness_mindis_kernel(ty *S , ty *O, ty *F, ty *I, size_t width, 
						 pre_exec prep = pre_exec(), 
						 post_exec post = post_exec() )
{

    //  unsigned int bx = blockIdx.x, by = blockIdx.y;
    //Todo remove S[1] as its not used
    unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int offset = tx * 4;


    apply<ty, pre_exec, 4> pre_app;
    pre_app(&S[offset], &S[offset]);

    ty distance = sqrt(S[offset] * S[offset] + S[offset + 2] * S[offset + 2]);
    ty speed    = sqrt(S[offset + 1] * S[offset + 1] + S[offset + 3] * S[offset + 3]);		// sqrt(vx^2 + vy^2)

    O[tx*3] =  1/( 1 + distance );
    O[tx*3 + 1] =  distance;
    O[tx*3 + 2] =  speed;
    apply<ty, post_exec, 3>post_app;

    post_app(&O[tx*3], &O[tx*3]);
}


template <typename ty, typename pre_exec, typename post_exec >
cudaError_t cu_compute_fitness_mindis(ty *S , ty *O, ty *F, ty *I, size_t width, cuda::kernel_dimensions * dims_  )
{
    cu_compute_fitness_mindis_kernel<ty, pre_exec, post_exec >      
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size() >>>
	(S , O, F, I, width);
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
	(S , O, F, I, width);
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
	(S , O, F, I, width);
    cudaThreadSynchronize();
    return cudaGetLastError();
}


/////////////////////////////////////////////////////////////////////////////
//Misc kernels


#endif 
