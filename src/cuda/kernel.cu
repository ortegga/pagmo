//-*-c++-*-
//Basic kernel to compute the result of a layer's operation

#ifndef __PAGMO_CUDA_KERNELS__
#define  __PAGMO_CUDA_KERNELS__

#include "stdio.h"
#include "kernel.h"
#include "common.cu"


///////////////////////////////////////////////////////////////////////////////////////////////
// computes y += alpha * x1
template <typename fty, size_t size>
__device__ __forceinline__ void increment(fty *Y,  fty *X,  const fty alpha) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] += alpha * X[idx];
    }
}

// computes y = x1 - x2
template <typename fty, size_t size>
__device__ __forceinline__ void assign_diff(fty *Y,  fty *X1,  fty * X2) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] = X1[idx] - X2[idx];
    }
}

// computes y = x1 + alpha * x2
template <typename fty, size_t size>
__device__ __forceinline__ void assign_sum(fty *Y,  fty *X1,  
					   fty* X2, const fty alpha) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] = X1[idx] + alpha * X2[idx];
    }
}

// computes y = alpha1 * ( x1 + x2 + beta*x3 )
template <typename fty, size_t size>
__device__ __forceinline__ void increment_sum_sum(fty *Y,  fty *X1,  fty* X2, 
						  fty* X3, const fty alpha, 
						  const fty beta) 
{
    for (int i=0; i<size; ++i)
    {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x + i;
	Y[idx] = alpha*(X1[idx] + X2[idx] + beta*X3[idx]);
    }
}

// computes y = x1 + alpha * x2 ; x2 += x3
template <typename fty, size_t size>
__device__ __forceinline__ void assign_sum_increment(fty *Y,  fty *X1,  fty* X2, 
						     fty* X3, fty alpha) 
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

template <typename fty, typename preprocessor, typename pre_exec, typename post_exec>
struct hills_dynamical_system 
{
    //static size_t get_size () {return 2;} 
    enum { size = 2};

    __device__ void operator () (fty *S,  fty *D,  fty* O, 
				 fty t, preprocessor prec = preprocessor (),
				 pre_exec pre = pre_exec(), 
				 post_exec post = post_exec())
	{

	    const fty nu = 0.08, mR = (1.5 * 0.5);	
	    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	    unsigned int sstride = 6*idx;
	    unsigned int ostride = 2*idx;

	    fty x = pre(S[sstride]);
	    fty vx = pre(S[++sstride]);
	    fty y = pre(S[++sstride]);
	    fty vy = pre(S[++sstride]);
	    fty theta = pre(S[++sstride]);	
	    fty omega = pre(S[++sstride]);
	
	    fty distance = sqrt(x * x + y * y);

	    if(theta < -M_PI) theta += 2 * M_PI;
	    if(theta > M_PI) theta -= 2 * M_PI;
	
	    fty ul = prec (O[ostride]);
	    fty ur = prec (O[++ostride]);
       
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

//assume task size = 1
extern __shared__ char rk_shared_mem [];

template <typename fty, typename DynamicSystem, typename o_pre_exec, typename pre_exec, typename post_exec>

__global__ void cu_runge_kutta_integrate (fty  * X , fty * O, const fty t , const fty dt ,  
					  const fty max_val,
					  size_t task_size,
					  size_t tasks_per_block, 
					  size_t individuals, size_t points, 
					  DynamicSystem system = DynamicSystem(),
					  pre_exec pre = pre_exec(),
					  post_exec post = post_exec())
{


    size_t block_individuals = tasks_per_block / (points * task_size);


    //0. load shared memory with inputs and outputs. 
    const size_t order = 6;
    size_t offset = block_individuals*task_size*points*order;
    

    fty * Os = (fty *) rk_shared_mem; 
    fty * Xs = & ((fty *) rk_shared_mem)[offset];// inputs come after the weights
    fty * DxDt = & ((fty *) rk_shared_mem)[2*offset];
    fty * Xt = & ((fty *) rk_shared_mem)[3*offset];
    fty * Dxt = & ((fty *) rk_shared_mem)[4*offset];
    fty * Dxm = & ((fty *) rk_shared_mem)[5*offset];

    size_t globaloffset = threadIdx.x + blockIdx.x* tasks_per_block;


    if (threadIdx.x < tasks_per_block)
    {
	cuda_copy1<fty, o_pre_exec> (&Os[threadIdx.x*2], &O[globaloffset*2], max_val, 2);
	cuda_copy<fty, pre_exec> (&Xs[threadIdx.x*order], &X[globaloffset*7], order);
    }


    __syncthreads();


    //<TODO> handle last block
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadIdx.x < tasks_per_block)
    {

	//fty dxdt [ datasize ] ;
	const fty  dh = fty( 0.5 ) * dt;
	const fty th = t + dh;
	const fty val2 = fty( 2.0 );
	
	// k1
	system(Xs, DxDt, Os, t);

	assign_sum<fty, order>( Xt , Xs , DxDt , dh );
	
	//k2
	system( Xt , Dxt , Os, th );
	assign_sum<fty, order>( Xt, Xs, Dxt , dh );


	//k3
	system( Xt , Dxm , Os, th );
	assign_sum_increment<fty, order>( Xt, Xs, Dxm, Dxt, dt );
	
	//k4
	system( Xt , Dxt , Os, fty( t + dt ));

	increment_sum_sum<fty, 6>( Xs, DxDt, Dxt,  Dxm, dt /  fty( 6.0 ) , val2 );

	cuda_copy<fty, post_exec> (&X[globaloffset*7], &Xs[threadIdx.x*order], order);
    }

    __syncthreads();

}


static void print_parameters(const char * name, cuda::kernel_dimensions * dims_)
{
    printf("%x\n", dims_);
    printf("%s with \n grid size = %d\n block size = %d\n shared mem = %d\n tasks per block = %d\n", 
	   name, dims_->get_grid_dims().x, dims_->get_block_dims().x, dims_->get_shared_mem_size(), dims_->get_tasks_per_block());
    printf("individuals = %d\n points = %d\n task size = %d\n ", 
	   dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());

}

//////////////////////////////////////////////////////////////////////////////////////////////
// kernel interface functions


template <typename fty, typename dynamicalsystem, typename o_pre_exec, typename pre_exec, typename post_exec>
cudaError_t runge_kutta_integrate (fty  * X , fty * O, fty t , fty dt , fty max_val, cuda::kernel_dimensions * dims_)
{
    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <fty, dynamicalsystem, pre_exec, post_exec >
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
cudaError_t runge_kutta_integrate <float, hills_dynamical_sys_float , scale_functor<float>, nop_functor<float> , nop_functor<float> >
                                                                                           (float  * X , float * O, float t , float dt , 
											    float max_val, cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <float, hills_dynamical_sys_float , scale_functor<float>, nop_functor<double>, nop_functor<double> >
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
cudaError_t runge_kutta_integrate <double, hills_dynamical_sys_double , scale_functor<double>, nop_functor<double>, nop_functor<double> > 
                                                                                         (double  * X , double * O, double t , double dt , 
											       double max_val, cuda::kernel_dimensions * dims_)
{

    print_parameters("runge_kutta_integrate", dims_);
    cu_runge_kutta_integrate <double, hills_dynamical_sys_double ,scale_functor<double>, nop_functor<double>, nop_functor<double> >
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

    size_t block_individuals = tasks_per_block / (points * task_size);


    //0. load shared memory with inputs and outputs. 
    const size_t order = 4;
    //size_t offset = block_individuals*task_size*points*order;
    

    ty * Ss = (ty *) rk_mindis_kernel_mem; 

    size_t globaloffset = threadIdx.x + blockIdx.x* tasks_per_block;

    size_t offset = threadIdx.x*order;
    if (threadIdx.x < tasks_per_block)
    {
	cuda_copy<ty, pre_exec> (&Ss[offset], &S[globaloffset*order], order);
    }

    __syncthreads();


    //<TODO> handle last block
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadIdx.x < tasks_per_block)
    {

	ty distance = sqrt(Ss[offset] * Ss[offset] + Ss[offset + 2] * Ss[offset + 2]);
	ty speed    = sqrt(Ss[offset + 1] * Ss[offset + 1] + Ss[offset + 3] * Ss[offset + 3]);		// sqrt(vx^2 + vy^2)

	O[globaloffset*4] =  1/( 1 + distance );
	O[globaloffset*4 + 1] =   distance;
	O[globaloffset*4 + 2] =  speed;
	O[globaloffset*4 + 3] =  Ss[offset + 4];

    }
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
//Misc kernels


#endif 

