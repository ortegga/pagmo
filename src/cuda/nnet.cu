
//-*-c++-*-
//////////////////////////////////////////////////////////////////
#include "stdio.h"
#include "cudainfo.h"
#include "nnet.h"
#include "common.cu"
#include "kernel_dims.h"



template <typename cuda_type>
struct sigmoid_functor 
{
    __device__ __forceinline__ cuda_type operator ()( cuda_type val)
	{
	    return 1.0f/(1 + exp(-val));
	}
};

template <typename cuda_type>
struct linear_functor 
{
    __device__ __forceinline__ cuda_type operator () ( cuda_type val)
	{
	    return val > 0.0f ? 1.0f : 0.0f;
	}
};



/////////////////////////////////////////////////////////////////////////////////////
//Description: neural networks layer compute
// whole individuals. outputs*points = individual size
// continuous memory in I, O, W. Need to compute netid as
// indivId*points+pointid


//calculate correct offsets
extern __shared__ char compute_layer_shared_mem [];


template <typename cuda_type, typename pre_exec, typename activ_type >
__global__ void cu_compute_layer_kernel(cuda_type *X, cuda_type *W,  
					cuda_type *Y, size_t inputs, size_t outputs,
					size_t tasks_per_block, 
					size_t individuals, size_t points, 
					pre_exec pre = pre_exec(),
					activ_type activator = activ_type()) 
{

    size_t netsize = (inputs + 1)*outputs;
    size_t block_individuals = BLOCK_INDIVIDUALS(tasks_per_block, outputs, points);
    size_t tx = GLOBAL_ADJUSTED_TX(tasks_per_block);
    size_t individ = GLOBAL_INDIV_ID(tasks_per_block,outputs, points);


    //0. load shared memory with inputs and weights. 
    cuda_type * Ws = (cuda_type *) compute_layer_shared_mem; 
    cuda_type * Xs = & ((cuda_type *) compute_layer_shared_mem)[block_individuals*netsize];// inputs come after the weights

    copy_to_shared_mem<cuda_type, pre_exec>(Ws, W, block_individuals * netsize);
    copy_to_shared_mem<cuda_type, pre_exec>(Xs, X, BLOCK_POINTS(tasks_per_block,outputs) * inputs);
    __syncthreads();

    //Add check for last block that will be running less than normal threads
    if (IS_VALID_FOR_BLOCK(tasks_per_block,points, outputs))
    {

	size_t taskid = GLOBAL_POINT_ID(tasks_per_block,outputs);
	size_t yid = tx % outputs;
	size_t sha_individ = BLOCK_INDIV_ID(tasks_per_block, outputs,points); 
	size_t sha_taskid = BLOCK_POINT_ID(tasks_per_block, outputs);   
	cuda_type * Wc = &Ws[netsize * sha_individ];

	//1. load in the bias
	cuda_type value = Wc[(inputs + 1)*yid];// + inputs];
	//2. Add the weights * inputs.
	for (int i=0; i < inputs; ++i)
	{
	    value += Xs[inputs*sha_taskid+i]*Wc[(inputs + 1)*yid + i + 1];
	}
	//3. save to output
	Y[taskid*outputs+yid] = activator ( value );     
    }
}

static void print_parameters(const char * name, cuda::kernel_dimensions * dims_)
{
/*    printf("%s with \n grid size = %d\n block size = %d\n shared mem = %d\n tasks per block = %d\n", 
	   name, dims_->get_grid_dims().x, dims_->get_block_dims().x, dims_->get_shared_mem_size(), dims_->get_tasks_per_block());
    printf(" individuals = %d\n points = %d\n task size = %d\n ", 
    dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());*/

}


//////////////////////////////////////////////////////////////////////////////////////////////
// kernel interface functions

template <typename cuda_type, typename pre_exec, typename activ_type>
cudaError_t cu_compute_layer(cuda_type *X, cuda_type *W,  cuda_type *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{
    print_parameters("cu_compute_layer", dims_);
    cu_compute_layer_kernel<cuda_type, pre_exec, activ_type> 
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X, W, Y, inputs, dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), 
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_layer<float, nop_functor<float>, linear_functor<float> >(float *X, float *W,  float *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{
    print_parameters("cu_compute_layer", dims_);
    cu_compute_layer_kernel<float, nop_functor<float>, linear_functor <float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X, W, Y, inputs, dims_->get_task_size(),
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(),
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_layer<float, nop_functor<float>, sigmoid_functor<float> > (float *X, float *W,  float *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{

    print_parameters("cu_compute_layer", dims_);
    cu_compute_layer_kernel<float, nop_functor<float>, sigmoid_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X, W, Y, inputs, dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), dims_->get_points());
	 cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_layer<double, nop_functor<double>, linear_functor<double> > (double *X, double *W,  double *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{
    print_parameters("cu_compute_layer", dims_);
    cu_compute_layer_kernel<double, nop_functor<double>, linear_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X, W, Y, inputs, dims_->get_task_size(),
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(),
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t  cu_compute_layer<double, nop_functor<double>, sigmoid_functor<double> > (double *X, double *W,  double *Y, 
								  size_t inputs, cuda::kernel_dimensions * dims_)
{
    print_parameters("cu_compute_layer", dims_);
    cu_compute_layer_kernel<double, nop_functor<double>, sigmoid_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(X, W, Y, inputs, dims_->get_task_size(), 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(),
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}
