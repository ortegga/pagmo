
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
	    //Needs to be a better way to do this
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
    size_t block_individuals = tasks_per_block / (points * outputs);
    size_t btx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tx = blockIdx.x * tasks_per_block + threadIdx.x;
    size_t individ = tx / (points * outputs);


    //0. load shared memory with inputs and weights. 
/*    cuda_type * Ws = (cuda_type *) compute_layer_shared_mem; 
    load_to_shared<cuda_type>(Ws, W, block_individuals*netsize);

    cuda_type * Xs = & ((cuda_type *) compute_layer_shared_mem)[block_individuals*netsize];// inputs come after the weights
    load_to_shared<cuda_type>(Xs, X, inputs*block_individuals*points);

    __syncthreads();

    //Add check for last block that will be running less than normal threads
    if (threadIdx.x < tasks_per_block && individ < individuals)
    {

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
	}*/
};

static void print_parameters(const char * name, cuda::kernel_dimensions * dims_)
{
    printf("%x\n", dims_);
    printf("%s with \n grid size = %d\n block size = %d\n shared mem = %d\n tasks per block = %d\n", 
	   name, dims_->get_grid_dims().x, dims_->get_block_dims().x, dims_->get_shared_mem_size(), dims_->get_tasks_per_block());
    printf(" individuals = %d\n points = %d\n task size = %d\n ", 
	   dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());

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


///////////////////////////////////////////////////////////
/*template <typename cuda_type, typename activ_type>
  __global__ void cu_compute_layer_with_segments_kernel(cuda_type *X,  cuda_type *W,  cuda_type *Y, int width, int seg,    
  activ_type activator = activ_type()) 
  {

  unsigned int bx = blockIdx.x, by = blockIdx.y;
  unsigned int tx = threadIdx.x, ty = threadIdx.y;

  //The order of weights is as follows:
  //1) the weights between X and Y
  //2) the bias for Y
  //3) the weights for the memory component*
  unsigned int offset = tx*(width+1);

  cuda_type value = W[offset + seg];
  for (unsigned int i=0; i < seg; ++i)
  {
  value += X[i]*W[offset + i];
  }

  for (unsigned int i=seg; i < width; ++i)
  {
  value += X[i]*W[offset +  i  + 1];
  }

  Y[tx] = activator( value );
  }


  template <typename cuda_type, typename activ_type>
  void cu_compute_layer_with_segments(cuda_type *X, cuda_type *W,  
  cuda_type *Y, int width, int seg,
  dim3 gridsize, dim3 blocksize)
  {
  cu_compute_layer_with_segments_kernel<cuda_type, activ_type><<<gridsize, blocksize>>>(X, W, Y, width, seg);
  }*/
