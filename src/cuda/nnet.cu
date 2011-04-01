
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
__global__ void cu_compute_layer_kernel(cudaPitchedPtr X, cudaPitchedPtr W,  
					cudaPitchedPtr Y,
					size_t tasks_per_block, 
					size_t individuals, size_t points, 
					pre_exec pre = pre_exec(),
					activ_type activator = activ_type()) 
{

    size_t block_individuals = BLOCK_INDIVIDUALS(tasks_per_block,  points);
    size_t block_points = BLOCK_POINTS(tasks_per_block);

    //0. load shared memory with inputs and weights. 
    cuda_type * Ws = &((cuda_type *) compute_layer_shared_mem)[0]; 
    cuda_type * Xs = & ((cuda_type *) compute_layer_shared_mem)[block_individuals*W.ysize];// inputs come after the weights

    copy_to_shared_mem<cuda_type, pre_exec>(Ws, W, block_individuals);
    copy_to_shared_mem<cuda_type, pre_exec>(Xs, X, block_points);
    
    __syncthreads();

    for(int i=0; i < tasks_per_block; i+= blockDim.x)// task id
    {
	size_t taskid = i + threadIdx.x;//block's task id and individual id
	size_t individ = taskid / points;

	if (individ + block_individuals * blockIdx.x  < individuals)
	{

	    for(int j=0; j < Y.ysize; j+= blockDim.y)//the right y
	    {
		size_t jobid = j + threadIdx.y;
		cuda_type value = Ws[jobid * (X.ysize + 1) *  block_individuals  + individ];			    
		for (int k=0; k < X.ysize; ++k)//the right x
		{
		    value += Ws[(jobid * (X.ysize + 1) + k + 1) * block_individuals + individ] * Xs[k * block_points + taskid] ;
		}
		char * yp = ((char *) Y.ptr + jobid * Y.pitch);
		((cuda_type *)yp)[ blockIdx.x * block_points + taskid ] = activator(value);

	    }
	}
    }

    __syncthreads();
}

static void print_parameters(const char * name, cuda::kernel_dimensions * dims_)
{
/*    printf("%s with \n grid size = %d\n block size = <%d, %d>\n shared mem = %d\n tasks per block = %d\n", 
	   name, dims_->get_grid_dims().x, dims_->get_block_dims().x,dims_->get_block_dims().y,  dims_->get_shared_mem_size(), dims_->get_tasks_per_block());
    printf(" individuals = %d\n points = %d\n task size = %d\n ", 
    dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());*/

}


//////////////////////////////////////////////////////////////////////////////////////////////
// kernel interface functions

template <typename cuda_type, typename pre_exec, typename activ_type>
cudaError_t cu_compute_layer(cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{
    print_parameters("cu_compute_layer unknown call", dims_);
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_layer<float, nop_functor<float>, linear_functor<float> >(cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{
    /*print_parameters("cu_compute_layer", dims_);
    printf("cu_compute_layer with W = <%x, %d, %d, %d>", W->ptr, W->pitch, W->xsize, W->ysize );
    printf(" X = <%x, %d, %d, %d>", X->ptr, X->pitch, X->xsize, X->ysize );
    printf(" Y = <%x, %d, %d, %d>\n", Y->ptr, Y->pitch, Y->xsize, Y->ysize );
    printf(" individuals = %d\n points = %d\n task size = %d\n ", 
    dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());*/
    cu_compute_layer_kernel<float, nop_functor<float>, linear_functor <float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X, *W, *Y, //inputs, dims_->get_task_size(),
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(),
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_layer<float, nop_functor<float>, sigmoid_functor<float> > (cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{

/*    print_parameters("cu_compute_layer", dims_);
    printf("cu_compute_layer with W = <%x, %d, %d, %d>", W->ptr, W->pitch, W->xsize, W->ysize );
    printf(" X = <%x, %d, %d, %d>", X->ptr, X->pitch, X->xsize, X->ysize );
    printf(" Y = <%x, %d, %d, %d>\n", Y->ptr, Y->pitch, Y->xsize, Y->ysize );
    printf(" individuals = %d\n points = %d\n task size = %d\n ", 
    dims_->get_individuals(), dims_->get_points(), dims_->get_task_size());*/
    cu_compute_layer_kernel<float, nop_functor<float>, sigmoid_functor<float> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X, *W, *Y,
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(), dims_->get_points());
	 cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t cu_compute_layer<double, nop_functor<double>, linear_functor<double> > (cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, size_t inputs, cuda::kernel_dimensions * dims_)
{
    print_parameters("cu_compute_layer", dims_);
    cu_compute_layer_kernel<double, nop_functor<double>, linear_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X, *W, *Y, 
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(),
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}

template <>
cudaError_t  cu_compute_layer<double, nop_functor<double>, sigmoid_functor<double> > (cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, 
								  size_t inputs, cuda::kernel_dimensions * dims_)
{
    print_parameters("cu_compute_layer", dims_);
    cu_compute_layer_kernel<double, nop_functor<double>, sigmoid_functor<double> >
	<<<dims_->get_grid_dims(),
	dims_->get_block_dims(), 
	dims_->get_shared_mem_size()>>>
	(*X, *W, *Y,
	 dims_->get_tasks_per_block(), 
	 dims_->get_individuals(),
	 dims_->get_points());
    cudaThreadSynchronize();
    return cudaGetLastError();
}
