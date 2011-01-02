//-*-c++-*-
////////////////////////////////////////////////////////////////////////////////////


#include "common.h"


template <typename ty>
struct nop_functor 
{
    __device__ __forceinline__ ty operator() ( ty val )
	{
	    return val;
	}
};


template <typename ty>
struct scale_functor 
{
    __device__ __forceinline__ ty operator ()(ty val, ty max_val)
	{
	    return ( val - 0.5f )* 2 * max_val;
	}
};



template <typename ty, typename functor, size_t s = 1>
struct apply 
{
    __device__ __forceinline__ void operator () (ty * dest, ty * src, const size_t size = s, functor f = functor())
	{
            #pragma unroll
	    for (int i=0; i < size; ++i)
	    {
		dest[i] = f(src[i]);
	    }    
	}
};

template <typename ty, typename ftor >
__device__ __forceinline__ void copy_to_shared_mem(ty * shared, ty * global, size_t size, ftor f = ftor())
{
    size_t segment = size / blockDim.x + (size % blockDim.x ? 1 : 0);//bad bad bad
    for (int i=0; i < segment ; ++i)
    {
	if (i*segment + threadIdx.x < size)
	    shared [i*segment + threadIdx.x] = f(global[blockIdx.x*size +  i*segment + threadIdx.x]);
    }
}

template <typename ty, typename ftor >
__device__ __forceinline__ void copy_to_shared_mem1(ty * shared, ty * global, size_t size, ty param, ftor f = ftor())
{
    size_t segment = size / blockDim.x + (size % blockDim.x ? 1 : 0);//bad bad bad
    for (int i=0; i < segment ; ++i)
    {
	if (i*segment + threadIdx.x < size)
	    shared [i*segment + threadIdx.x] = f(global[blockIdx.x*size +  i*segment + threadIdx.x], param);
    }
}


template <typename ty, typename ftor >
__device__ __forceinline__ void copy_to_global_mem(ty * global, ty * shared, size_t size, ftor f = ftor())
{
    size_t segment = size / blockDim.x + (size % blockDim.x ? 1 : 0);//bad bad bad
    for (int i=0; i < segment ; ++i)
    {
	if (i*segment + threadIdx.x < size)
	    global[blockIdx.x*size +  i*segment + threadIdx.x] = f(shared [i*segment + threadIdx.x]);
    }
}

template <typename ty, typename ftor >
__device__ __forceinline__ void copy_to_global_mem1(ty * global, ty * shared, size_t size, ty param, ftor f = ftor())
{
    size_t segment = size / blockDim.x + (size % blockDim.x ? 1 : 0);//bad bad bad
    for (int i=0; i < segment ; ++i)
    {
	if (i*segment + threadIdx.x < size)
	    global[blockIdx.x*size +  i*segment + threadIdx.x] = f(shared [i*segment + threadIdx.x],param);
    }
}
