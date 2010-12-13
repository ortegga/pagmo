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

template <typename ty>
__device__ __forceinline__ void load_to_shared(ty * shared, ty * global, size_t size)
{
    size_t segment = size / blockDim.x + (size % blockDim.x ? 1 : 0);//bad bad bad
    for (int i=0; i < segment ; ++i)
    {
	if (i*segment + threadIdx.x < size)
	    shared [i*segment + threadIdx.x] = global[blockIdx.x*size +  i*segment + threadIdx.x];
    }
}

template <typename ty, typename ftor >
__device__ __forceinline__ void cuda_copy (ty * to, ty * from, size_t size, ftor f = ftor() )
{
    for (int i=0; i < size ; ++i)
    {
	to [i] = f(from[i]);
    }
}

template <typename ty, typename ftor >
__device__ __forceinline__ void cuda_copy1 (ty * to, ty * from, ty param, size_t size, ftor f = ftor() )
{
    for (int i=0; i < size ; ++i)
    {
	to [i] = f(from[i], param);
    }
}
