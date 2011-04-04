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
    
    __device__ __forceinline__ ty operator() ( ty val, ty )
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
__device__ __forceinline__ void copy_to_shared_mem(ty * shared, void * global, size_t globalpitch, size_t globalheight, size_t count, ftor f = ftor())
{
    for (int i=0; i < count; i += blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < count)
	{
	    for (int j = 0; j < globalheight; j+= blockDim.y )
	    {
		size_t row = j  + threadIdx.y;
		char * gr = ((char *) global + row * globalpitch);
		if (row < globalheight)
		{
		    shared[ row * count + col] = f( ((ty *)gr)[blockIdx.x * count + col] );
		}
	    }
	}
    }
}


template <typename ty, typename ftor >
__device__ __forceinline__ void copy_to_shared_mem1(ty * shared, void * global, size_t globalpitch, size_t globalheight, size_t count, ty param, ftor f = ftor())
{
    for (int i=0; i < count; i += blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < count)
	{
	    for (int j = 0; j < globalheight; j+= blockDim.y )
	    {
		size_t row = j  + threadIdx.y;
		char * gr = ((char *) global + row * globalpitch);
		if (row < globalheight)
		{
		    shared[ row * count + col] = f( ((ty *)gr)[blockIdx.x * count + col], param );
		}
	    }
	}
    }
}

template <typename ty, typename ftor >
__device__ __forceinline__ void copy_to_global_mem(void * global, size_t globalpitch, size_t globalheight, ty * shared, size_t count, ftor f = ftor())
{

    for (int i=0; i < count; i += blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < count)
	{
	    for (int j = 0; j < globalheight; j+= blockDim.y )
	    {
		size_t row = j  + threadIdx.y;
		char * gr = ((char *) global + row * globalpitch);
		if (row < globalheight)
		{
		    ((ty *)gr)[blockIdx.x * count + col] = f(shared[ row * count + col]);
		}
	    }
	}
    }
}

template <typename ty, typename ftor >
__device__ __forceinline__ void copy_to_global_mem1(void * global, size_t globalpitch, size_t globalheight, ty * shared, size_t count, ty param, ftor f = ftor())
{
    for (int i=0; i < count; i += blockDim.x )
    {
	size_t col = i + threadIdx.x;
	if (col < count)
	{
	    for (int j = 0; j < globalheight; j+= blockDim.y )
	    {
		size_t row = j  + threadIdx.y;
		char * gr = ((char *) global + row * globalpitch);
		if (row < globalheight)
		{
		    ((ty *)gr)[blockIdx.x * count + col] = f(shared[ row * count + col], param);
		}
	    }
	}
    }
}

