
#include "cudamem.h"

CudaAllocator::CudaAllocator(const CudaInfo & info)
{
  UseAsync(info.);
}

CudaAllocator::~CudaAllocator()
{

}

template < class Type > 
Type * CudaAllocator::AllocHost(size_t size)
{

}

Type * CudaAllocator::AllocDevice(size_t size)
{

}
void CudaAllocator::FreeHost(Type * data)
{

}
void CudaAllocator::FreeDevice(Type * data)
{

}
bool CudaAllocator::MemcpyToHost(Type * hostMem, const CUDA_TY * devMem)
{

}
bool CudaAllocator::MemcpyToDev(CUDA_TY * devMem, const  CUDA_TY * hostMem)
{

}
