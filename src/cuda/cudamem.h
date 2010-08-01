
#ifndef __PAGMO_CUDA_MEM_H__
#define __PAGMO_CUDA_MEM_H__

#include "cudaty.h"

//parameters should be, 
// 1) whether to used page locked memory
// 2) whether to allocate host memory
// 3) caching
// 4) other attributes

class CudaAllocator
{
 public:
  CudaAllocator(const CudaInfo & info );
  virtual ~CudaAllocator();

  template < class Type > 
  Type * AllocHost(size_t size);
  Type * AllocDevice(size_t size);
  void FreeHost(Type * data);
  void FreeDevice(Type * data);
  bool MemcpyToHost(Type * hostMem, const CUDA_TY * devMem);
  bool MemcpyToDev(CUDA_TY * devMem, const  CUDA_TY * hostMem);

  bool UseAsync() { return m_useAsync;}
  void UseAsync(bool useAsync) { m_useAsync = useAsync;}
 protected:
  bool m_useAsync;
};
#endif
