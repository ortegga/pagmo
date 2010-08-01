#ifndef __PAGMO_CUDA_DATASET__
#define  __PAGMO_CUDA_DATASET__

#include "cudaty.h"
#include "cudainfo.h"
#include <vector>

class CudaDataSet
{
 public:
  CudaDataSet(CudaInfo & info, int subTaskCount, int stride, bool bHost = false);
  virtual ~CudaDataSet();
  bool GetValues(int taskId, CUDA_TY * subTaskData);
  bool SetValues(int taskId, const CUDA_TY * subTaskData);
  int GetTaskSize() {return m_stride;}
  int GetSize() {return m_taskCount * m_stride;}
  CUDA_TY ** GetData() {return &m_data;}

 private:
  CUDA_TY * m_data;
  int m_taskCount;
  int m_stride;
  bool m_bHost;
};

#endif
