#ifndef __PAGMO_CUDA_SCHED_H__
#define __PAGMO_CUDA_SCHED_H__

#include <map>
#include "cudaty.h"
#include "dataset.h"


class CudaScheduler
{
 public:
  CudaScheduler(CudaInfo & info, int taskCount);
  virtual ~CudaScheduler();

  bool SetData (int taskid, int parameterid, const std::vector<CUDA_TY> & data);
  bool GetData (int taskid, int parameterid, std::vector<CUDA_TY> & data);
  bool CreateData(int parameterid,  int stride, bool bHost);
  virtual bool Launch () = 0;

 protected:

  CudaDataSet * GetDataSet( int parameterid);
  std::map<int, CudaDataSet * > m_dataSet;
  int m_taskCount;
  CudaInfo & m_info;

};

#endif
