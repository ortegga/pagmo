#ifndef __PAGMO_CUDA_TASK_H__
#define __PAGMO_CUDA_TASK_H__

#include <map>
#include "cudaty.h"
#include "dataset.h"


class CudaTask
{
 public:
  CudaTask(CudaInfo & info, int taskCount);
  virtual ~CudaTask();

  enum 
  {
    InputParam = 0,
    WeightParam = 1,
    HiddenParam = 2,
    OutputParam = 3
  };


  bool SetData (int taskid, int parameterid, const std::vector<CUDA_TY> & data);
  bool GetData (int taskid, int parameterid, std::vector<CUDA_TY> & data);
  bool CreateData(int parameterid,  int stride, bool bHost);
  bool HasData(int parameterid);
  virtual bool Launch () = 0;

 protected:

  CudaDataSet * GetDataSet( int parameterid);
  typedef std::map<int, CudaDataSet * > DataSetMap;
  int GetTaskSize() {return m_taskCount;}
  DataSetMap m_dataSet;
  int m_taskCount;
  CudaInfo & m_info;

};

class PerceptronTask : public CudaTask
{
 public:
  PerceptronTask(CudaInfo & info, int taskCount);
  virtual bool Launch();
};

class MultilayerPerceptronTask : public CudaTask
{
 public:
  MultilayerPerceptronTask(CudaInfo & info, int taskCount);
  virtual bool Launch();
};

class ElmanTask : public CudaTask
{
 public:
  virtual bool Launch();
};

class CTRNNTask : public CudaTask
{
 public:
  virtual bool Launch();
};




#endif
