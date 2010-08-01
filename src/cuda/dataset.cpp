


#include "dataset.h"
#include "cuda.h"
#include "cuda_runtime.h"


CudaDataSet::CudaDataSet(CudaInfo & info, int subTaskCount, int stride, bool bHost):
  m_data(0),  m_taskCount(subTaskCount), m_stride(stride),m_bHost(bHost)
{
   std::cout<<"CudaDataSet::CudaDataSet: m_stride "<< m_stride<<std::endl;
   std::cout<<"CudaDataSet::CudaDataSet: subTaskCount "<< subTaskCount <<std::endl;
   std::cout<<"CudaDataSet::CudaDataSet: bHost "<< bHost <<std::endl;
  if (m_bHost)
  {
    cudaMallocHost(&m_data, GetSize() * sizeof(CUDA_TY));
  }
  else
  {
    cudaMalloc(&m_data, GetSize() * sizeof(CUDA_TY));
  }
}

bool CudaDataSet::GetValues(int taskId, CUDA_TY * subTaskData)
{
  std::cout<<"CudaDataSet GetValues: m_stride"<< m_stride<<std::endl;
  std::cout<<"CudaDataSet GetValues: taskid"<< taskId <<std::endl;

  cudaMemcpy(subTaskData, &m_data[taskId * m_stride], 
	     m_stride * sizeof(CUDA_TY), cudaMemcpyDeviceToHost);
  return true;
}
bool CudaDataSet::SetValues(int taskId, const CUDA_TY * subTaskData)
{
  //   std::cout<<"CudaDataSet SetValues: m_stride"<< m_stride<<std::endl;
  //std::cout<<"CudaDataSet SetValues: taskid"<< taskId <<std::endl;

  cudaMemcpy(&m_data[taskId * m_stride], subTaskData , 
	     m_stride * sizeof(CUDA_TY), cudaMemcpyHostToDevice);
  return true;
}


CudaDataSet::~CudaDataSet()
{
  if (m_bHost)
  {
    cudaFreeHost(m_data);
  }
  else
  {

    cudaFree(m_data);
  }
}
