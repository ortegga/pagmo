

#include "cudasched.h"


CudaScheduler:: CudaScheduler(CudaInfo & info, int taskCount) : m_info(info), m_taskCount(taskCount)
{

}
bool CudaScheduler::SetData (int taskid, int parameterid, const std::vector<CUDA_TY> & data)
{
  CudaDataSet * pData = GetDataSet(parameterid);
  if (!pData)
  {
    return false;
  }

  CUDA_TY * temp = new CUDA_TY(pData->GetTaskSize());
  std::copy(data.begin(), data.end(), temp);
  bool bSuccess = pData->SetValues(taskid, temp);
  delete temp;
  return bSuccess;
}
bool CudaScheduler::GetData (int taskid, int parameterid, std::vector<CUDA_TY> & data)
{
  CudaDataSet * pData = GetDataSet(parameterid);
  if (!pData)
  {
    return false;
  }

  CUDA_TY * temp = new CUDA_TY(pData->GetTaskSize());
  bool bSuccess = pData->GetValues(taskid, temp);
  if (bSuccess)
  {
    data.insert(data.begin(),temp, temp + pData->GetTaskSize());
  }
  delete temp;
  return bSuccess;
}


bool CudaScheduler::CreateData(int parameterid, int stride, bool bHost)
{

  if (m_dataSet.find(parameterid) == m_dataSet.end()) 
  {
    CudaDataSet * s = new CudaDataSet(m_info, m_taskCount, stride, bHost);
    m_dataSet[parameterid] = s;
    return true;
  }
  else
  {
    return false;
  }

}

CudaDataSet * CudaScheduler::GetDataSet (int parameterid)
{
  if (m_dataSet.find(parameterid) != m_dataSet.end())
  {
    return m_dataSet[parameterid];
  }
  return NULL;
}
