
#include "cudatask.h"
#include "layer.h"


CudaTask:: CudaTask(CudaInfo & info, int taskCount) : m_taskCount(taskCount), m_info(info)
{

}

CudaTask:: ~CudaTask()
{
  std::cout<<"CudaTask::~CudaTask"<<std::endl;
  for (DataSetMap::iterator iter = m_dataSet.begin(); iter != m_dataSet.end(); ++iter)
  {
    delete iter->second;
  }
  m_dataSet.clear();
}


bool CudaTask::SetData (int taskid, int parameterid, const std::vector<CUDA_TY> & data)
{
  CudaDataSet * pData = GetDataSet(parameterid);
  if (!pData || pData->GetTaskSize() != data.size())
  {
    std::cout<<"Could not set cuda dataset"<<std::endl;
    return false;
  }

  CUDA_TY * temp = new CUDA_TY[pData->GetTaskSize()];
  std::copy(data.begin(), data.end(), temp);
  bool bSuccess = pData->SetValues(taskid, temp);
  delete temp;
  return bSuccess;
}


bool CudaTask::GetData (int taskid, int parameterid, std::vector<CUDA_TY> & data)
{
  CudaDataSet * pData = GetDataSet(parameterid);
  if (!pData)
  {
    std::cout<<"CudaTask::GetData could not find dataset"<<std::cout;
    return false;
  }

  data.clear();
  CUDA_TY * temp = new CUDA_TY[pData->GetTaskSize()];
  bool bSuccess = pData->GetValues(taskid, temp);
  if (bSuccess)
  {
    data.insert(data.begin(),temp, temp + pData->GetTaskSize());
  }
  delete temp;
  return bSuccess;
}


bool CudaTask::CreateData(int parameterid, int stride, bool bHost)
{

  if (m_dataSet.find(parameterid) == m_dataSet.end()) 
  {
    std::cout <<"CudaTask::CreateData creating parameter: "<< parameterid <<std::endl;
    CudaDataSet * s = new CudaDataSet(m_info, m_taskCount, stride, bHost);
    m_dataSet[parameterid] = s;
    return true;
  }
  else
  {
    std::cout <<"CudaTask::CreateData dataset already exists"<<std::endl;
    return false;
  }

}

bool CudaTask::HasData(int parameterid)
{
  return GetDataSet(parameterid) != NULL;
}

CudaDataSet * CudaTask::GetDataSet (int parameterid)
{
  if (m_dataSet.find(parameterid) != m_dataSet.end())
  {
    return m_dataSet[parameterid];
  }
  return NULL;
}


PerceptronTask::PerceptronTask(CudaInfo & info, int taskCount) 
  : CudaTask (info, taskCount) 
{
  
}

bool PerceptronTask::Launch() 
{
  //The perceptron task has the following inputs:
  // param0: input vector set
  // param1: weights set
  // param2: output set

  CudaDataSet * pOutData = GetDataSet(CudaTask::OutputParam);
  CudaDataSet * pInput = GetDataSet(CudaTask::InputParam);
  CudaDataSet * pWeights = GetDataSet(CudaTask::WeightParam);

  if (!(pInput && pWeights && pOutData))
    {
      std::cout <<" Could not find a dataset"<<std::endl;
      //Raise error that something was not initialised
      return false;
    }

  //each thread block contains O number of threads
  dim3 blocksize1(pOutData->GetTaskSize(),1,1);

  //The number of neural networks to simulate
  dim3 gridsize1(m_taskCount,1,1);

  cuComputeLayer(*pInput->GetData(), *pWeights->GetData(), 
		 *pOutData->GetData(),  pInput->GetTaskSize(), gridsize1, blocksize1);
  return true;
}


MultilayerPerceptronTask::MultilayerPerceptronTask(CudaInfo & info, int taskCount) 
  : CudaTask (info, taskCount) 
{
  
}

bool MultilayerPerceptronTask::Launch() 
{
  //The perceptron task has the following inputs:
  // param0: input vector set
  // param1: weights set
  // param2: hidden set
  // param3: output set

  CudaDataSet * pOutData = GetDataSet(CudaTask::OutputParam);
  CudaDataSet * pInput = GetDataSet(CudaTask::InputParam);
  CudaDataSet * pHidden = GetDataSet(CudaTask::HiddenParam);
  CudaDataSet * pWeights = GetDataSet(CudaTask::WeightParam);

  if (!(pInput && pWeights && pHidden && pOutData))
    {
      std::cout <<" Could not find a dataset"<<std::endl;
      if (!pInput)
	std::cout <<"pInput"<<std::endl;
      if (!pWeights)
	std::cout <<"pWeights"<<std::endl;
      if (!pHidden)
	std::cout <<"pHidden"<<std::endl;
      if (!pOutData)
	std::cout <<"pOutData"<<std::endl;
      //Raise error that something was not initialised
      return false;
    }

  //each thread block contains O number of threads
  dim3 blocksize1(pHidden->GetTaskSize(),1,1);

  //The number of neural networks to simulate
  dim3 gridsize1(m_taskCount,1,1);

  cuComputeLayer(*pInput->GetData(), *pWeights->GetData(), 
		 *pHidden->GetData(),  pInput->GetTaskSize(), gridsize1, blocksize1);


  //each thread block contains O number of threads
  dim3 blocksize2(pOutData->GetTaskSize(),1,1);

  int offset = (pInput->GetSize() * pHidden->GetTaskSize()) + pHidden->GetSize();

  CUDA_TY * pSecLayerWeights = &((*pWeights->GetData())[offset]);

  cuComputeLayer(*pHidden->GetData(), pSecLayerWeights, 
		 *pOutData->GetData(),  pHidden->GetTaskSize(), gridsize1, blocksize2);
  return true;
}
