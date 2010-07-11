

#include "cudainfo.h"
#include "boost/foreach.hpp"

CudaInfo::CudaInfo()
{
  if(!Load())
  {
    //pagmo throw;
  }
}

bool CudaInfo::Load()
{
  int devNum = 0;
  if (cudaGetDeviceCount(&devNum) != cudaSuccess || devNum == 0)
  {
    return false;
  }
  for (int i = 0; i<devNum; ++i)
  {
    CudaDeviceInfo * pInfo = new CudaDeviceInfo(i);
    m_devices.push_back(pInfo);
  }
  return true;
}

CudaInfo::~CudaInfo()
{
  BOOST_FOREACH(CudaDeviceInfo * info, m_devices)
  {
    delete info;
  }
}


CudaDeviceInfo::CudaDeviceInfo(int devId) : m_id(devId)
{
  if(!Load(m_id))
  {
    //throw pagmo exception
  }
}

bool CudaDeviceInfo::Load(int devId)
{
  if(cudaGetDeviceProperties(&m_prop,devId) != cudaSuccess)
  {
    return true;
  }
  return false;
}


std::ostream & operator <<(std::ostream & os, const CudaInfo & dev)
{
  BOOST_FOREACH(CudaDeviceInfo * info, dev.m_devices)
  {
    os << *info;
  }
  return os;
}

std::ostream & operator <<(std::ostream & os, const CudaDeviceInfo & dev)
{
  os <<"Device ("<<dev.m_id<<") - "<<dev.m_prop.name<<std::endl;

  os<<"clock: "<<(dev.m_prop.clockRate/1000)<<"MHz x "<<dev.m_prop.multiProcessorCount<<" multiprocessors"<<std::endl;
  os<<"memory: "
    <<(dev.m_prop.totalGlobalMem>>20)<<"MB(global) "
    <<(dev.m_prop.sharedMemPerBlock>>10)<<"KB(shared/block) "
    <<(dev.m_prop.totalConstMem>>10)<<"KB(const)"
    <<std::endl;
  os<<"compute capable: "<<dev.m_prop.major<<"."<<dev.m_prop.minor;
  os<<" device overlap: "<<dev.m_prop.deviceOverlap
    <<" timeout enabled: "<<dev.m_prop.kernelExecTimeoutEnabled
    <<" integrated mem: "<<dev.m_prop.integrated
    <<" map host mem: "<<dev.m_prop.canMapHostMemory
    <<std::endl;

  os<<"registers per block: "<<dev.m_prop.regsPerBlock<<std::endl;
  os<<"warp size: "<<dev.m_prop.warpSize<<std::endl;
  os<<"memory pitch (bytes): "<<dev.m_prop.memPitch<<std::endl;
  os<<"threads per block: " <<dev.m_prop.maxThreadsPerBlock<<std::endl;

  os<<"thread dimensions: " 
    << dev.m_prop.maxThreadsDim[0]<<"x"
    << dev.m_prop.maxThreadsDim[1]<<"x"
    << dev.m_prop.maxThreadsDim[2];
  os<<" grid size: " 
    << dev.m_prop.maxGridSize[0]<<"x"
    << dev.m_prop.maxGridSize[1]<<"x"
    << dev.m_prop.maxGridSize[2] << std::endl;

  return os;
}
