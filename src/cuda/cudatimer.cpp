

#include "cudatimer.h"
#include <iostream>



CudaTimer::CudaTimer(const std::string & desc, cudaStream_t st) : 
  m_start(0), m_stop(0), m_started(false), 
  m_description(desc), m_elapsed(0.0f), m_stream(st)
{
  
}

CudaTimer::~CudaTimer()
{
  if (m_started)
  {
    Stop();
  }
}

bool CudaTimer::Start()
{
  if (!m_started)
  {
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
    cudaEventRecord(m_start, 0);
    m_elapsed = 0.0f;
    return m_started = true;
  }
  return false;
}

bool CudaTimer::Stop()
{
  if (m_started)
  {
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
    cudaEventElapsedTime(&m_elapsed, m_start, m_stop);
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
    m_started = false;
    return true;
  }
  return false;
}


 std::ostream & operator <<(std::ostream & os, const CudaTimer & t)
{
  if (t.m_started)
  {
    os<<"Event: "<<t.m_description<<" "<<" invalid state "<<std::endl;
  }
  os<<"Event: "<<t.m_description<<" "<<t.m_elapsed<<" ms elapsed"<<std::endl;
  return os;
}


ScopedCudaTimer::~ScopedCudaTimer()
{
  Stop();
  CudaTimesKeeper::GetInstance().AddTime(*this);
}


CudaTimesKeeper::CudaTimesKeeper(std::ostream & os) 
  : m_ostream(os)
{
  
}

CudaTimesKeeper::~CudaTimesKeeper() {}

bool CudaTimesKeeper::AddTime(CudaTimer & t)
{
    m_ostream<<t;
    return true;
}

boost::shared_ptr<CudaTimesKeeper> CudaTimesKeeper::m_instancePtr;

CudaTimesKeeper & CudaTimesKeeper::GetInstance()
{
  if (!m_instancePtr)
    {
      m_instancePtr =  boost::shared_ptr< CudaTimesKeeper> (new CudaTimesKeeper(std::cout));
    }
  return *m_instancePtr;
}

