

#ifndef __PAGMO_CUDA_TIMER_H__
#define __PAGMO_CUDA_TIMER_H__

#include <string>
#include <ostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "boost/shared_ptr.hpp"


class CudaTimer;

class CudaTimesKeeper 
{
 public:
  ~CudaTimesKeeper();
  static CudaTimesKeeper & GetInstance();
  bool AddTime(CudaTimer & t);
 private:
  CudaTimesKeeper(std::ostream & os);
  CudaTimesKeeper(CudaTimesKeeper & ); // no copy
  //  friend std::ostream & operator <<(std::ostream & os, const CudaTimesKeeper & tk );
  std::ostream & m_ostream;
  static boost::shared_ptr<CudaTimesKeeper> m_instancePtr;
};

class CudaTimer
{
 public: 
  CudaTimer(const std::string & description, cudaStream_t st = 0);
  virtual bool Start();
  virtual bool Stop();
  virtual bool Started() { return m_started;} 
  virtual float GetElapsed() { return m_elapsed;} 
  virtual ~CudaTimer();
  friend std::ostream & operator <<(std::ostream & os, const CudaTimer & t);
 protected:
  bool m_started;
  std::string m_description;
  cudaEvent_t m_start, m_stop;
  cudaStream_t m_stream;
  float m_elapsed;
};

class ScopedCudaTimer : public CudaTimer
{
 public:
 ScopedCudaTimer(const std::string & description, cudaStream_t st = 0) 
   : CudaTimer(description, st)
  {
    Start();
  }
  ~ScopedCudaTimer();
};

#endif
