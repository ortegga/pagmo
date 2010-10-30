

#ifndef __PAGMO_CUDA_LOGGER__
#define __PAGMO_CUDA_LOGGER__


#ifndef CUDA_LOG_DISABLED

#define CUDA_LOG_INFO(X, Y) (logger_info << (X) << (Y) << std::endl)
#define CUDA_LOG_WARN(X, Y) (logger_warn << (X) << (Y) << std::endl)
#define CUDA_LOG_ERR(X, Y) (logger_err << (X) << (Y) << std::endl)

#else

#define CUDA_LOG_INFO(x, y) ;
#define CUDA_LOG_WARN(x, y) ;
#define CUDA_LOG_ERR(x, y) ;

#endif


#include <iostream>
#include "pagmo_cuda.h"

namespace cuda
{

  class Logger : public std::ostream
  {
  public:
    Logger (bool bActivated = true)
      {
	m_bActivated = bActivated;
      }
    bool active() { return m_bActivated;}
    bool m_bActivated;
  };

  template <typename T>
    Logger & operator << (Logger & logger, const T & t)
    {
      if (logger.active())
	{
	  std::cout << t;
	}
      return logger;
    }

  Logger& operator << (Logger & logger, cudaError_t & err) ;
    

    extern Logger logger_info;
    extern Logger logger_warn;
    extern Logger logger_err;
}

#endif
