

#include "logger.h"

namespace cuda
{
 
  Logger logger_info(false);
  Logger logger_warn(true);
  Logger logger_err;

  Logger& operator << (Logger & logger, cudaError_t & err)
    {
      if (logger.active())
	{
	  logger << cudaGetErrorString(err);
	}
      return logger;
    }
  
}
