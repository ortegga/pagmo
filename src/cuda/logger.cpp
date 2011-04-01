

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

    Logger& operator << (Logger& logger, const dim3 & d)
    {
	logger << "< " << d.x <<", "<< d.y<<", "<< d.z << " >";
	return logger;
    }
  
}


std::ostream& operator << (std::ostream & logger, cudaError_t & err)
{
    logger << cudaGetErrorString(err);
    return logger;
}


    std::ostream& operator << (std::ostream & logger, const dim3 & d)
    {
	logger << "< " << d.x <<", "<< d.y<<", "<< d.z << " >";
	return logger;
    }
