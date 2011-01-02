

#ifndef __PAGMO_CUDA_LOGGER__
#define __PAGMO_CUDA_LOGGER__


#ifndef CUDA_LOG_DISABLED

//#define CUDA_LOG_INFO(N, X, Y) (std::cout <<std::endl<< "[I] " << (N) <<" "<< (X) <<" "<< (Y) << std::endl)
#define CUDA_LOG_WARN(N, X, Y) (std::cout <<std::endl<< "[W] "<< (N) <<" "<< (X) <<" "<< (Y) << std::endl)
#define CUDA_LOG_ERR(N, X, Y)  (std::cout <<std::endl<< "[E] "<< (N) <<" "<< (X) <<" "<< (Y) << std::endl)
#define CUDA_LOG_INFO(n, x, y) ;

#else

//#define CUDA_LOG_INFO(x, y) ;
//#define CUDA_LOG_WARN(x, y) ;
//#define CUDA_LOG_ERR(x, y) ;

#endif


#include <iostream>
#include "pagmo_cuda.h"
#include "../exceptions.h"



std::ostream& operator << (std::ostream & logger, cudaError_t & err) ;    

namespace cuda
{
    
    struct cuda_error: public p_base_exception 
    {
        cuda_error(const std::string &s): 
	p_base_exception(s) 
	{
	    
	}

	cuda_error (cudaError_t & err) : 
	p_base_exception(cudaGetErrorString(err))
	{

	}
    };
    
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
