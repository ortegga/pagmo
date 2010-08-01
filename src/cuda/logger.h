

#ifndef __PAGMO_CUDA_LOGGER__
#define __PAGMO_CUDA_LOGGER__

#define LOG_CUDA_INFO
#define LOG_CUDA_ERROR

#ifdef LOG_CUDA_INFO
#define LOG_INFO(X) printf(X)
#endif 

#ifdef LOG_CUDA_ERROR
#define LOG_WARN(X) printf(X)
#endif

#endif
