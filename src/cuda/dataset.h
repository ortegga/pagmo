#ifndef __PAGMO_CUDA_DATASET__
#define  __PAGMO_CUDA_DATASET__

#include "cudainfo.h"
#include "pagmo_cuda.h"
#include "logger.h"
#include "boost/shared_ptr.hpp"


namespace cuda
{
    template <class cuda_type>
	class dataset
    {
    public:
    dataset(info & info, size_t taskcount, size_t size_, bool bhost):
	m_data(0),  m_task_count(taskcount), m_size(size_),m_host(bhost), m_info(info)
	{
	    cudaError_t err;

	    size_t size = get_size() * sizeof(cuda_type);
	    if (m_host)
	    {
		CUDA_LOG_INFO("","Allocating host dataset: ", size);
		err = cudaMallocHost(&m_data, size);
	    }
	    else
	    {
		CUDA_LOG_INFO("","Allocating device dataset: ", size);
		err = cudaMalloc(&m_data, size);
	    }
	
	    if (err != cudaSuccess)
		CUDA_LOG_ERR("","Could not allocate dataset ", err);
	  
	}

	bool get_values(unsigned int taskId, cuda_type * sub_data)
	{

	    CUDA_LOG_INFO("","get dataset values for task: ", taskId);
	    cudaError_t err = cudaMemcpy(sub_data, &m_data[taskId * m_size], 
					 m_size * sizeof(cuda_type), cudaMemcpyDeviceToHost);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR("","Could not get dataset values ", err);
		return false;
	    }
	    return true; 
	}

	bool set_values(unsigned int taskId, const cuda_type * sub_data)
	{

	    CUDA_LOG_INFO("","set dataset values for task: ", taskId);
	    cudaError_t err = cudaMemcpy(&m_data[taskId * m_size], sub_data , 
					 m_size * sizeof(cuda_type), cudaMemcpyHostToDevice);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR("","Could not set dataset values ", err);
		CUDA_LOG_ERR("","\nsize: ", taskId*m_size);
		return false;
	    }
	    return true;
	}

	~dataset()
	{
	    cudaError_t err;
	    if (m_host)
	    {
		err = cudaFreeHost(m_data);
	    }
	    else
	    {

		err = cudaFree(m_data);
	    }
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR("","Failed to deallocate dataset", err);
	    }
	}

	unsigned int get_task_size() 
	{
	    return m_size;
	}
	unsigned int get_task_byte_size() 
	{
	    return m_size * sizeof(cuda_type);
	}
	unsigned int get_size() 
	{
	    return m_task_count * m_size;
	}
	unsigned int get_byte_size() 
	{
	    return m_task_count * get_task_byte_size();
	}
	cuda_type ** get_data() 
	{
	    return &m_data;
	}

    private:
	cuda_type * m_data;
	size_t m_task_count;
	size_t m_size;
	bool m_host;
	info & m_info;
    public:	
	typedef  boost::shared_ptr<dataset<cuda_type> > ptr;
    };
}



#endif
