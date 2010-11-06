#include "kernel_dims.h"
#include "cudatask.h"
#include "cudainfo.h"
#include "pagmo_cuda.h"

namespace cuda
{

    bool kernel_dimensions::check_shared_mem()
    {
	const cudaDeviceProp * props = m_inf->get_prop();
	if (get_shared_mem_size() > props->sharedMemPerBlock)
	{
	    return false;
	}
	return true;
    }

    bool kernel_dimensions::check_global_mem()
    {
	const cudaDeviceProp * props = m_inf->get_prop();
	if (get_global_mem_size() >  props->totalGlobalMem)
	{
	    return false;
	}
	return true;
    }

    bool kernel_dimensions::check_task_size()
    {
	const cudaDeviceProp * props = m_inf->get_prop();
	if (get_tasks_per_block() > (size_t)props->maxThreadsPerBlock)
	{
	    return false;
	}
	return true;
    }

    bool kernel_dimensions::check_problems()
    {
	const cudaDeviceProp * props = m_inf->get_prop();
	if (!check_shared_mem())
	{
	    CUDA_LOG_ERR("shared mem requirements exceed device capability", 0);
	    return false;
	}
	if (!check_task_size())
	{
	    CUDA_LOG_ERR("task cannot fit in a block", 0);
	    return false;
	}
	if (check_global_mem())
	{
	    CUDA_LOG_ERR("task cannot fit in global memory", 0);
	    return false;
	}
	return true;
    }

}
