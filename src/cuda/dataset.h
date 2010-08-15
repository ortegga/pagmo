#ifndef __PAGMO_CUDA_DATASET__
#define  __PAGMO_CUDA_DATASET__

#include "cudainfo.h"
#include "boost/shared_ptr.hpp"

#include "cuda.h"
#include "cuda_runtime.h"


namespace cuda
{
  template <class cuda_type>
    class dataset
    {
    public:
    dataset(info & info, unsigned int taskcount, unsigned int stride, bool bhost):
      m_data(0),  m_task_count(taskcount), m_stride(stride),m_host(bhost), m_info(info)
      {
	if (m_host)
	  {
	    cudaMallocHost(&m_data, get_size() * sizeof(cuda_type));
	  }
	else
	  {
	    cudaMalloc(&m_data, get_size() * sizeof(cuda_type));
	  }
      }

      bool get_values(unsigned int taskId, cuda_type * sub_data)
      {

	cudaMemcpy(sub_data, &m_data[taskId * m_stride], 
		   m_stride * sizeof(cuda_type), cudaMemcpyDeviceToHost);
	return true;
      }

      bool set_values(unsigned int taskId, const cuda_type * sub_data)
      {

	cudaMemcpy(&m_data[taskId * m_stride], sub_data , 
		   m_stride * sizeof(cuda_type), cudaMemcpyHostToDevice);
	return true;
      }


      ~dataset()
	{
	  if (m_host)
	    {
	      cudaFreeHost(m_data);
	    }
	  else
	    {

	      cudaFree(m_data);
	    }
	}

      unsigned int get_tasksize() {return m_stride;}
      unsigned int get_size() {return m_task_count * m_stride;}
      cuda_type ** get_data() {return &m_data;}

    private:
      cuda_type * m_data;
      unsigned int m_task_count;
      unsigned int m_stride;
      bool m_host;
      info & m_info;
    };
}

#endif
