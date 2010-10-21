#ifndef __PAGMO_CUDA_KERNEL_DIMS__
#define __PAGMO_CUDA_KERNEL_DIMS__

#include "cudatask.h"
#include "cudatimer.h"
#include <iostream>

namespace cuda
{

  class kernel_dimensions
  {
  public:
    kernel_dimensions (info * inf, task_profile * prof) :  
    m_inf(inf), m_prof(prof)
    {      
      check_problems();
    }
    virtual ~kernel_dimensions() {}

    // checks for validity

    bool check_shared_mem()
    {
      /*const cudaDeviceProp & props = m_inf.get_prop();
      if (m_prof->shared_chunk > props.sharedMemPerBlock)
	{
	  return false;
	  }*/
      return true;
    }

    bool check_global_mem()
    {
      /*const cudaDeviceProp & props = m_inf.get_prop();
      if (m_prof->global_chunk*m_prof->task_count >  props.totalGlobalMem)
	{
	  return false;
	  }*/
      return true;
    }

    bool check_task_size()
    {
      /*      const cudaDeviceProp & props = m_inf.get_prop();
      if (m_prof->task_size > (size_t)props.maxThreadsPerBlock)
	{
	  return false;
	  }*/
      return true;
    }

    bool check_problems()
    {
      /*const cudaDeviceProp & props = m_inf.get_prop();
      if (m_prof->shared_chunk > props.sharedMemPerBlock)
	{
	  CUDA_LOG_ERR("shared mem requirements exceed device capability", m_prof->shared_chunk);
	  return false;
	}
      if (m_prof->task_size > (size_t)props.maxThreadsPerBlock)
	{
	  CUDA_LOG_ERR("task cannot fit in a block", m_prof->task_size);
	  return false;
	}
      if (m_prof->global_chunk >  props.sharedMemPerBlock)
	{
	  CUDA_LOG_ERR("task cannot fit in global memory", m_prof->global_chunk);
	  return false;
	  }*/
      return true;
    }

    //Getters 

    //assume one dimensional array of threads
    virtual dim3 get_block_dims() = 0;
    //assume a single row of blocks
    virtual dim3 get_grid_dims() = 0;
    //Amount to allocate to each block. 
    //Will be the max of something
    virtual size_t get_shared_mem_size() = 0;
    //The number of tasks per block can be variable?
    virtual size_t get_tasks_per_block() = 0;

    virtual bool refresh() 
    {
      return true;
    }

    //Possibly unused
    virtual size_t get_tasks_per_thread()
    {
      return 1;
    }

    // simple attributes
    virtual size_t get_task_size()
    {
      return m_prof->task_size;
    }
    virtual size_t get_individuals()
    {
      return m_prof->individuals;
    }
    virtual size_t get_points()
    {
      return m_prof->points;
    }

  protected:

    info * m_inf;
    task_profile * m_prof;
    
  };



  // Each block contains an integer number of individuals (which cant be subdivided further)
  // smallest sized blocks that maximize occupancy

  // block size depends on:
  // shared memory availability
  // optimal warp size
  // max threads per block allowed

  //TODO---------- fix this class
  class block_complete_dimensions : public kernel_dimensions
  {
  public:
    block_complete_dimensions (cuda::info * inf, task_profile * prof): 
    kernel_dimensions(inf, prof), m_block_count(0),  m_block_size(0), m_block_shared_mem(0),m_indivs_per_block(0)
      {
	std::cout<<"block_complete_dimensions::block_complete_dimensions"<<std::endl;
	refresh();
      }

    virtual dim3 get_block_dims()
    {
      return dim3(m_block_size,1,1);
    }

    virtual dim3 get_grid_dims()
    {
      return dim3(m_block_count,1,1);
    }

    virtual size_t get_shared_mem_size()
    {
      return m_block_shared_mem;
    }

    virtual size_t get_tasks_per_block()
    {
      return m_indivs_per_block * m_prof->get_individual_job_count();
    }

    virtual bool refresh()
    {
      size_t jobs = m_prof->get_job_count();
      const cudaDeviceProp * props = m_inf->get_prop();	    

      size_t block_size = 1;

      block_size = use_thread_suggestion(props,m_prof);

      size_t shared_suggestion = use_shared_mem_suggestion(props, m_prof);

      if (shared_suggestion < block_size)
	m_block_size = shared_suggestion;
      else
	m_block_size = block_size;

      m_indivs_per_block = m_block_size / m_prof->get_individual_job_count();
      
      // Assuming equal threads go in all the blocks.
      m_block_count = m_prof->individuals / m_indivs_per_block + (m_prof->individuals % m_indivs_per_block ? 1 : 0);

      //m_block_count = jobs / m_block_size + (jobs % m_block_size ? 1 : 0);
      //m_indivs_per_block = m_prof->individuals / m_block_count;// + (m_prof->individuals % m_block_count ? 1 : 0);

      m_block_shared_mem = m_indivs_per_block * m_prof->get_total_indiv_shared_chunk();

      //m_indivs_per_block = (m_block_size / m_prof->get_individual_job_count()) * m_prof->task_size;


      std::cout<<"m_block_count "<<m_block_count<<std::endl;
      std::cout<<"m_indivs_per_block "<<m_indivs_per_block<<std::endl;
      std::cout<<"m_prof->get_total_indiv_shared_chunk "<<m_prof->get_total_indiv_shared_chunk()<<std::endl;
      std::cout<<"m_prof->get_individual_job_count()"<<m_prof->get_individual_job_count()<<std::endl;
      std::cout<<"Kernel dimensions "<<m_block_size<<" "<<m_block_count<<" "<<m_block_shared_mem<<" "<<m_indivs_per_block<<std::endl;
      return true;
    }
  protected:

    // use the block size that maximizes shared memory use
    //TODO add some code to make sure individual chunks lie in the same block
    size_t use_shared_mem_suggestion( const cudaDeviceProp * props, task_profile * prof)
    {
      /*size_t indiv_jobs =  prof->get_individual_job_count();
      size_t totalmem = prof->get_total_indiv_shared_chunk(); 

      size_t block_count = totalmem / props.sharedMemPerBlock + (totalmem % props.sharedMemPerBlock ? 1 : 0);
      for (size_t i = props.warpSize; i <= props.maxThreadsPerBlock; i=i<<1 )
	{
	  if (i * block_count > indiv_jobs * tasksize)
	    {
	      CUDA_LOG_INFO("shared memory suggestion: ", i);
	      return i;
	    }
	}
      CUDA_LOG_WARN("could not get a valid shared memory suggestion", totalmem);
      return props.warpSize;*/
      return 200000;
    }

    // gives blocksize with the least residue (most occupancy)
    // given a number of indivisible jobs, find the block size/block count that fits the most jobs with the least residue.
    size_t use_thread_suggestion(const cudaDeviceProp * props, task_profile * prof)
    {
      size_t indiv_jobs =  prof->get_individual_job_count();
      size_t minval = props->maxThreadsPerBlock;
      size_t minind = 0;
      size_t i = props->warpSize;

      for (; i <= props->maxThreadsPerBlock; i=i<<1 )
	{
	  if (i >= indiv_jobs && minval >= (i - (indiv_jobs % i)))
	    {
	      minval = i - ( indiv_jobs % i);			 
	      minind = i;
	    }
	}
      std::cout<<"warp size suggestion: " <<minind <<" for "<<indiv_jobs<<" jobs"<<std::endl;
      return minind;
      
    }

    size_t m_block_count;
    size_t m_block_size;
    size_t m_block_shared_mem;
    size_t m_indivs_per_block;
    
  };


  class learning_dimensions : public kernel_dimensions
  {
  public:
  learning_dimensions(cuda::info * inf, task_profile * prof) : 
    kernel_dimensions (inf, prof), 
      m_duration("learning dimensions timer") , m_started(false)
	{
	  refresh();
	}

    bool start()
    {
      if (m_duration.start()) 
	return m_started = true;
      return false;
    }
    bool stop()
    {
      if(m_duration.stop())
	{
	  m_started = false;
	  return true;
	}
      return false;
    }
    bool refresh()
    {
      if(!m_started)
	{
	  //Add code to refresh
	  return true;
	}
      return false;
    }
  protected:
    cuda::timer m_duration;
    bool m_started;
  };

}

#endif
