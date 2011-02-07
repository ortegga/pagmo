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

	enum completeness
	{
	    point = 1, 
	    individual = 2,
	    island = 3
	};

	kernel_dimensions (info * inf, task_profile * prof,
			   const std::string & name, completeness c = kernel_dimensions::individual) :   // <TODO> implement completeness
	m_inf(inf), 
	    m_prof(prof), 
	    m_name(name),
	    m_completeness(c)
	{      
      
	}
	virtual ~kernel_dimensions() {}

	// checks for validity

	bool check_shared_mem();

	bool check_global_mem();

	bool check_task_size();

	bool check_problems();

	//Getters 

	//assume one dimensional array of threads
	virtual dim3 get_block_dims() = 0;
	//assume a single row of blocks
	virtual dim3 get_grid_dims() = 0;
	//Amount to shared memory to allocate to each block. 
	virtual size_t get_shared_mem_size() = 0;
	//The number of tasks per block can be variable?
	virtual size_t get_tasks_per_block() = 0;
	//Amount of global data
	virtual size_t get_global_mem_size()
	{
	    return m_prof->get_total_global_chunk();
	}

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

	size_t get_indivisible_task_size()
	{
	    switch(m_completeness)
	    {
	    case island:
		return m_prof->get_job_count();
	    case individual:		
		return m_prof->get_individual_job_count();
	    case point: 
		return m_prof->get_task_size();
	    default:
		return 0;
	    };
	}

    protected:

	info * m_inf;
	task_profile * m_prof;
	std::string m_name;
	completeness m_completeness;
    public:	
	typedef  boost::shared_ptr<kernel_dimensions > ptr;    
    };



    template <size_t block_size>
    class adhoc_dimensions : public kernel_dimensions
    {
    public:
	adhoc_dimensions (cuda::info * inf, task_profile * prof, const std::string & name): 
	kernel_dimensions(inf, prof, name), m_block_count(0),  m_block_size(block_size), m_block_shared_mem(0),m_indivs_per_block(1)
	{
	    refresh();
	}

	virtual dim3 get_block_dims()
	{
	    return dim3(m_block_size ,1,1);
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
	    const cudaDeviceProp * props = m_inf->get_prop();	    

	    size_t indiv_jobs = m_prof->get_individual_job_count();

	    m_block_size -= m_block_size % indiv_jobs; // remove unused threads	    
	    if (m_block_size > indiv_jobs * m_prof->individuals )
	    {
		m_block_size = indiv_jobs * m_prof->individuals;
	    }

	    if (m_block_size % props->warpSize)
	    {
		CUDA_LOG_WARN(m_name, " adhoc kernel dimensions dont match warp size, occupancy is not optimal ", m_block_size);
	    }

	    if ( m_block_size / indiv_jobs == 0)
	    {
		CUDA_LOG_ERR(m_name, " block size is too small for job", m_block_size);
	    }

	    m_indivs_per_block = m_block_size / m_prof->get_individual_job_count();
	    m_block_shared_mem = m_indivs_per_block * m_prof->get_total_indiv_shared_chunk();
	    m_block_count = m_prof->individuals / m_indivs_per_block + (m_prof->individuals % m_indivs_per_block ? 1 : 0);

	    CUDA_LOG_INFO(m_name, "m_indivs_per_block ", m_indivs_per_block);
	    return true;
	}
    protected:

	size_t m_block_count;
	size_t m_block_size;
	size_t m_block_shared_mem;
	size_t m_indivs_per_block;
    
    };



    class block_complete_dimensions : public kernel_dimensions
    {
    public:

	block_complete_dimensions (cuda::info * inf, task_profile * prof, const std::string & name): 
	kernel_dimensions(inf, prof, name), m_block_count(0),  m_block_size(0), m_block_shared_mem(0),m_indivs_per_block(0)
	{
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
	    return m_block_shared_mem * sizeof(float);
	}

	virtual size_t get_tasks_per_block()
	{
	    return m_indivs_per_block * m_prof->get_individual_job_count();
	}

	virtual bool refresh()
	{
	    const cudaDeviceProp * props = m_inf->get_prop();	    

	    m_block_count = m_prof->individuals;
	    m_indivs_per_block = m_prof->individuals / m_block_count;
	    m_block_size = m_indivs_per_block * m_prof->get_individual_job_count();
	    m_block_shared_mem = m_indivs_per_block * m_prof->get_total_indiv_shared_chunk();

	    CUDA_LOG_INFO(m_name, "m_indivs_per_block ", m_indivs_per_block);
	    CUDA_LOG_INFO(m_name, "m_prof->get_individual_job_count() ", m_prof->get_individual_job_count());
	    return true;
	}
    protected:

	// use the block size that maximizes shared memory use
	//TODO add some code to make sure individual chunks lie in the same block

	size_t m_block_count;
	size_t m_block_size;
	size_t m_block_shared_mem;
	size_t m_indivs_per_block;
    
    };
}

#endif
