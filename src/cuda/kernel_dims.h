/*****************************************************************************
 *   Copyright (C) 2004-2009 The PaGMO development team,                     *
 *   Advanced Concepts Team (ACT), European Space Agency (ESA)               *
 *   http://apps.sourceforge.net/mediawiki/pagmo                             *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Developers  *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Credits     *
 *   act@esa.int                                                             *
 *                                                                           *
 *   This program is free software; you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation; either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program; if not, write to the                           *
 *   Free Software Foundation, Inc.,                                         *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.               *
 *****************************************************************************/


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
	kernel_dimensions(inf, prof, name), m_block_shared_mem(0),m_indivs_per_block(1)
	{
	    if (!refresh())
	    {
		pagmo_throw(value_error,"adhoc dimensions insufficient");
	    }
	}

	virtual dim3 get_block_dims()
	{
	    return m_block_size;
	}

	virtual dim3 get_grid_dims()
	{
	    return m_grid_size;
	}

	virtual size_t get_shared_mem_size()
	{
	    return m_block_shared_mem;
	}

	virtual size_t get_tasks_per_block()
	{
	    return m_indivs_per_block * m_prof->points;
	}


	virtual bool refresh()
	{
	    const cudaDeviceProp * props = m_inf->get_prop();
	    m_block_size.z = 1;
	    m_block_size.y = m_prof->get_task_size();
	    CUDA_LOG_INFO(m_name, "block_size", block_size);
	    m_block_size.x = block_size / m_block_size.y;
	    CUDA_LOG_INFO(m_name, "m_block_size.x", m_block_size.x);

	    m_block_size.x -= m_block_size.x % m_prof->points;
	    if (m_block_size.x > m_prof->islands * m_prof->individuals * m_prof->points )
	    {
		m_block_size.x = m_prof->islands * m_prof->individuals * m_prof->points;
	    }

	    if (m_block_size.x * m_block_size.y % props->warpSize)
	    {
		CUDA_LOG_WARN(m_name, " adhoc kernel dimensions dont match warp size, occupancy is not optimal ", m_block_size.x);
	    }

	    if ( m_block_size.x / m_prof->points == 0)
	    {
		CUDA_LOG_ERR(m_name, " block size is too small for individuals", m_block_size.x);
		return false;
	    }

	    m_indivs_per_block = m_block_size.x / m_prof->points; 
	    m_block_shared_mem = m_indivs_per_block * m_prof->get_total_indiv_shared_chunk();
	    if (m_block_shared_mem > m_inf->get_block_shared_mem())
	    {
		CUDA_LOG_ERR(m_name, " block size is too small for shared memory requirements", m_block_shared_mem);
		return false;
	    }
	    
	    m_grid_size.x = m_prof->individuals / m_indivs_per_block + (m_prof->individuals % m_indivs_per_block ? 1 : 0);
	    m_grid_size.y = m_grid_size.z = 1;

	    CUDA_LOG_INFO(m_name, "m_indivs_per_block ", m_indivs_per_block);
	    return true;
	}
    protected:

	size_t m_block_shared_mem;
	size_t m_indivs_per_block;
	dim3 m_block_size;
	dim3 m_grid_size;
    
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
