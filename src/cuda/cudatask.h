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



#ifndef __PAGMO_CUDA_TASK_H__
#define __PAGMO_CUDA_TASK_H__

#include <map>
#include "dataset.h"
#include "logger.h"
#include "boost/foreach.hpp"


namespace cuda
{

// task_profile
/*
  Describes the task's enumerative attributes. It assists the cudatask object in 
  determining the number of tasks and their granularity. It also determines the 
  size of the dataset and the method to index it.
*/
    struct task_profile
    {
    task_profile(size_t islands_, size_t individuals_, size_t points_,  size_t task_size_)
    :islands(islands_),individuals(individuals_),points(points_), task_size(task_size_),
	    individual_chunk (0), point_chunk(0), task_chunk(0), 
	    indiv_global_chunk(0), point_global_chunk(0), 
	    task_global_chunk(0)
	    {
	
	    }

	task_profile(task_profile & prof);

	size_t get_task_count()
	    {
		return points * individuals;
	    }

	size_t get_job_count()
	    {
		return task_size * points * individuals;
	    }

	size_t get_individual_job_count()
	    {
		return task_size * points;
	    }
	size_t get_task_size()
	    {
		return task_size;
	    }

	size_t get_total_indiv_shared_chunk()
	    {
		return task_chunk * get_individual_job_count ()  + point_chunk;
	    }

	size_t get_total_global_chunk()
	    {
		return (task_global_chunk * get_individual_job_count() + point_global_chunk) * individuals + indiv_global_chunk;
	    }

	void set_global_chunk(size_t i, size_t p, size_t t)
	    {
		indiv_global_chunk = i;
		point_global_chunk = p;
		task_global_chunk = t;
	    }


	void set_shared_chunk(size_t i, size_t p, size_t t)
	    {
		individual_chunk = i;
		point_chunk = p;
		task_chunk = t;
	    }

	//sample size
	size_t islands;
	size_t individuals;
	size_t points;
	size_t task_size;   

	//shared memory chunks
	size_t island_chunk;
	size_t individual_chunk;
	size_t point_chunk;
	size_t task_chunk;

	//global chunks
	size_t island_global_chunk;
	size_t indiv_global_chunk;
	size_t point_global_chunk;
	size_t task_global_chunk;

	//registers
	size_t regs_per_thread;
    };


//////////////////////////////////////////////////////////////////////////////
    template <typename cuda_type>
      class task;

    template <typename cuda_type>
	class task_queue;


/// task_predecessor_mappings class
/*
  When multiple tasks are to be executed in synch, this class records the dependencies 
  between individual tasks. 
*/    
    template <typename cuda_type>
	class task_predecessor_mappings
    {
    public:
	typedef std::map<size_t, size_t> precondition_datamap_type;
	typedef std::map<task<cuda_type> *, precondition_datamap_type * >  task_precondition_map;

	void add_mapping(task<cuda_type> * t, size_t second, size_t first)
	{
	    if (m_pretasks.find(t) == m_pretasks.end())
	    {
		precondition_datamap_type *mapping = new precondition_datamap_type();
		(*mapping)[first] = second;
		m_pretasks[t] =  mapping;
	    }
	    else
	    {
		(*m_pretasks[t])[first] = second;
	    }
	}
	class Ftor 
	{
	public:
	    void operator () (std::pair<const task<cuda_type> *, precondition_datamap_type *> & p)
	    {
		delete p.second;
	    }
	};
	~task_predecessor_mappings() 
	{
	    typename task_precondition_map::iterator iter;
	    for (iter = m_pretasks.begin(); iter != m_pretasks.end(); ++iter)
	    {
		delete iter->second;
	    }
	}
    protected:
	friend class task<cuda_type>;
	task_precondition_map m_pretasks;
    };

//////////////////////////////////////////////////////////////////////////////
/// task class
/*
  Handles each cuda task 

*/    

    template <typename cuda_type>
	class task
    {
    public:

	//data is stored against an enum in a map. Just a consistent way of specifying 
	// data for cuda tasks. 
	typedef std::map <size_t, typename dataset<cuda_type>::ptr > datamap;

	//ctor
    task(info & in, const std::string & name, size_t islands, size_t individuals, size_t task_count, size_t task_size) : 
	m_info(in), m_name(name), m_stage(0), m_profile(islands, individuals, task_count, task_size)
	{
	
	}

	//dtor
	virtual ~task()
	{
	    clear();
	}
	
	virtual void clear()
	{
	    m_data.clear();
	}

	// sets the inputs of parameter for each task defined by item. Create if necessary
	virtual bool set_inputs(const data_item & item, size_t parameter, const std::vector<cuda_type> & inputs, size_t size, const std::string & name = "")
	{
	    if (!has_data(parameter))
	    {
		data_dimensions dims = create_data_dims(item.m_type);
		create_data(parameter, size, dims, name);
	    }
	    return set_data(item, parameter, inputs);
	}

	// gets the outputs of parameter for each task defined by item.
	virtual bool get_outputs( const data_item &  item, size_t parameterid, std::vector<cuda_type> & outputs)
	{
	    outputs.clear();
	    if (!has_data(parameterid))
	    {
		CUDA_LOG_ERR(m_name, "get_outputs failed id:", item);
		CUDA_LOG_ERR(m_name, "get_outputs failed parameterid:", parameterid);
		return false;
	    }
	    return get_data(item, parameterid, outputs);
	}

	// prepare output dataset. The outputs arent automatically created and have to be created manually

	virtual bool  prepare_dataset(data_item::type type, size_t parameter, size_t size, const std::string & name = "")
	{
	    if (!has_data(parameter))
	    {
		CUDA_LOG_INFO(m_name, "prepare_dataset creating dataset:", parameter);		
		data_dimensions dims = create_data_dims(type);
		return create_data(parameter, size, dims, this->m_name + ":" + name);
	    }
	    CUDA_LOG_WARN(m_name, "prepare_dataset dataset already exists:", name);
	    return false;
	}

	//assign_data used to link data from one task to another. Can be forced if task already exists
	virtual bool assign_data (size_t parameterid, typename dataset<cuda_type>::ptr  pdata, bool force = false)
	{
	    if (force || !has_data(parameterid)) 
	    {
		m_data[parameterid] = pdata;
		return true;
	    }
	    else
	    {
		CUDA_LOG_ERR(m_name, "Could not assign_data parameterid:", parameterid);
		return false;
	    }
	}


	//Get a specific parameter for the task.
	typename dataset<cuda_type>::ptr  get_dataset (size_t parameterid)
	{
	    if (m_data.find(parameterid) != m_data.end())
	    {
		return m_data[parameterid];
	    }
	    return typename dataset<cuda_type>::ptr();
	}
      
	//Check parameter already exists
	bool has_data(size_t parameterid)
	{
	    if (get_dataset(parameterid))
	    {
		return true;
	    }
	    return false;
	}
      
	//return the size of each task  - The number of threads to be assigned to each task.
	size_t get_tasksize() 
	{
	    return m_profile.task_size;
	}

	size_t get_individuals() 
	{
	    return m_profile.individuals;
	}


	//Link this task to another based on the prev_output, curr_input pair - The output of the previous
	//task is a pre-condition to the current task on the curr_input
	void add_association(task<cuda_type> * precon_task, size_t prev_output, size_t curr_input)
	{
	    m_preconditions.add_mapping(precon_task,prev_output, curr_input);
	}


	//Link up the tasks based on their pre-conditions
	bool execute_associations()
	{
	    typename task_predecessor_mappings<cuda_type>::task_precondition_map::iterator iter;
	    for (iter = m_preconditions.m_pretasks.begin();iter != m_preconditions.m_pretasks.end(); ++iter)
	    {
		typename task_predecessor_mappings<cuda_type>::precondition_datamap_type * sub_cond =  iter->second;
		task<cuda_type> * t = iter->first;
		typename task_predecessor_mappings<cuda_type>::precondition_datamap_type::iterator iter1 = sub_cond->begin();
		for (;iter1 != sub_cond->end(); ++iter1)
		{
		    typename dataset<cuda_type>::ptr d = t->get_dataset(iter1->second);
		    CUDA_LOG_INFO(m_name, "retrieved ptr", d);
		    if(!assign_data(iter1->first, d, true))
		    {
			CUDA_LOG_ERR(m_name, "could not assign data " , d);
			return false;
		    }
		}
	    }
	    return true;
	}
	
	//Task stages not completed yet. 
	size_t stage ()
	{
	    return m_stage;
	}
	void reset_stage()
	{
	    m_stage = 0;
	}
	void next_stage()
	{
	    ++m_stage;
	}


	//Workhorse of the task class, this is responsible for the actual call to the __kernel__
	virtual bool launch () = 0;

    protected:


	//Amount of global memory used by the individual granularity levels of the cuda task. 
	// some intelligence is needed in determining this. Its used primarily to check that
	// the amount of global memory needed doesnt exceed the amount available for the GPU
	void set_global_chunk(size_t individual_chunk, size_t point_chunk, size_t task_chunk)
	{
	    m_profile.set_global_chunk(individual_chunk, point_chunk, task_chunk);
	}


	//Amount of shared memory used by the individual granularity levels of the cuda task. 
	// some intelligence is needed in determining this. Its used primarily to check that
	// the amount of shared memory needed doesnt exceed the amount available for the GPU
	// per block
	void set_shared_chunk(size_t individual_chunk, size_t point_chunk, size_t task_chunk)
	{
	    m_profile.set_shared_chunk(individual_chunk, point_chunk, task_chunk);
	}

	task_profile  * get_profile() 
	{
	    return & m_profile;
	}

	//Helper function to get data for job defined by item, Does not create if dataset doesnt exist
	virtual bool get_data (const data_item& item, size_t parameterid, std::vector<cuda_type> & data)
	{
	    typename dataset<cuda_type>::ptr pData = get_dataset(parameterid);
	    if (!pData)
	    {
		CUDA_LOG_ERR(m_name, "failed to get data", parameterid);
		return false;
	    }
	    return pData->get_data(item, data);
	}


	//Helper function to set data for job defined by item, Does not create if dataset doesnt exist
	virtual bool set_data (const data_item & item, size_t parameterid, 
			       const std::vector<cuda_type> & data)
	{
	    typename dataset<cuda_type>::ptr pData = get_dataset(parameterid);
	    if (!pData || pData->get_task_size() != data.size())
	    {
		return false;
	    }

	    return pData->set_data(item, data);
	}

	//Create dataset instance for the parameter defined by parameterid, stride defines the size of each datum and dims 
	//define the number of tasks.
	virtual bool create_data(size_t parameterid, size_t stride, const data_dimensions & dims, const std::string & name)
	{

	    if (!has_data(parameterid)) 
	    {
		typename dataset<cuda_type>::ptr  s = typename dataset<cuda_type>::ptr(new dataset<cuda_type>(m_info, dims, stride, 
													      this->m_name + ":" + name));
		m_data[parameterid] = s;
		return true;
	    }
	    else
	    {
		return false;
	    }
	}

	//Creates a data_dimensions instance for the task. 
	data_dimensions create_data_dims (const data_item::type & type)
	{
	    data_dimensions dims( m_profile.islands, m_profile.individuals, m_profile.points, type);
	    return dims;
	}

	friend class task_queue<cuda_type>;
	//collection of in/out parameters for the cuda task. 
	datamap m_data;
	//reference to information
	info & m_info;
	//description for logging reasons
	std::string m_name;
	//staging for task, not implemented yet
	size_t m_stage;
	//meta information of the cudatask
	task_profile m_profile;
	//pre-condition tasks
	task_predecessor_mappings<cuda_type> m_preconditions;
    };
}


#endif
