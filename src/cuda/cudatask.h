#ifndef __PAGMO_CUDA_TASK_H__
#define __PAGMO_CUDA_TASK_H__

#include <map>
#include "dataset.h"
#include "logger.h"
#include "boost/foreach.hpp"


namespace cuda
{

    struct task_profile
    {
    task_profile(size_t individuals_, size_t points_,  size_t task_size_)
    :individuals(individuals_),points(points_), task_size(task_size_),
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
	size_t individuals;
	size_t points;
	size_t task_size;   

	//shared memory chunks
	size_t individual_chunk;
	size_t point_chunk;
	size_t task_chunk;

	//global chunks
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
    

    template <typename cuda_type>
	class task
    {
    public:

	typedef std::map <size_t, typename dataset<cuda_type>::ptr > datamap;

    task(info & in, const std::string & name, size_t individuals, size_t task_count, size_t task_size) : 
	m_info(in), m_name(name), m_stage(0), m_profile(individuals, task_count, task_size)
	{
	
	}

	virtual ~task()
	{
	    m_data.clear();
	}


	// sets the inputs for each point. So we only need point ids
	virtual bool set_point_inputs(size_t pt, size_t parameter, const std::vector<cuda_type> & inputs, size_t size)
	{
	    if (is_valid(point, pt))
	    {
		if (!has_data(parameter))
		{
		    create_data(parameter, size, point, false);
		}
		return set_data(pt, parameter, inputs);
	    }
	    else
	    {
		CUDA_LOG_ERR(m_name, " set_input id is not valid ", pt);
	    }
	    return false;
	}

	// sets the inputs for each individual. 

	virtual bool set_individual_inputs(size_t indiv, size_t parameter, const std::vector<cuda_type> & inputs, size_t size)
	{
	    if (is_valid(individual, indiv))
	    {
		if (!has_data(parameter))
		{
		    create_data(parameter, size, individual, false);
		}
		return set_data(indiv, parameter, inputs);
	    }
	    else
	    {
		CUDA_LOG_ERR(m_name, " set_input id is not valid ", indiv);
	    }
	    return false;
	}

	// sets the inputs for each instance (point x individual)
	virtual bool set_inputs(size_t indiv, size_t pt, size_t parameter, const std::vector<cuda_type> & inputs, size_t size)
	{
	    size_t realid = indiv*m_profile.points + pt;

	    if (is_valid(element, realid))
	    {
		if (!has_data(parameter))
		{
		    create_data(parameter, size, element, false);
		}
		return set_data(realid, parameter, inputs);
	    }
	    else
	    {
		CUDA_LOG_ERR(m_name, " set_input id is not valid ", realid);
	    }
	    return false;
	}

	virtual bool get_outputs( size_t individual, size_t id, size_t parameterid, std::vector<cuda_type> & outputs)
	{
	    outputs.clear();
	    size_t realid = individual*m_profile.points + id;
	    if (!has_data(parameterid) || !is_valid(element, realid))
	    {
		CUDA_LOG_ERR(m_name, " get_outputs failed id:", id);
		CUDA_LOG_ERR(m_name, " get_outputs failed parameterid:", parameterid);
		return false;
	    }
	    return get_data(realid, parameterid, outputs);
	}

	virtual bool get_individual_outputs( size_t id, size_t parameterid, std::vector<cuda_type> & outputs)
	{
	    outputs.clear();
	    if (!has_data(parameterid) || !is_valid(individual, id))
	    {
		CUDA_LOG_ERR(m_name, " get_outputs failed id:", id);
		CUDA_LOG_ERR(m_name, " get_outputs failed parameterid:", parameterid);
		return false;
	    }
	    return get_data(id, parameterid, outputs);
	}

	virtual bool get_point_outputs( size_t id, size_t parameterid, std::vector<cuda_type> & outputs)
	{
	    outputs.clear();
	    if (!has_data(parameterid) || !is_valid(point, id))
	    {
		CUDA_LOG_ERR(m_name, " get_outputs failed id:", id);
		CUDA_LOG_ERR(m_name, " get_outputs failed parameterid:", parameterid);
		return false;
	    }
	    return get_data(id, parameterid, outputs);
	}

	virtual bool  prepare_dataset(size_t parameter, size_t size)
	{
	    if (!has_data(parameter))
	    {
		CUDA_LOG_WARN(m_name, " prepare_dataset creating dataset:", parameter);
		return create_data(parameter, size, element, false);
	    }
	    CUDA_LOG_WARN(m_name, " prepare_dataset dataset already exists:", parameter);
	    return false;
	}

	virtual bool  prepare_individual_dataset(size_t parameter, size_t size)
	{
	    if (!has_data(parameter))
	    {
		CUDA_LOG_WARN(m_name, " prepare_individual_dataset creating dataset:", parameter);
		return create_data(parameter, size, individual, false);
	    }
	    CUDA_LOG_WARN(m_name, " prepare_individual_dataset dataset already exists:", parameter);
	    return false;
	}

	virtual bool  prepare_point_dataset(size_t parameter, size_t size)
	{
	    if (!has_data(parameter))
	    {
		CUDA_LOG_WARN(m_name, " prepare_point_dataset creating dataset:", parameter);
		return create_data(parameter, size, point, false);
	    }
	    CUDA_LOG_WARN(m_name, " prepare_point_dataset dataset already exists:", parameter);
	    return false;
	}


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

	typename dataset<cuda_type>::ptr  get_dataset (size_t parameterid)
	{
	    if (m_data.find(parameterid) != m_data.end())
	    {
		return m_data[parameterid];
	    }
	    return typename dataset<cuda_type>::ptr();
	}
      
	bool has_data(size_t parameterid)
	{
	    if (get_dataset(parameterid))
	    {
		return true;
	    }
	    return false;
	}
      
	size_t get_tasksize() 
	{
	    return m_profile.task_size;
	}

	size_t get_individuals() 
	{
	    return m_profile.individuals;
	}

	void add_association(task<cuda_type> * precon_task, size_t prev_output, size_t curr_input)
	{
	    m_preconditions.add_mapping(precon_task,prev_output, curr_input);
	}


/*
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
	friend class task_queue<cuda_type>;
	task_precondition_map m_pretasks;
 */
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
		    CUDA_LOG_INFO(m_name, " retrieved ptr ", d);
		    if(!assign_data(iter1->first, d, true))
		    {
			CUDA_LOG_ERR(m_name, "could not assign data " , d);
			return false;
		    }
		}
	    }
	    return true;
	}

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

	virtual bool launch () = 0;

    protected:


	enum datatype
	{
	    unknown = 0,
	    point,
	    individual,
	    element
	};



	bool is_valid(datatype t, size_t id)
	{
	    switch (t)
	    {
	    case individual:
		return id < m_profile.individuals;
	    case point:
		return id < m_profile.points;
	    case element:
	    default:
		return id < m_profile.get_task_count();
	    }
	}

	void set_global_chunk(size_t individual_chunk, size_t point_chunk, size_t task_chunk)
	{
	    m_profile.set_global_chunk(individual_chunk, point_chunk, task_chunk);
	}


	void set_shared_chunk(size_t individual_chunk, size_t point_chunk, size_t task_chunk)
	{
	    m_profile.set_shared_chunk(individual_chunk, point_chunk, task_chunk);
	}

	task_profile  * get_profile() 
	{
	    return & m_profile;
	}

	virtual bool get_data (size_t taskid, size_t parameterid, std::vector<cuda_type> & data)
	{
	    typename dataset<cuda_type>::ptr pData = get_dataset(parameterid);
	    if (!pData)
	    {
		CUDA_LOG_ERR(m_name, "failed to get data ", parameterid);
		return false;
	    }

	    data.clear();
	    cuda_type * temp = new cuda_type[pData->get_task_size()];
	    bool bSuccess = pData->get_values(taskid, temp);
	    if (bSuccess)
	    {
		data.insert(data.begin(),temp, temp + pData->get_task_size());
	    }
	    delete temp;
	    return bSuccess;
	}


	virtual bool set_data (size_t taskid, size_t parameterid, 
			       const std::vector<cuda_type> & data)
	{
	    typename dataset<cuda_type>::ptr pData = get_dataset(parameterid);
	    if (!pData || pData->get_task_size() != data.size())
	    {
		return false;
	    }

	    cuda_type * temp = new cuda_type[pData->get_task_size()];
	    std::copy(data.begin(), data.end(), temp);
	    bool bSuccess = pData->set_values(taskid, temp);
	    delete temp;
	    return bSuccess;
	}

	size_t get_element_count(datatype settype)
	{
	    switch (settype)
	    {
	    case individual:
		return m_profile.individuals;
	    case point:
		return m_profile.points;
	    case element:
	    default:
		return m_profile.get_task_count();
	    }
	}

	virtual bool create_data(size_t parameterid, size_t stride, datatype settype, bool bHost)
	{

	    size_t instances = 0;
	    if (!has_data(parameterid)) 
	    {
	       	instances = get_element_count(settype);
		typename dataset<cuda_type>::ptr  s = typename dataset<cuda_type>::ptr(new dataset<cuda_type>(m_info, instances, stride, bHost));
		m_data[parameterid] = s;
		return true;
	    }
	    else
	    {
		return false;
	    }
	}

	friend class task_queue<cuda_type>;
	datamap m_data;
	info & m_info;
	std::string m_name;
	size_t m_stage;
	task_profile m_profile;
	task_predecessor_mappings<cuda_type> m_preconditions;
    };
}


#endif
