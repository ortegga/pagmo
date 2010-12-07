#ifndef __CUDA_TASK_QUEUE__

#define __CUDA_TASK_QUEUE__

#include "cudatask.h"


template <typename ty>
namespace cuda
{
    class task_queue 
    {
    task_queue(info & inf) : m_info(inf), m_stage(0)
	{
	    
	}
	virtual ~task_queue();
	bool push_back(task<ty> * task)
	{
	    if (task)
	    {
		m_tasks.push_back(task);
		return true;
	    }
	    return false;
	}
	size_t size() 
	{
	    return m_tasks.size();
	}
	//launch all the tasks in sequence in collaboration
	//This can probably be done more optimally so that we dont
	//have to go through n tasks each time.
	virtual bool launch()
	{
	    size_t remaining = m_tasks.size();
	    
	    while(remaining)
	    {
		for (size_t  i = m_tasks.size() - 1; i >= 0 ; --i)
		{
		    task<ty> * t = m_tasks[i];
		    if (t->stage() == stage())
		    {
			bool valid = true;
			for (task_predecessor_mappings::iterator iter = t->m_preconditions.task_precondition_map.begin(); 
			     iter != t->m_preconditions.task_precondition_map.end(); ++iter)
			{
			    task<ty> *dep = (*iter).first;
			    if (dep->stage () <= t->stage() )
			    {
				//Some precondition unmet
				valid = false;
				CUDA_LOG_INFO("Unmet preconditions for task",0);
				break;
			    }
			}
			//For all preconditions of the task, make sure they have been launched and completed.
			//map outputs of the previous to the current's inputs 
			if (valid)
			{
			    if(!map_data(t))
			    {
				CUDA_LOG_ERR("failed to map data",0);
				return false;
			    }
			    if (!t->launch())
			    {
				CUDA_LOG_ERR("failed to launch task",0);
				return false;
			    }
			    t->next_stage();
			    --remaining;
			}
		    }		    
		}
	    }
	    next_stage();
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
	
    protected:

	bool map_data(task<ty> * mapto, size_t to, task<ty> * mapfrom,  size_t from)
	{
	    if (mapfrom->has_data(from) && !mapto->has_data(to))
	    {
		return mapto->assign_data(to, mapfrom->get_dataset(from));
	    }
	    return false;
	}

	bool map_data(task<ty> * t)
	{
	    task_precondition_map &mappings =  t->m_preconditions.m_pretasks;
	    task_precondition_map::const_iterator it = mappings.begin();
	    for(;it != t->mappings.end(), ++it)
	    {
		precondition_datamap_type::const_iterator itt = (*it).second.begin();
		for(;itt != (*it).second.end(); ++itt)
		{
		    if (!map_data(task<t, (*itt).first, (*it).first, (*itt).second))
		    {
			CUDA_LOG_ERR("One mapping was invalid");
			return false;
		    }
		}		
	    }
	}
	typedef std::vector<task<ty> *> task_vector;
	task_vector m_tasks;
	size_t m_stage;
	info & m_info;
    };
};


#endif
