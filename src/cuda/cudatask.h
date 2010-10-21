#ifndef __PAGMO_CUDA_TASK_H__
#define __PAGMO_CUDA_TASK_H__

#include <map>
#include "dataset.h"
#include "logger.h"


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

  template <typename cuda_type>
    class task
    {
    public:

      typedef std::map <size_t, dataset<cuda_type>* > datamap;

    task(info & in, size_t individuals, size_t task_count, size_t task_size) : 
      m_info(in), m_profile(individuals, task_count, task_size)
      {
	
      }

      virtual ~task()
	{
	  typename datamap::iterator iter;
	  for (iter = m_data.begin(); iter != m_data.end(); ++iter)
	    {
	      delete iter->second;
	    }
	  m_data.clear();
	}


      virtual bool set_individual_inputs(size_t individual, size_t id, size_t parameter, const std::vector<cuda_type> & inputs)
      {
	size_t realid = individual*m_profile.points + id;
	std::cout<<"set_individual_inputs: Id"<<realid<<std::endl;
	if (is_valid(realid))
	  {
	    if (!has_data(parameter))
	      {
		create_data(parameter, inputs.size(), true, false);
	      }
	    return set_data(realid, parameter, inputs);
	  }
	else
	  {
	    CUDA_LOG_ERR(" set_input id is not valid ", realid);
	  }
	return false;
      }

      virtual bool set_inputs(size_t id, size_t parameter, const std::vector<cuda_type> & inputs)
      {
	if (is_valid(id))
	  {
	    if (!has_data(parameter))
	      {
		create_data(parameter, inputs.size(), false, false);
	      }
	    return set_data(id, parameter, inputs);
	  }
	else
	  {
	    CUDA_LOG_ERR(" set_input id is not valid ", id);
	  }
	return false;
      }

      virtual bool get_individual_outputs( size_t individual, size_t id, size_t parameterid, std::vector<cuda_type> & outputs)
      {
	outputs.clear();
	size_t realid = individual*m_profile.points + id;
	std::cout<<"ID: "<<realid<<std::endl;
	if (!has_data(parameterid) || 	!is_valid(realid))
	  {
	    CUDA_LOG_ERR(" get_outputs failed id:", id);
	    CUDA_LOG_ERR(" get_outputs failed parameterid:", parameterid);
	    return false;
	  }
	return get_data(realid, parameterid, outputs);
      }

      virtual bool get_outputs( size_t id, size_t parameterid, std::vector<cuda_type> & outputs)
      {
	outputs.clear();
	if (!has_data(parameterid) || 	!is_valid(id))
	  {
	    CUDA_LOG_ERR(" get_outputs failed id:", id);
	    CUDA_LOG_ERR(" get_outputs failed parameterid:", parameterid);
	    return false;
	  }
	return get_data(id, parameterid, outputs);
      }

      virtual bool  prepare_dataset(size_t parameter, size_t size)
      {
	if (!has_data(parameter))
	  {
	    CUDA_LOG_WARN(" prepare_dataset creating dataset:", parameter);
	    return create_data(parameter, size, false, false);
	  }
	CUDA_LOG_WARN(" prepare_dataset dataset already exists:", parameter);
	return false;
      }

      virtual bool  prepare_individual_dataset(size_t parameter, size_t size)
      {
	if (!has_data(parameter))
	  {
	    CUDA_LOG_WARN(" prepare_dataset creating dataset:", parameter);
	    return create_data(parameter, size, true, false);
	  }
	CUDA_LOG_WARN(" prepare_individual_dataset dataset already exists:", parameter);
	return false;
      }


      virtual bool assign_data (size_t parameterid, dataset<cuda_type> * pdata, bool force = false)
      {
	if (force || !has_data(parameterid)) 
	  {
	    m_data[parameterid] = pdata;
	    return true;
	  }
	else
	  {
	    CUDA_LOG_ERR("Could not assign_data parameterid:", parameterid);
	    return false;
	  }
      }

      dataset<cuda_type> * get_dataset (size_t parameterid)
	{
	  if (m_data.find(parameterid) != m_data.end())
	    {
	      return m_data[parameterid];
	    }
	  return NULL;
	}
      
      bool has_data(size_t parameterid)
      {
	return get_dataset(parameterid) != NULL;
      }
      
      size_t get_tasksize() 
      {
	return m_profile.task_size;
      }

      size_t get_individuals() 
      {
	return m_profile.individuals;
      }

      bool is_valid(size_t id)
      {
	return id < m_profile.get_task_count();
      }

      virtual bool launch () = 0;

    protected:


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
	dataset<cuda_type> * pData = get_dataset(parameterid);
	if (!pData)
	  {
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
	dataset<cuda_type> * pData = get_dataset(parameterid);
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

      virtual bool create_data(size_t parameterid, size_t stride, bool bindividual, bool bHost)
      {

	size_t instances = 0;
	if (!has_data(parameterid)) 
	  {
	    if (bindividual)
	      {
		instances = m_profile.get_task_count();
	      }
	    else
	      {
		instances = m_profile.get_task_count() / get_individuals();
	      }
	    dataset<cuda_type> * s = new dataset<cuda_type>(m_info, instances, stride, bHost);
	    m_data[parameterid] = s;
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }


      datamap m_data;
      info & m_info;
      task_profile m_profile;
    };
}


#endif
