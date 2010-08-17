#ifndef __PAGMO_CUDA_TASK_LET__
#define  __PAGMO_CUDA_TASK_LET__

#include "cudatask.h"

namespace cuda
{
  template <typename ty>
  class tasklet 
  {
  public:
    tasklet (task<ty> * pTask) : m_id (0), m_task(pTask) 
    {
    }
    void set_task(int id) { m_id = id;}
    unsigned int get_id() const { return m_id;} 
    virtual ~tasklet() {}
    task<ty> * get_task ()
    {
      return m_task;
    }
  protected:

    virtual bool  prepare_dataset(int parameter, int size)
      {
	if (m_task)
	  {
	    if (!m_task->has_data(parameter))
	      {
		return m_task->create_data(parameter, size, false);
	      }
	  }
	return false;
      }

    bool set_inputs(int parameter, const std::vector<ty> & inputs)
      {
	if (m_task)
	  {
	    if (!m_task->has_data(parameter))
	      {
		m_task->create_data(parameter, inputs.size(), false);
	      }
	    return m_task->set_data(m_id, parameter, inputs);
	  }
	return false;
      }

    virtual bool get_outputs( int parameterid, std::vector<ty> & outputs)
    {
        outputs.clear();
	if (m_task)
	  {
	    if (!m_task->has_data(parameterid))
	      {
		return false;
	      }
	    return m_task->get_data(m_id, parameterid, outputs);
	  }
	return false;
    }

    int m_id;
    task<ty> * m_task;
  };
}

#endif
