#ifndef __PAGMO_CUDA_TASK_H__
#define __PAGMO_CUDA_TASK_H__

#include <map>
#include "layer.h"
#include "dataset.h"


namespace cuda
{

  template <typename cuda_type>
    class task
    {
    public:

      typedef std::map <int, dataset<cuda_type>* > datamap;

      enum 
      {
	inputs = 0,
	weights = 1,
	hiddens = 2,
	outputs = 3
      };

    task(info & in, unsigned int task_count) : 
      m_info(in), m_task_count(task_count)
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


      bool set_data (int taskid, int parameterid, 
		     const std::vector<cuda_type> & data)
      {
	dataset<cuda_type> * pData = get_dataset(parameterid);
	if (!pData || pData->get_tasksize() != data.size())
	  {
	    return false;
	  }

	cuda_type * temp = new cuda_type[pData->get_tasksize()];
	std::copy(data.begin(), data.end(), temp);
	bool bSuccess = pData->set_values(taskid, temp);
	delete temp;
	return bSuccess;
      }

      bool get_data (int taskid, int parameterid, std::vector<cuda_type> & data)
      {
	dataset<cuda_type> * pData = get_dataset(parameterid);
	if (!pData)
	  {
	    return false;
	  }

	data.clear();
	cuda_type * temp = new cuda_type[pData->get_tasksize()];
	bool bSuccess = pData->get_values(taskid, temp);
	if (bSuccess)
	  {
	    data.insert(data.begin(),temp, temp + pData->get_tasksize());
	  }
	delete temp;
	return bSuccess;
      }

      bool create_data(int parameterid, int stride, bool bHost)
      {

	if (m_data.find(parameterid) == m_data.end()) 
	  {
	    dataset<cuda_type> * s = new dataset<cuda_type>(m_info, m_task_count, stride, bHost);
	    m_data[parameterid] = s;
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }


      bool assign_data (int parameterid, dataset<cuda_type> * pdata, bool force = false)
      {
	if (force || m_data.find(parameterid) == m_data.end()) 
	  {
	    m_data[parameterid] = pdata;
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }

      bool has_data(int parameterid)
      {
	return get_dataset(parameterid) != NULL;
      }
      virtual bool launch () = 0;

    protected:

      datamap m_data;

      dataset<cuda_type> * get_dataset (int parameterid)
	{
	  if (m_data.find(parameterid) != m_data.end())
	    {
	      return m_data[parameterid];
	    }
	  return NULL;
	}
      unsigned int get_tasksize() 
      {
	return m_task_count;
      }
      info & m_info;
      unsigned int m_task_count;
    };

  template <typename cuda_type, int activ_type>
    class perceptron_task : public task <cuda_type>
  { 
 public:
    perceptron_task(info & inf, unsigned int taskCount) : 
        task<cuda_type>::task(inf, taskCount) 
    {}

    bool launch() 
      {

	dataset<cuda_type> * pOutData = this->get_dataset(task<cuda_type>::outputs);
	dataset<cuda_type> * pInput = this->get_dataset(task<cuda_type>::inputs);
	dataset<cuda_type> * pWeights = this->get_dataset(task<cuda_type>::weights);

	if (!(pInput && pWeights && pOutData))
	  {
	    std::cout <<" Could not find a dataset"<<std::endl;
	    //Raise error that something was not initialised
	    return false;
	  }

	//each thread block contains O number of threads
	dim3 blocksize1(pOutData->get_tasksize(),1,1);

	//The number of neural networks to simulate
	dim3 gridsize1(this->m_task_count,1,1);

	cu_compute_layer<cuda_type,activ_type >(*pInput->get_data(), *pWeights->get_data(), 
		       *pOutData->get_data(),  pInput->get_tasksize(), gridsize1, blocksize1);
	return true;
      }
  };


  template <class cuda_type, int activ_type>
    class multilayer_perceptron_task : public task <cuda_type>
  {
  public:
  multilayer_perceptron_task(info & inf, unsigned int taskCount) :
    task<cuda_type>(inf, taskCount){}
    bool launch() 
    {

      dataset<cuda_type> * pOutData = this->get_dataset(task<cuda_type>::outputs);
      dataset<cuda_type> * pInput = this->get_dataset(task<cuda_type>::inputs);
      dataset<cuda_type> * pHidden = this->get_dataset(task<cuda_type>::hiddens);
      dataset<cuda_type> * pWeights = this->get_dataset(task<cuda_type>::weights);

      if (!(pInput && pWeights && pHidden && pOutData))
	{
	  std::cout <<"failure"<<pInput <<" "<< pWeights<<" "<<pHidden<<" "<<pOutData<<std::endl;
	  return false;
	}

      dim3 blocksize1(pHidden->get_tasksize(),1,1);

      dim3 gridsize1(this->m_task_count,1,1);

      cu_compute_layer<cuda_type, activ_type>(*pInput->get_data(), *pWeights->get_data(), 
						     *pHidden->get_data(),  pInput->get_tasksize(), gridsize1, blocksize1);

      dim3 blocksize2(pOutData->get_tasksize(),1,1);

      int offset = (pInput->get_size() * pHidden->get_tasksize()) + pHidden->get_size();

      cuda_type * pSecLayerWeights = &((*pWeights->get_data())[offset]);

      cu_compute_layer<cuda_type, activ_type>(*pHidden->get_data(), pSecLayerWeights, 
						     *pOutData->get_data(),  pHidden->get_tasksize(), gridsize1, blocksize2);
      return true;
    }
  };
}


#endif
