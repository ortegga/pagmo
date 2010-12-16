#ifndef __PAGMO_CUDA_DATASET__
#define  __PAGMO_CUDA_DATASET__

#include "cudainfo.h"
#include "pagmo_cuda.h"
#include "logger.h"
#include "boost/shared_ptr.hpp"


namespace cuda
{

//////////////////////////////////////////////////////////////////////////////////
    struct data_item
    {
	enum etype
	{
	    island_type = 0x1, 
	    individual_type = 0x2, 
	    point_type = 0x4
	};

	size_t island;
	size_t individual;
	size_t point; 
	//mask type
	typedef int type;
	type m_type;

    data_item(size_t is, size_t in, size_t pt, type t) : island(is), individual(in), point(pt), m_type(t)
	    {}

	static const int individual_mask = island_type | individual_type;
	static const int point_mask = island_type | individual_type | point_type;
	static const int point_only_mask = island_type | point_type;

	static data_item individual_data(size_t island, size_t individual)
	    {
		return data_item(island,individual,1,individual_mask);
	    }
	static data_item point_data(size_t island, size_t individual, size_t point)
	    {
		return data_item(island,individual,point, point_mask);
	    }

	static data_item point_only_data(size_t island, size_t point)
	    {
		return data_item(island,1, point, point_only_mask);
	    }

	friend std::ostream & operator << (std::ostream & os, const data_item & it)
	    {
		os << "["<<it.island<<"]["<<it.individual<<"]["<<it.point<<"]";
		return os;
	    }
    };


//////////////////////////////////////////////////////////////////////////////////

    struct data_dimensions : public data_item
    {

    data_dimensions(size_t is, size_t in, size_t pt, type t) : data_item(is,in,pt,t){}
	size_t get_count()
	{
	    return (m_type & point_type ? point : 1) * (m_type & island_type ? island : 1) * (m_type & individual_type ?  individual : 1) ;
	}

	bool valid(const data_item & item)
	{
	    bool v = true;
	    if (m_type & point_type)
	    {
		v &= point > item.point;
	    }
	    if (m_type & individual_type)
	    {
		v &= individual > item.individual;
	    }
	    if (m_type & island_type)
	    {
		v &= island > item.island;
	    }
	    return v;
	}

	size_t serialize(const data_item & item)
	{
	    /*size_t result = 0;
	    if (m_type & island_type)
	    {
		result += item.island;
		result *=  individual;
	    }

	    if (m_type & individual_type)
	    {
		result += item.individual;
		result *= point;
	    }
	    if (m_type & point_type)
	    {
		result += item.point;
	    }
	    return result;*/
	    return ((item.island * individual) + item.individual)*point + item.point;
	}
    };


//////////////////////////////////////////////////////////////////////////////////

    template <class cuda_type>
	class dataset
    {
    public:
    dataset(info & info, const data_dimensions & count,  size_t size_, bool bhost):
	m_data(0),  m_count(count), m_size(size_),m_host(bhost), m_info(info)
	{
	    cudaError_t err;
	    
	    CUDA_LOG_INFO("","Data size: ", m_count);
	    size_t size = get_byte_size();
	    if (m_host)
	    {
		CUDA_LOG_INFO("","Allocating host dataset: ", size);
		err = cudaMallocHost(&m_data, size);
	    }
	    else
	    {
		CUDA_LOG_INFO("","Allocating device dataset: ", size);
		err = cudaMalloc(&m_data, size);
	    }
	
	    if (err != cudaSuccess)
		CUDA_LOG_ERR("","Could not allocate dataset ", err);
	  
	}

	bool get_values(const data_item & item, cuda_type * sub_data)
	{

	    CUDA_LOG_INFO("","get dataset values for task: ", item);
	    cudaError_t err = cudaMemcpy(sub_data, &m_data[get_serial(item)], 
					 m_size * sizeof(cuda_type), cudaMemcpyDeviceToHost);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR("","Could not get dataset values ", err);
		return false;
	    }
	    return true; 
	}

	bool set_values(const data_item & item, const cuda_type * sub_data)
	{

	    CUDA_LOG_INFO("","set dataset values for task: ", item);
	    cudaError_t err = cudaMemcpy(&m_data[get_serial(item)], sub_data , 
					 m_size * sizeof(cuda_type), cudaMemcpyHostToDevice);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR("","Could not set dataset values ", err);
		CUDA_LOG_ERR("","\n:item: ", item);
		CUDA_LOG_ERR("","\nsize: ", m_size);
		return false;
	    }
	    return true;
	}

	~dataset()
	{
	    cudaError_t err;
	    if (m_host)
	    {
		err = cudaFreeHost(m_data);
	    }
	    else
	    {

		err = cudaFree(m_data);
	    }
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR("","Failed to deallocate dataset", err);
	    }
	}

	unsigned int get_task_size() 
	{
	    return m_size;
	}
	unsigned int get_task_byte_size() 
	{
	    return m_size * sizeof(cuda_type);
	}
	unsigned int get_size() 
	{
	    return m_count.get_count() * m_size;
	}
	unsigned int get_byte_size() 
	{
	    return m_count.get_count() * get_task_byte_size();
	}
	cuda_type ** get_data() 
	{
	    return &m_data;
	}

	size_t get_serial(const data_item & item)
	{
	    return m_count.serialize(item) * m_size;
	}

    private:
	cuda_type * m_data;
	data_dimensions m_count;
	size_t m_size;
	bool m_host;
	info & m_info;
    public:	
	typedef  boost::shared_ptr<dataset<cuda_type> > ptr;
    };
    
}



#endif
