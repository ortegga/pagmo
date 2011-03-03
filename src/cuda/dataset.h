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
	static const int island_mask = island_type ;

	static data_item individual_data(size_t island, size_t individual)
	    {
		return data_item(island,individual,0,individual_mask);
	    }
	static data_item point_data(size_t island, size_t individual, size_t point)
	    {
		return data_item(island,individual,point, point_mask);
	    }

	static data_item point_only_data(size_t island, size_t point)
	    {
		return data_item(island,0, point, point_only_mask);
	    }

	static data_item island_data(size_t island)
	    {
		return data_item(island,0, 0, island_mask);
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
	    size_t result = 0;
	    if (m_type & island_type)
	    {
		result += item.island;
	    }

	    if (m_type & individual_type)
	    {
		result *=  individual;
		result += item.individual;
	    }
	    if (m_type & point_type)
	    {
		result *= point;
		result += item.point;
	    }
	    return result;
	}
    };


//////////////////////////////////////////////////////////////////////////////////

    template <class cuda_type>
	class dataset
    {
    public:
    dataset(info & info, const data_dimensions & count,  size_t size_, const std::string & name, bool bhost):
	m_data(0),  m_count(count), m_size(size_),m_host(bhost), m_info(info), m_name(name)
	{
	    cudaError_t err;
	    
	    CUDA_LOG_INFO(m_name, "Data size:", m_count);
	    size_t size = get_byte_size();
	    if (m_host)
	    {
		CUDA_LOG_INFO(m_name,"Allocating host dataset:", size);
		err = cudaMallocHost((void **)&m_data, size);
	    }
	    else
	    {
		CUDA_LOG_INFO(m_name,"Allocating device dataset:", size);
		err = cudaMalloc((void **)&m_data, size);
		CUDA_LOG_INFO(m_name,"Allocating device dataset: ", m_data);
	    }
	
	    if (err != cudaSuccess)
		CUDA_LOG_ERR(m_name,"Could not allocate dataset:", err);
	  
	}


	virtual bool get_data (const data_item& item, std::vector<cuda_type> & data)
	{

	    data.clear();
	    cuda_type * temp = new cuda_type[this->get_task_size()];
	    bool bSuccess = this->get_values(item, temp);
	    if (bSuccess)
	    {
		data.insert(data.begin(),temp, temp + this->get_task_size());
	    }
	    delete temp;
	    return bSuccess;
	}

	virtual bool set_data (const data_item & item, const std::vector<cuda_type> & data)
	{
	    cuda_type * temp = new cuda_type[this->get_task_size()];
	    std::copy(data.begin(), data.end(), temp);
	    bool bSuccess = this->set_values(item, temp);
	    delete temp;
	    return bSuccess;
	}

	virtual ~dataset()
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
		CUDA_LOG_ERR(m_name,"Failed to deallocate dataset", err);
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

    protected:

	bool get_values(const data_item & item, cuda_type * sub_data)
	{

	    CUDA_LOG_INFO(m_name,"get dataset values for task:", item);
	    CUDA_LOG_INFO(m_name,"get dataset values for task:", get_serial(item));
	    cudaError_t err = cudaMemcpy(sub_data, &m_data[get_serial(item)], 
					 m_size * sizeof(cuda_type), cudaMemcpyDeviceToHost);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(m_name,"Could not get dataset values", err);
		return false;
	    }
	    return true; 
	}

	bool set_values(const data_item & item, const cuda_type * sub_data)
	{

	    CUDA_LOG_INFO(m_name,"set dataset values for task:", item);
	    CUDA_LOG_INFO(m_name,"serialized:", m_count.serialize(item));
	    cudaError_t err = cudaMemcpy(&m_data[get_serial(item)], sub_data , 
					 m_size * sizeof(cuda_type), cudaMemcpyHostToDevice);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(m_name,"Could not set dataset values ", err);
		CUDA_LOG_ERR(m_name,"\n:item: ", item);
		CUDA_LOG_ERR(m_name,"\nsize: ", m_size);
		return false;
	    }
	    return true;
	}

    private:
	cuda_type * m_data;
	data_dimensions m_count;
	size_t m_size;
	bool m_host;
	info & m_info;
	std::string m_name;
    public:	
	typedef  boost::shared_ptr<dataset<cuda_type> > ptr;
    };


    template<typename ty>
	class individual_dataset : dataset<ty> 
    {
    public:
    individual_dataset(info & info, size_t islands, size_t individuals,  
		       size_t size_, const std::string & name, bool bHost):
	dataset<ty>::dataset(info,data_dimensions( islands, individuals, 1, data_item::individual_mask),size_, name, bHost )
	{

	}
	virtual bool get_data (size_t island, size_t individual, std::vector<ty> & data)
	{
	    return dataset<ty>::get_data(data_item::individual_data(island, individual), data);
	}
	virtual bool set_data (size_t island, size_t individual, const std::vector<ty> & data)
	{
	    return dataset<ty>::set_data(data_item::individual_data(island, individual), data);
	}
    };

    template<typename ty>
	class point_dataset : dataset<ty> 
    {
    public:
    point_dataset(info & info, size_t islands, size_t individuals, size_t points, 
		       size_t size_, const std::string & name, bool bHost):
	dataset<ty>::dataset(info,data_dimensions( islands, individuals, points, data_item::point_mask),size_, name, bHost )
	{}
	virtual bool get_data (size_t island, size_t individual, size_t point, std::vector<ty> & data)
	{
	    return dataset<ty>::get_data(data_item::point_data(island, individual, point), data);
	}
	virtual bool set_data (size_t island, size_t individual, size_t point, const std::vector<ty> & data)
	{
	    return dataset<ty>::set_data(data_item::point_data(island, individual, point), data);
	}
    };

    template<typename ty>
	class point_only_dataset : dataset<ty> 
    {
    public:
    point_only_dataset(info & info, size_t islands, size_t points, 
		       size_t size_, const std::string & name, bool bHost):
	dataset<ty>::dataset(info,data_dimensions( islands, 1, points, data_item::point_only_mask),size_, name, bHost )
	{}
	virtual bool get_data (size_t island, size_t point, std::vector<ty> & data)
	{
	    return dataset<ty>::get_data(data_item::point_only_data(island, point), data);
	}
	virtual bool set_data (size_t island, size_t point, const std::vector<ty> & data)
	{
	    return dataset<ty>::set_data(data_item::point_only_data(island, point), data);
	}
    };

    template<typename ty>
	class island_dataset : dataset<ty> 
    {
    public:
    island_dataset(info & info, size_t islands, size_t size_, const std::string & name, bool bHost):
	dataset<ty>::dataset(info,data_dimensions( islands, 1, 1, data_item::island_mask),size_, name, bHost )
	{}
	virtual bool get_data (size_t island, std::vector<ty> & data)
	{
	    return dataset<ty>::get_data(data_item::island_data(island), data);
	}
	virtual bool set_data (size_t island, const std::vector<ty> & data)
	{
	    return dataset<ty>::set_data(data_item::island_data(island), data);
	}
    };


}



#endif
