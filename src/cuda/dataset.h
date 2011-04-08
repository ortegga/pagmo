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


#ifndef __PAGMO_CUDA_DATASET__
#define  __PAGMO_CUDA_DATASET__

#include "cudainfo.h"
#include "pagmo_cuda.h"
#include "logger.h"
#include "boost/shared_ptr.hpp"
#include "stdio.h"


//////////////////////////////////////////////////////////////////////////////////
/*
  dataset.h
  includes a set of classes that manage the device memory for CUDA. Terminology:

  island: The islands of the Island based Global Optimisation model
  individual: The GA individual 
  point: Subtasks for each individual. USeful if we're working with a set of tasks
  for each individual

*/
//////////////////////////////////////////////////////////////////////////////////


namespace cuda
{

//////////////////////////////////////////////////////////////////////////////////
/// data_item struct
/*
  data_item struct, used to define indexes for  dataset instances. Depending on 
  the kind of dataset, it will work with islands, individuals and/or points. 

*/
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
/// data_dimensions struct
/*
  defines the dimensions of the dataset instance. Used in conjunction with data_items, it
  also enables us to translate data_item instances to raw indexes. 
 */


    struct data_dimensions : public data_item
    {

    data_dimensions(size_t is, size_t in, size_t pt, type t) : data_item(is,in,pt,t){}

	//Compute the total number of instances of data in the dataset.
	size_t get_count()
	{
	    return (m_type & point_type ? point : 1) * (m_type & island_type ? island : 1) * (m_type & individual_type ?  individual : 1) ;
	}

	//Validate the data_item for if its even possible 
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

	//This method evaluates the datum id for the data_item. 
	//This will vary according to the type of dataset
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
/// dataset class
/*
  template class that encapsulates segment of device memory. On creation it creates a 2D
  array of memory (it uses cudaMemcpy3D to do so). Consecutive datum along the 
  x-axis will belong to consecutive tasks. While the data for each task is laid out along 
  the y-axis. For instance for tasks a,b,c,d,e... each with data sized 0-5:


  a0 b0 c0 d0 e0 ....
  a1 b1 c1 d1 e1 ....
  a2 b2 c2 d2 e2 ....
  a3 b3 c3 d3 e3 ....
  a4 b4 c4 d4 e4 ....
  a5 b5 c5 d5 e5 ....

  This ensures that the data is laid out in an order that the GPU will work optimally with.

*/
    template <class cuda_type>
	class dataset
    {
    public:
	//ctor. 
	// allocates the device memory as well. 
    dataset(info & info, const data_dimensions & count,  size_t size_, const std::string & name):
	m_count(count), m_size(size_),m_info(info), m_name(name)
	{
	    cudaError_t err;
	    
	    CUDA_LOG_INFO(m_name, "Data size:", m_count);
	    size_t tasks = m_count.get_count();
	    struct cudaExtent extent;
	    extent.width = tasks * sizeof(cuda_type);
	    extent.height = get_task_size();
	    extent.depth = 1;
	    CUDA_LOG_INFO(m_name,"Allocating device dataset:", size_);
	    err = cudaMalloc3D(&m_data, extent);
	    CUDA_LOG_INFO(m_name,"Allocating device dataset: ", m_data.ptr);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(m_name,"Could not allocate dataset:", err);
	    }
	  
	}


	//Copy device data from location <item> to vector. 
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

	//Copy host data from vector to location at <item>
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
	    cudaError_t err = cudaFree(m_data.ptr);
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
	struct cudaPitchedPtr* get_data() 
	{
	    return &m_data;
	}

	size_t get_serial(const data_item & item)
	{
	    return m_count.serialize(item);
	}

    protected:

	//Copy device data to contageous memory sub_data
	bool get_values(const data_item & item, cuda_type * sub_data)
	{

	    //Initialize copy parameters with 0. 
	    struct cudaMemcpy3DParms copy_params = {0};

	    CUDA_LOG_INFO(m_name,"get dataset values for task:", item);
	    CUDA_LOG_INFO(m_name,"get dataset values for task:", m_count.serialize(item));

	    //Using pitched ptrs for copying as thats what dataset works withConstruct pitched ptr
	    //for destination/host sub_data
	    copy_params.srcPtr = m_data;
	    copy_params.dstPtr = make_cudaPitchedPtr((void *)sub_data, sizeof(cuda_type), 1 , m_size);

	    copy_params.dstPos.x = 0;
	    copy_params.dstPos.y = 0;
	    copy_params.dstPos.z = 0;

	    //X needs to be multiplied with the size of the base type because serialize gives us the datum id. 
	    copy_params.srcPos.x = sizeof(cuda_type)*m_count.serialize(item);
	    copy_params.srcPos.y = 0;
	    copy_params.srcPos.z = 0;


	    //We're copying a column of data from device to host. The width will be the size of the base type and the 
	    //height will be the amount of data to copy. Depth is fixed to 1. 
	    copy_params.extent.width  = sizeof(cuda_type);
	    copy_params.extent.height =  m_size;
	    copy_params.extent.depth = 1;

	    copy_params.kind = cudaMemcpyDeviceToHost;

	    cudaError_t err = cudaMemcpy3D(&copy_params);
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(m_name,"Could not get dataset values", err);
		return false;
	    }
	    return true; 
	}

	bool set_values(const data_item & item, cuda_type * sub_data)
	{
	    CUDA_LOG_INFO(m_name,"set dataset values for task:", item);
	    CUDA_LOG_INFO(m_name,"set dataset values for task:", m_count.serialize(item));

	    //Initialize copy parameters with 0. 
	    struct cudaMemcpy3DParms copy_params = {0};

	    //Using pitched ptrs for copying as thats what dataset works with. Construct pitched ptr
	    //for source/host data
	    copy_params.srcPtr = make_cudaPitchedPtr((void *)sub_data, sizeof(cuda_type), 1, m_size);
	    copy_params.dstPtr = m_data;


	    CUDA_LOG_INFO(m_name,"set dataset values for task:", m_data.ysize);
	    CUDA_LOG_INFO(m_name,"size:", m_size);

	    copy_params.srcPos.x = 0;
	    copy_params.srcPos.y = 0;
	    copy_params.srcPos.z = 0;

	    //X needs to be multiplied with the size of the base type because serialize gives us the datum id. 
	    copy_params.dstPos.x = sizeof(cuda_type)*m_count.serialize(item);
	    copy_params.dstPos.y = 0;
	    copy_params.dstPos.z = 0;

	    //We're copying a column of data from device to host. The width will be the size of the base type and the 
	    //height will be the amount of data to copy. Depth is fixed to 1. 
	    copy_params.extent.width = sizeof(cuda_type);
	    copy_params.extent.height =  m_size;
	    copy_params.extent.depth = 1;

	    copy_params.kind = cudaMemcpyHostToDevice;

	    cudaError_t err = cudaMemcpy3D(&copy_params);

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
	//Device memory pointer (ptr) with attributes (pitch, xsize, ysize)
	struct cudaPitchedPtr m_data;
	//dimensions of the data. Copied from creating cuda::task
	data_dimensions m_count;
	//size of the data.
	size_t m_size;
	//cuda device information
	info & m_info;
	//label for logging
	std::string m_name;
    public:	
	typedef  boost::shared_ptr<dataset<cuda_type> > ptr;
    };


///individual_dataset class
/*
  Useful to create a dataset instance which will work with individuals
 */

    template<typename ty>
	class individual_dataset : dataset<ty> 
    {
    public:
    individual_dataset(info & info, size_t islands, size_t individuals,  
		       size_t size_, const std::string & name):
	dataset<ty>::dataset(info,data_dimensions( islands, individuals, 1, data_item::individual_mask),size_, name )
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

///individual_dataset class
/*
  Useful to create a dataset instance which will work with individuals and their points
 */

    template<typename ty>
	class point_dataset : dataset<ty> 
    {
    public:
    point_dataset(info & info, size_t islands, size_t individuals, size_t points, 
		       size_t size_, const std::string & name):
	dataset<ty>::dataset(info,data_dimensions( islands, individuals, points, data_item::point_mask),size_, name)
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

///points_only_dataset class
/*
  Useful to create a dataset instance which will work with points only 
  (when they're a common resource for all individuals )
 */


    template<typename ty>
	class point_only_dataset : dataset<ty> 
    {
    public:
    point_only_dataset(info & info, size_t islands, size_t points, 
		       size_t size_, const std::string & name):
	dataset<ty>::dataset(info,data_dimensions( islands, 1, points, data_item::point_only_mask),size_, name )
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


///points_only_dataset class
/*
  Useful to create a dataset instance which will work with islands only 
 */

    template<typename ty>
	class island_dataset : dataset<ty> 
    {
    public:
    island_dataset(info & info, size_t islands, size_t size_, const std::string & name):
	dataset<ty>::dataset(info,data_dimensions( islands, 1, 1, data_item::island_mask),size_, name)
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
