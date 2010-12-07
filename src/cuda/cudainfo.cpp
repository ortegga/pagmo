


#include "cudainfo.h"
#include "boost/foreach.hpp"


namespace cuda
{
    info::info()
    {
	m_activedev = -1;
	if(!load())
	{
	    CUDA_LOG_ERR("info", "cuda failed to load", false);
	    //pagmo throw;
	}
    }

    bool info::load()
    {
	int devnum = 0;
	if (cudaGetDeviceCount(&devnum) != cudaSuccess || devnum == 0)
	{
	    return false;
	}
	for (int i = 0; i<devnum; ++i)
	{
	    deviceinfo * pInfo = new deviceinfo(i);
	    m_devices.push_back(pInfo);
	}

	CUDA_LOG_INFO("info", "info::load loaded ", devnum);
	set_device(0);
	return true;
    }

    bool info::set_device(unsigned int index)
    {
	if (index > m_devices.size())
	{
	    return false;
	}
	if (cudaSetDevice(index) == cudaSuccess)
	{
	    m_activedev = index;
	    return true;
	}
	return  false;
    }

    deviceinfo * info::get_device()
    {
	return get_at(m_activedev);
    }

    deviceinfo* info::get_at(unsigned int index) 
    { 
	if (index < get_count()) 
	{
	    CUDA_LOG_INFO("info", "info::get_at ", index);
	    return m_devices[index];
	}
	return NULL;
    }

    unsigned int info::get_maxthreadcount()
    {
	if (get_count())
	    return get_at(m_activedev)->get_maxthreadcount();
	return 0;
    }

    unsigned int info::get_warpsize()
    {
	if (get_count())
	    return get_at(m_activedev)->get_warpsize();
	return 0;
    }

    info::~info()
    {
	BOOST_FOREACH(deviceinfo * i, m_devices)
	{
	    delete i;
	}
    }


    deviceinfo::deviceinfo(unsigned int devId) : m_id(devId)
    {
	if(!load(m_id))
	{
	    //throw pagmo exception
	}
    }

    bool deviceinfo::load(unsigned int devId)
    {
	if(cudaGetDeviceProperties(&m_prop,devId) != cudaSuccess)
	{
	    return true;
	}
	return false;
    }

    unsigned int deviceinfo::get_maxthreadcount()
    {
	return m_prop.multiProcessorCount * 8 * m_prop.warpSize;
    }

    unsigned int deviceinfo::get_warpsize()
    {
	return  m_prop.warpSize;
    }

    std::ostream & operator <<(std::ostream & os, const info & dev)
    {
	BOOST_FOREACH(deviceinfo * info, dev.m_devices)
	{
	    os << *info;
	}
	return os;
    }

    std::ostream & operator <<(std::ostream & os, const deviceinfo & dev)
    {
	os <<"Device ("<<dev.m_id<<") - "<<dev.m_prop.name<<std::endl;

	os<<"clock: "<<(dev.m_prop.clockRate/1000)<<"MHz x "<<dev.m_prop.multiProcessorCount<<" multiprocessors"<<std::endl;
	os<<"memory: "
	  <<(dev.m_prop.totalGlobalMem>>20)<<"MB(global) "
	  <<(dev.m_prop.sharedMemPerBlock>>10)<<"KB(shared/block) "
	  <<(dev.m_prop.totalConstMem>>10)<<"KB(const)"
	  <<std::endl;
	os<<"compute capable: "<<dev.m_prop.major<<"."<<dev.m_prop.minor;
	os<<" device overlap: "<<dev.m_prop.deviceOverlap
	  <<" timeout enabled: "<<dev.m_prop.kernelExecTimeoutEnabled
	  <<" integrated mem: "<<dev.m_prop.integrated
	  <<" map host mem: "<<dev.m_prop.canMapHostMemory
	  <<std::endl;
    
	os<<"registers per block: "<<dev.m_prop.regsPerBlock<<std::endl;
	os<<"warp size: "<<dev.m_prop.warpSize<<std::endl;
	os<<"memory pitch (MB): "<<(dev.m_prop.memPitch >> 20)<<std::endl;
	os<<"threads per block: " <<dev.m_prop.maxThreadsPerBlock<<std::endl;

	os<<"thread dimensions: " 
	  << dev.m_prop.maxThreadsDim[0]<<"x"
	  << dev.m_prop.maxThreadsDim[1]<<"x"
	  << dev.m_prop.maxThreadsDim[2];
	os<<" grid size: " 
	  << dev.m_prop.maxGridSize[0]<<"x"
	  << dev.m_prop.maxGridSize[1]<<"x"
	  << dev.m_prop.maxGridSize[2] << std::endl;

	return os;
    }
}
