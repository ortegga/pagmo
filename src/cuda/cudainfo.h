#ifndef __PAGMO_CUDAINFO_H__
#define __PAGMO_CUDAINFO_H__

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


namespace cuda
{

  class deviceinfo 
  {
  public:
    deviceinfo(unsigned int deviceid);
    bool load(unsigned int devid);
    unsigned int get_maxthreadcount();
    unsigned int get_warpsize();
    friend std::ostream &operator<<(std::ostream &, const deviceinfo &);	
    //Aux functions to check for various capabilities for the device
  private:
    struct cudaDeviceProp m_prop;
    unsigned int m_id;
  };

  class info
  {
  public:
    info();
    ~info();
    bool load();
    unsigned int get_count() {return m_devices.size();}
    deviceinfo* get_at(unsigned int index) 
    { 
      if (index && index < get_count()) 
	return m_devices[index];
      return NULL;
    }
    unsigned int get_maxthreadcount();
    unsigned int get_warpsize();
    bool set_device(unsigned int index);
    deviceinfo * get_device();
    int get_deviceindex() { return m_activedev;}
    friend std::ostream &operator<<(std::ostream &, const info &);
  private:
    typedef  std::vector<deviceinfo *> deviceinfo_vector;
    deviceinfo_vector m_devices;
    unsigned int m_activedev; //Enforces one device only for now?
  };
}


#endif
