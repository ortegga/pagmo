#ifndef __PAGMO_CUDAINFO_H__
#define __PAGMO_CUDAINFO_H__

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


class CudaDeviceInfo;
typedef std::vector<CudaDeviceInfo*> CudaDeviceInfoVector;



class CudaInfo
{
 public:
  CudaInfo();
  ~CudaInfo();
  bool Load();
  int GetCount() {return m_devices.size();}
  CudaDeviceInfo* GetAt(int index) 
  { 
    if (index && index < GetCount()) 
      return m_devices[index];
  }
  int GetMaxThreadCount();
  int GetWarpSize();
  bool SetCudaDevice(int index);
  CudaDeviceInfo * GetAciveDevice();
  int GetAciveDeviceIndex() { return m_activeDev;}
  friend std::ostream &operator<<(std::ostream &, const CudaInfo &);
 private:
  CudaDeviceInfoVector m_devices;
  int m_activeDev; //Enforces one device only for now?
};

class CudaDeviceInfo 
{
 public:
  CudaDeviceInfo(int deviceId);
  bool Load(int devId);
  int GetMaxThreadCount();
  int GetWarpSize();
  friend std::ostream &operator<<(std::ostream &, const CudaDeviceInfo &);	
  //Aux functions to check for various capabilities for the device
 private:
  struct cudaDeviceProp m_prop;
  int m_id;
};


#endif
