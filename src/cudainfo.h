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
  friend std::ostream &operator<<(std::ostream &, const CudaInfo &);	
 private:
  CudaDeviceInfoVector m_devices;
};

class CudaDeviceInfo 
{
 public:
  CudaDeviceInfo(int deviceId);
  bool Load(int devId);
  friend std::ostream &operator<<(std::ostream &, const CudaDeviceInfo &);	
 private:
  struct cudaDeviceProp m_prop;
  int m_id;
};


#endif
