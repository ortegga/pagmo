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


#ifndef __PAGMO_CUDAINFO_H__
#define __PAGMO_CUDAINFO_H__

#include <vector>
#include "pagmo_cuda.h"
#include <iostream>
#include "boost/shared_ptr.hpp"
#include "stdlib.h"
#include "math.h"
#include "logger.h"


namespace cuda
{

    //deviceinfo class
/*
  This class describes each device found on the system. 
 */
    class deviceinfo 
    {
    public:
	deviceinfo(unsigned int deviceid);
	bool load(unsigned int devid);
	unsigned int get_maxthreadcount();
	unsigned int get_warpsize();
	unsigned int get_block_shared_mem();
	friend std::ostream &operator<<(std::ostream &, const deviceinfo &);
	cudaDeviceProp * get_prop() 
	{ 
	    CUDA_LOG_INFO("deviceinfo", "deviceinfo::get_prop", this);
	    return &m_prop;
	}
	//Aux functions to check for various capabilities for the device
    private:
	struct cudaDeviceProp m_prop;
	unsigned int m_id;
    };

    //info class
/*
  This stores descriptors for all the devices on the host. With the first device
  selected by default
 */


    class info
    {
    public:
	info();
	~info();
	bool load();
	unsigned int get_count() {return m_devices.size();}
	deviceinfo* get_at(unsigned int index);
	unsigned int get_maxthreadcount();
	unsigned int get_warpsize();
	unsigned int get_block_shared_mem();
	bool set_device(unsigned int index);
	deviceinfo * get_device();
	cudaDeviceProp *  get_prop() 
	{ 
	    return get_device()->get_prop();
	} 
	int get_deviceindex() 
	{ 
	    return m_activedev;
	}
	friend std::ostream &operator<<(std::ostream &, const info &);
    private:
	info (info & inf);
	typedef  std::vector<deviceinfo *> deviceinfo_vector;
	deviceinfo_vector m_devices;
	unsigned int m_activedev; //Enforces one device only for now?
    };

}


#endif
