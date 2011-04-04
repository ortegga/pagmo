
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

#include "cudatask.h"
#include "cudainfo.h"
#include "kernel_dims.h"
#include "pagmo_cuda.h"

namespace cuda
{

    bool kernel_dimensions::check_shared_mem()
    {
	const cudaDeviceProp * props = m_inf->get_prop();
	if (get_shared_mem_size() > props->sharedMemPerBlock)
	{
	    return false;
	}
	return true;
    }

    bool kernel_dimensions::check_global_mem()
    {
	const cudaDeviceProp * props = m_inf->get_prop();
	if (get_global_mem_size() >  props->totalGlobalMem)
	{
	    return false;
	}
	return true;
    }

    bool kernel_dimensions::check_task_size()
    {
	const cudaDeviceProp * props = m_inf->get_prop();
	if (get_tasks_per_block() > (size_t)props->maxThreadsPerBlock)
	{
	    return false;
	}
	return true;
    }

    bool kernel_dimensions::check_problems()
    {
	//const cudaDeviceProp * props = m_inf->get_prop();
	if (!check_shared_mem())
	{
	    CUDA_LOG_ERR("kernel_dimensions", "shared mem requirements exceed device capability", 0);
	    return false;
	}
	if (!check_task_size())
	{
	    CUDA_LOG_ERR("kernel_dimensions", "task cannot fit in a block", 0);
	    return false;
	}
	if (check_global_mem())
	{
	    CUDA_LOG_ERR("kernel_dimensions", "task cannot fit in global memory", 0);
	    return false;
	}
	return true;
    }

}
