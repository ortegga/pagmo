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


#ifndef __PAGMO_CUDA_EULER_INT__
#define __PAGMO_CUDA_EULER_INT__

#include "integrator.h"
#include "../cuda/kernel.h"
#include "../cuda/kernel_dims.h"
#include "../cuda/logger.h"

namespace pagmo
{
    namespace odeint
    {

	template <typename ty, typename system, size_t in_, size_t out_, size_t system_params, 
	    typename kernel_dims1 = block_complete_dimensions, typename pre_exec, typename post_exec >
	    class euler_integrator : public integrator<ty, system, in_, out, system_params>
	{
	public:

	typedef typename integrator<ty, system, in_, out_, system_params> base;
    
	euler_integrator (cuda::info & inf, const std::string & name, size_t islands, size_t individuals, size_t task_count_) : base(inf, name, islands, individuals, task_count_)
	    {
		//this->set_shared_chunk(0, 0 , 24 + 6 + 2);
		//this->set_global_chunk(0, 0 , 6 + 2)
		this->m_dims = kernel_dimensions::ptr( new kernel_dims1 (&this->m_info, this->get_profile(), this->m_name));	    
	    }

	    bool launch()
	    {
		using namespace cuda;
		//InParams
		dataset<ty>::ptr pX = this->get_dataset(this->param_x);
		dataset<ty>::ptr pO = this->get_dataset(this->param_o);

		if (!pX || !pO)
		{
		    CUDA_LOG_ERR(this->m_name, " Could not find a dataset ", 0);
		    CUDA_LOG_ERR(this->m_name, " inputs " , pX);
		    CUDA_LOG_ERR(this->m_name, " outputs ",  pO);
		    return false;
		}
	  
		//TODO handle shared memory and different individuals
		cudaError_t err =  euler_integrate<ty, system, pre_exec, post_exec>(pX->get_data(), pO->get_data(), m_param_t, m_param_dt,pX->get_tasksize(), 
										    m_param_scale_limits, task_data_size, this->m_dims.get());
		
		if (err != cudaSuccess)
		{
		    CUDA_LOG_ERR(this->m_name, " Launch fail ", err);
		    return false;
		}
		return true;

	    }
	protected:
	    kernel_dimensions::ptr m_dims;

	};

    }
}



#endif
