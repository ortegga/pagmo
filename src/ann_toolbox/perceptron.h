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

// Created by Juxi Leitner on 2009-12-21.
// based on the TwoDee Artificial Neural Network Code

#ifndef ANN_TB_PERCEPTRON_H
#define ANN_TB_PERCEPTRON_H

#include "neural_network.h"
#include "../cuda/cudatask.h"
#include "../cuda/kernel_dims.h"
#include "../cuda/nnet.h"

namespace ann_toolbox {
	
    /**
     * A simple perceptron (a type of artificial neural network), representing the 
     * simplest kind of feedforward neural network. This basically refers to a linear
     * classifier. 
     * More info: http://en.wikipedia.org/wiki/Perceptron
     */	
    template <typename ty, size_t in_, size_t out_, typename pre_exec, typename activ_type>
	class perceptron : public neural_network <ty, in_, out_> {
    public:

	typedef neural_network<ty, in_, out_> base;
    
    perceptron(cuda::info & in, const std::string & name, size_t individuals, size_t task_count) : 
	neural_network<ty, in_, out_>::neural_network(in, name, individuals, task_count)
	{
	    this->m_weights = (in_ + 1) * out_;
	    this->set_shared_chunk(0, this->m_weights, in_);
	    this->set_global_chunk(0, this->m_weights, in_ + out_);
	}


	~perceptron() 
	{
	}

	bool launch() 
	{

	    dataset<ty> * pOutData = this->get_dataset(base::param_outputs);
	    dataset<ty> * pInput = this->get_dataset(base::param_inputs);
	    dataset<ty> * pWeights = this->get_dataset(base::param_weights);

	    if (!(pInput && pWeights && pOutData))
	    {
		CUDA_LOG_ERR(this->m_name, " Could not find a dataset ", 0);
		CUDA_LOG_ERR(this->m_name, " inputs " , pInput);
		CUDA_LOG_ERR(this->m_name, " weights ",  pWeights);
		CUDA_LOG_ERR(this->m_name, " outputs ",  pOutData);

		return false;
	    }
      
	    block_complete_dimensions dims (&this->m_info, this->get_profile(), this->m_name);

	    cudaError_t err;
	    err = cu_compute_layer<ty, pre_exec, activ_type >(*pInput->get_data(), *pWeights->get_data(), 
							      *pOutData->get_data(),  pInput->get_task_size(), &dims);

	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(this->m_name, " Launch fail ", err);
		return false;
	    }
	    
	    return true;
	}

    };

}
#endif
