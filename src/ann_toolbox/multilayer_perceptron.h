#ifndef ANN_TB_MULTILAYER_PERCEPTRON_H
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

#define ANN_TB_MULTILAYER_PERCEPTRON_H

#include "../cuda/cudatask.h"

#include "neural_network.h"
#include "../cuda/nnet.h"
#include "../cuda/cudatimer.h"
#include "../cuda/kernel_dims.h"
#include <vector>



//Contains an aux task which handles the input to hidden evaluation.
namespace ann_toolbox {
	
    template <typename ty , size_t in_, size_t hid_, size_t out_, typename pre_exec1 = nop_functor<ty>, typename pre_exec2 = nop_functor<ty>, 
	typename activ_type1 = sigmoid_functor<ty>, typename activ_type2 = sigmoid_functor<ty> >

	class multilayer_perceptron : public neural_network<ty, in_, out_> 
    {
    public:

    multilayer_perceptron(cuda::info & in, const std::string & name, size_t individuals, size_t task_count) : 
    neural_network<ty, in_, out_>::neural_network(in, name, individuals, task_count), 
	    m_hidden_task(individuals, task_count, hid_)
	    {

		this->m_hidden_weights = (in_ + 1) * hid_ ;
		this->m_weights  = this->m_hidden_weights + (hid_ + 1) * out_;

		this->m_hidden_task.set_shared_chunk(0, this->m_weights , in_);
		this->m_hidden_task.set_global_chunk(0, this->m_weights , in_ + hid_);

		this->set_shared_chunk(0, this->m_hidden_weights , hid_);
		this->set_global_chunk(0, this->m_hidden_weights , hid_ + out_);

		this->m_dims1 = kernel_dimensions::ptr(new block_complete_dimensions (&this->m_info, &(this->m_hidden_task), this->m_name));
		this->m_dims2 = kernel_dimensions::ptr(new block_complete_dimensions  (&this->m_info, this->get_profile(), this->m_name));

	    }
	typedef neural_network<ty, in_, out_> base;

	virtual bool get_hidden( const data_item & item, std::vector<ty> & outputs)
	{
	    return task<ty>::get_outputs (item, base::param_hiddens, outputs);
	}
    
	virtual bool set_weights(const data_item & item, const std::vector<ty> &chromosome)
	{
	    if (chromosome.size() == this->get_number_of_weights())
	    {
		std::vector<ty> first_segment, second_segment;
		//Optimize damnit!!!
		for (size_t i=0; i < m_hidden_weights; ++i)
		{
		    first_segment.push_back(chromosome[i]);
		}
		for (size_t i=m_hidden_weights; i < chromosome.size(); ++i)
		{
		    second_segment.push_back(chromosome[i]);
		}

		return task<ty>::set_inputs (item, this->param_weights, first_segment, first_segment.size()) && 
		    task<ty>::set_inputs (item, this->param_output_weights, second_segment, second_segment.size());
	    }
	    return false;
	}
    
	/// Destructor
	~multilayer_perceptron() {}
	
	virtual bool prepare_outputs()
	{
	    return base::prepare_outputs() &&  this->prepare_dataset(base::param_output_weights, m_hidden_weights) && 
		this->prepare_dataset(base::param_hiddens, hid_);
	}

	bool launch() 
	{

	    typename dataset<ty>::ptr pOutData = this->get_dataset(base::param_outputs);
	    typename dataset<ty>::ptr pInput = this->get_dataset(base::param_inputs);
	    typename dataset<ty>::ptr pHidden = this->get_dataset(base::param_hiddens);
	    typename dataset<ty>::ptr pWeights = this->get_dataset(base::param_weights);
	    typename dataset<ty>::ptr pOutputWeights = this->get_dataset(base::param_output_weights);

	    if (!(pInput && pWeights && pHidden && pOutData && pOutputWeights))
	    {
		CUDA_LOG_ERR(this->m_name, " Could not find a dataset ", 0);
		CUDA_LOG_ERR(this->m_name, " inputs " , pInput);
		CUDA_LOG_ERR(this->m_name, " weights ",  pWeights);
		CUDA_LOG_ERR(this->m_name, " second layer inputs ",  pHidden);
		CUDA_LOG_ERR(this->m_name, " second layer weights ",  pOutputWeights);
		CUDA_LOG_ERR(this->m_name, " outputs ",  pOutData);
		return false;
	    }

	    cudaError_t err;
	    err = cu_compute_layer<ty, pre_exec1, activ_type1>(*pInput->get_data(), *pWeights->get_data(), *pHidden->get_data(),  
							       pInput->get_task_size(), this->m_dims1.get());
	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(this->m_name, "Launch fail ", err);
		return false;
	    }
	    block_complete_dimensions dims2 (&this->m_info, this->get_profile(), this->m_name);

	    err = cu_compute_layer<ty, pre_exec2, activ_type2>(*pHidden->get_data(), *pOutputWeights->get_data(), *pOutData->get_data(),  
							       pHidden->get_task_size(), this->m_dims2.get());

	    if (err != cudaSuccess)
	    {
		CUDA_LOG_ERR(this->m_name, " Second Launch fail ", err);
		return false;
	    }

	    return true;
	}
    protected:
	size_t m_hidden_weights;
	task_profile m_hidden_task;
        kernel_dimensions::ptr m_dims1;
        kernel_dimensions::ptr m_dims2;
    };


}
#endif
