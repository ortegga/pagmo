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
	
  template <typename ty , size_t in_, size_t hid_, size_t out_, typename activ_type>
    class multilayer_perceptron : public neural_network<ty, in_, out_> 
    {
    public:

    multilayer_perceptron(cuda::info & in, size_t individuals, size_t task_count) : 
      neural_network<ty, in_, out_>::neural_network(in, individuals, task_count), 
	m_hidden_task(individuals, task_count, hid_)
	{

	  this->m_hidden_weights = (in_ + 1) * hid_ ;
	  this->m_weights  = this->m_hidden_weights + (hid_ + 1) * out_;

	  this->m_hidden_task.set_shared_chunk(0, this->m_weights , in_);
	  this->m_hidden_task.set_global_chunk(0, this->m_weights , in_ + hid_);

	  this->set_shared_chunk(0, this->m_hidden_weights , hid_);
	  this->set_global_chunk(0, this->m_hidden_weights , hid_ + out_);

	}
      typedef neural_network<ty, in_, out_> base;

      virtual bool get_hidden( size_t individual, size_t taskid, std::vector<ty> & outputs)
      {
	return task<ty>::get_individual_outputs (individual, taskid, base::param_hiddens, outputs);
      }
    
      virtual bool set_weights(int individual, const std::vector<ty> &chromosome)
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

	    return task<ty>::set_inputs (individual, this->param_weights, first_segment) && 
	      task<ty>::set_inputs (individual, this->param_output_weights, second_segment);
	  }
	return false;
      }
    
      /// Destructor
      ~multilayer_perceptron() {}
	
      virtual bool prepare_outputs()
      {
	return base::prepare_outputs() &&  
	  this->prepare_individual_dataset(base::param_hiddens, hid_);
      }

      bool launch() 
      {


	dataset<ty> * pOutData = this->get_dataset(base::param_outputs);
	dataset<ty> * pInput = this->get_dataset(base::param_inputs);
	dataset<ty> * pHidden = this->get_dataset(base::param_hiddens);
	dataset<ty> * pWeights = this->get_dataset(base::param_weights);
	dataset<ty> * pOutputWeights = this->get_dataset(base::param_output_weights);

	if (!(pInput && pWeights && pHidden && pOutData && pOutputWeights))
	  {
	    std::cout <<"failure"<<pInput <<" "<< pWeights<<" "<<pHidden<<" "<<pOutputWeights<<" "<<pOutData<<std::endl;
	    return false;
	  }

	block_complete_dimensions dims1 (this->m_info, &(this->m_hidden_task));

	cu_compute_layer<ty, activ_type>(*pInput->get_data(), *pWeights->get_data(), *pHidden->get_data(),  
					 pInput->get_task_size(), &dims1);

      
	block_complete_dimensions dims2 (this->m_info, this->get_profile());

	cu_compute_layer<ty, activ_type>(*pHidden->get_data(), *pOutputWeights->get_data(), *pOutData->get_data(),  
					 pHidden->get_task_size(), & dims2);
	return true;
      }
    protected:
      size_t m_hidden_weights;
      task_profile m_hidden_task;
    };


}
#endif
