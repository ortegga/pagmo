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

#include <cstdlib>
#include <iterator>
#include <cmath>
#include <vector>        
#include <algorithm>
#include <exception>
#include "../exceptions.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "layer.h"

#include "multilayer_perceptron.h"

namespace ann_toolbox {

  // Constructor
  multilayer_perceptron::multilayer_perceptron(unsigned int input_nodes_, unsigned int hidden_nodes_,
					       CudaTask * pTask, unsigned int output_nodes_) : 
    neural_network(input_nodes_, output_nodes_, pTask),
    m_hidden(hidden_nodes_)
  {
    // the number of weights is equal to all the inputs (and a bias)
    // for every hidden node, plus the connections from every hidden
    // node to every output, i.e. it is fully connected
	
    m_weights = (m_inputs + 1) * m_hidden + (m_hidden + 1) * m_outputs;

  }

  // Destructor
  multilayer_perceptron::~multilayer_perceptron() {}

  // Computing the outputs
  const std::vector<CUDA_TY> multilayer_perceptron::compute_outputs(std::vector<CUDA_TY> &inputs) 
  {
    //   cuComputeLayer(cHidden, & cWeights [(m_inputs + 1)*m_hidden], 
    //		   cOutput, m_hidden, gridsize2, blocksize2);
    std::vector<CUDA_TY> outputs;
    return outputs;
  }


 bool multilayer_perceptron::prepare_outputs()
 {
   std::cout<<"multilayer_perceptron::prepare_outputs"<<std::endl;
   return neural_network::prepare_outputs() &&  
     prepare_dataset(CudaTask::HiddenParam, m_hidden);
 }

  /////////////////////////////////

  std::ostream &operator<<(std::ostream &s, const multilayer_perceptron &ann)
  {
    //std::copy(ann.m_weights.begin(), ann.m_weights.end(), std::ostream_iterator<CUDA_TY>(s, ", "));
    return s;
  }

}
