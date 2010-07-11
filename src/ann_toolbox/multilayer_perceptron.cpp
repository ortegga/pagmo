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
	unsigned int output_nodes_, const std::vector<CUDA_TY> &w) : 
    	neural_network(input_nodes_, output_nodes_),
		m_hidden(hidden_nodes_)
{
	// the number of weights is equal to all the inputs (and a bias)
	// for every hidden node, plus the connections from every hidden
	// node to every output, i.e. it is fully connected
	unsigned int wghts  = (m_inputs + 1) * m_hidden + (m_hidden + 1) * m_outputs;
	m_weights = std::vector<CUDA_TY>(wghts, 0);
	
	if(! w.empty()) set_weights(w);
}

// Destructor
multilayer_perceptron::~multilayer_perceptron() {}

// Computing the outputs
const std::vector<CUDA_TY> multilayer_perceptron::compute_outputs(std::vector<CUDA_TY> &inputs) 
{

	// check for correct input size
	if(inputs.size() != m_inputs) {
		pagmo_throw(value_error, "incorrect size of input vector");
	}

	CUDA_TY * input = new CUDA_TY [inputs.size()];
	std::copy(inputs.begin(), inputs.end(), input);

	CUDA_TY * weights = new CUDA_TY [m_weights.size()];
	std::copy(m_weights.begin(), m_weights.end(), weights);
	int hiddenSize = m_hidden * sizeof(CUDA_TY); 
	int inputSize = inputs.size() * sizeof(CUDA_TY);
	int outputSize = m_outputs * sizeof(CUDA_TY);
	int weightSize = m_weights.size() * sizeof(CUDA_TY);
	CUDA_TY * cHidden, * cInput, * cWeights, * cOutput;

	//Hidden can be garbage for all we care over here
	cudaMalloc((void **) & cHidden, hiddenSize);
	cudaMalloc((void **) & cOutput, outputSize);

	cudaMalloc((void **) & cInput, inputSize);
	cudaMemcpy(cInput, input, inputSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **) & cWeights, weightSize);
	cudaMemcpy(cWeights, weights, weightSize, cudaMemcpyHostToDevice);

	//Run the hidden layer kernel
	dim3 blocksize1(m_hidden,1,1);
	dim3 gridsize1(1,1,1);


	cuComputeLayer(cInput, cWeights, cHidden, m_inputs, gridsize1, blocksize1);

	dim3 blocksize2(m_outputs,1,1);
	dim3 gridsize2(1,1,1);
	cuComputeLayer(cHidden, & cWeights [(m_inputs + 1)*m_hidden], 
	cOutput, m_hidden, gridsize2, blocksize2);
       //Run the output layer kernel

	CUDA_TY * output = new CUDA_TY [m_outputs];
	cudaMemcpy(output, cOutput, outputSize, cudaMemcpyDeviceToHost);
	std::vector<CUDA_TY>outputs;
	outputs.insert(outputs.begin(),output, output + m_outputs);

	delete [] input;
	delete [] output;
	delete [] weights;
	cudaFree(cHidden);cudaFree(cInput);cudaFree(cWeights);cudaFree(cOutput);

    return outputs;
}


/////////////////////////////////

std::ostream &operator<<(std::ostream &s, const multilayer_perceptron &ann)
{

	std::copy(ann.m_weights.begin(), ann.m_weights.end(), std::ostream_iterator<CUDA_TY>(s, ", "));
    return s;
}

}
