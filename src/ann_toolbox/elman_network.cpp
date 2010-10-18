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
#include <cmath>
#include <vector>        
#include <exception>
#include "../exceptions.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "layer.h"

#include "elman_network.h"

using namespace ann_toolbox;

// Constructor
elman_network::elman_network(unsigned int input_nodes_, unsigned int hidden_nodes_,
	unsigned int output_nodes_, const std::vector<CUDA_TY> &w) : 
    	neural_network(input_nodes_, output_nodes_),
		m_hidden(hidden_nodes_)
{
	// the number of weights is equal to all the inputs (and a bias)
	// for every hidden node, plus the connections from every hidden
	// node to every output, i.e. it is fully connected. The feedback
	// nodes are connected to the hidden (middle) layer too.
	unsigned int wghts = (m_inputs + 1 + m_hidden) * m_hidden + (m_hidden + 1) * m_outputs;


	m_weights = std::vector<CUDA_TY>(wghts, 0);
	
	// the memory (feedback) values are stored in this vector
	m_memory  = std::vector<CUDA_TY>(m_hidden, 0);
	
	if(! w.empty()) set_weights(w);

}

// Destructor
elman_network::~elman_network() {}

// Computing the outputs
const std::vector<CUDA_TY> elman_network::compute_outputs(std::vector<CUDA_TY> &inputs) 
{
	// check for correct input size
	if(inputs.size() != m_inputs) {
		pagmo_throw(value_error, "incorrect size of input vector");
	}

	print("memory before",m_memory);

	CUDA_TY * input = new CUDA_TY [inputs.size() + m_memory.size()];
	std::copy(inputs.begin(), inputs.end(), input);
	//Copy the memory to the input
	std::copy(m_memory.begin(), m_memory.end(), input + inputs.size());      

	std::cout<<"augmented inputs ";
	for (int k=0; k < inputs.size() + m_memory.size(); ++k)
	  std::cout<<input[k]<<" ";
	std::cout<<std::endl;

	//Copy the weights
	CUDA_TY * weights = new CUDA_TY [m_weights.size()];
	std::copy(m_weights.begin(), m_weights.end(), weights);

	//Calculate byte sizes
	int hiddenSize = m_hidden * sizeof(CUDA_TY); 
	int inputSize = (inputs.size() + m_memory.size()) * sizeof(CUDA_TY);
	int outputSize = m_outputs * sizeof(CUDA_TY);
	int weightSize = m_weights.size() * sizeof(CUDA_TY);

	CUDA_TY * cHidden, * cInput, * cWeights, * cOutput;

	//Allocate memories on device
	cudaMalloc((void **) & cHidden, hiddenSize);
	cudaMalloc((void **) & cOutput, outputSize);
	cudaMalloc((void **) & cInput, inputSize);
	cudaMalloc((void **) & cWeights, weightSize);


	//Copy the input to the device memory
	cudaMemcpy(cInput, input, inputSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cWeights, weights, weightSize, cudaMemcpyHostToDevice);

	//Run the hidden layer kernel
	dim3 blocksize1(m_hidden,1,1);
	dim3 gridsize1(1,1,1);
	cuComputeLayerWithSegment(cInput, cWeights, cHidden, inputs.size() + m_memory.size(), 
				  inputs.size(), gridsize1, blocksize1);

     
	CUDA_TY * memory = new CUDA_TY [m_hidden];
	cudaMemcpy(memory, cHidden, hiddenSize, cudaMemcpyDeviceToHost);
	m_memory.clear();
	m_memory.insert(m_memory.begin(),memory, memory + m_hidden);

	print("memory", m_memory);

	dim3 blocksize2(m_outputs,1,1);
	dim3 gridsize2(1,1,1);
	cuComputeLayer(cHidden, cWeights + (m_inputs + m_hidden + 1)*m_hidden, 
		       cOutput, m_hidden, gridsize2, blocksize2);
		       //Run the output layer kernel

	CUDA_TY * output = new CUDA_TY [m_outputs];
	cudaMemcpy(output, cOutput, outputSize, cudaMemcpyDeviceToHost);
	std::vector<CUDA_TY>outputs;
	outputs.insert(outputs.begin(),output, output + m_outputs);

	delete [] input;
	delete [] output;
	delete [] weights;
	delete [] memory;
	cudaFree(cHidden);cudaFree(cInput);cudaFree(cWeights);cudaFree(cOutput);

    return outputs;

}

