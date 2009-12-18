#include <cstdlib>
#include <cmath>
#include <vector>        
#include <exception>
#include "../exceptions.h"

#include "ctrnn.h"

using namespace ann_toolbox;

// create these funtions to easier access the weights
#define input_to_hidden_weights(idx)	m_weights[0 + (idx)]
#define hidden_to_hidden_weights(idx)	m_weights[m_inputs * m_hidden + (idx)]
#define hidden_bias(idx)				m_weights[m_inputs * m_hidden + m_hidden * m_hidden	+ (idx)]
#define hidden_taus(idx)				m_weights[m_inputs * m_hidden + m_hidden * m_hidden	+ m_hidden + (idx)]
#define hidden_to_output_weights(idx)	m_weights[m_inputs * m_hidden + m_hidden * m_hidden	+ m_hidden + m_hidden + (idx)]
#define output_bias(idx)				m_weights[m_inputs * m_hidden + m_hidden * m_hidden	+ m_hidden + m_hidden + m_hidden * m_outputs + (idx)]

// Constructor
ctrnn::ctrnn(unsigned int input_nodes_, unsigned int hidden_nodes_, unsigned int output_nodes_,
	const std::vector<double> &w) : 
    	neural_network(input_nodes_, output_nodes_),
		m_hidden(hidden_nodes_),
		m_time_step(0.1),
		m_hidden_neurons(std::vector<double>(hidden_nodes_, 0)),
		m_output_neurons(std::vector<double>(output_nodes_, 0))
{
	// the number of weights
	unsigned int wghts = m_inputs * m_hidden// synaptic weights from input to hidden layer
					+ m_hidden * m_hidden	// synaptic weights from hidden to hidden layer
					+ m_hidden + m_hidden	// bias and taus of the hidden layer
					+ m_hidden * m_outputs	// synaptic weights from hidden to output layer
					+ m_outputs;			// bias of the output layer

	m_weights = std::vector<double>(wghts, 0);
	
	if(! w.empty()) set_weights(w);
}

// Destructor
ctrnn::~ctrnn() {}


void ctrnn::set_weights(const std::vector<double> &w) {
	if(w.size() != m_weights.size()) {
		pagmo_throw(value_error, "number of weights is incorrect");
	}
	m_weights = w;
		
	// then scale them according to the data
	// TODO
}

// Computing the outputs
const std::vector<double> ctrnn::compute_outputs(std::vector<double> &inputs) 
{
	// check for correct input size
	if(inputs.size() != m_inputs) {
		pagmo_throw(value_error, "incorrect size of input vector");
	}
	
	unsigned int i, j;
	// Update delta state of hidden layer from inputs:
	std::vector<double> hidden(m_hidden, 0);
	for( i = 0; i < m_hidden; i++) {
    	hidden[i] = - m_hidden_neurons[i];
  
    	for ( j = 0; j < m_inputs; j++) {
        	// weight * sigmoid(state)
			hidden[i] += input_to_hidden_weights(i * m_inputs + j) * inputs[j] ;	  
		}	  
	}

	double h;
	// Update delta state from hidden layer, self-recurrent connections:
	for ( i = 0; i < m_hidden; i++) {
    	for ( j = 0; j < m_hidden; j++) {
        	h = (double(1.0)/( exp(-( m_hidden_neurons[j] + hidden_bias(j))) + 1.0 ));
        	hidden[i] += hidden_to_hidden_weights(i * m_hidden + j) * h;
    	}
	}

	for( i = 0; i < m_hidden; i++) {
    	m_hidden_neurons[i] += hidden[i] * m_time_step/hidden_taus(i);
	}

	// Update the outputs layer::
	for ( i = 0; i < m_outputs; i++) {
    	for ( j = 0; j < m_hidden; j++) {
        	double z = (double(1.0)/( exp(-( m_hidden_neurons[j] + hidden_bias(j))) + 1.0 ));
        	m_output_neurons[i] += hidden_to_output_weights(i * m_hidden + j) * z;
    	}

    	// Compute the activation function immediately, since this is
    	// what we return and since the output layer is not recurrent:
    	m_output_neurons[i] = double(1.0)/( exp(-( m_output_neurons[i] + output_bias(i))) + 1.0 );
	}	
	
    return m_output_neurons;
}