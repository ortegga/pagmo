#include <exception>
#include "../exceptions.h"

#include "neural_network.h"

using namespace ann_toolbox;

neural_network::neural_network(unsigned int input_nodes_, unsigned int output_nodes_) :
	m_inputs(input_nodes_), m_outputs(output_nodes_)
{}

neural_network::~neural_network() {}

void neural_network::set_weights(const std::vector<double> &chromosome) {
	if(chromosome.size() != m_weights.size()) {
		pagmo_throw(value_error, "number of weights is incorrect");
	}
	m_weights = chromosome;
}
