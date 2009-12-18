#ifndef ANN_TB_MULTILAYER_PERCEPTRON_H
#define ANN_TB_MULTILAYER_PERCEPTRON_H

#include "neural_network.h"

namespace ann_toolbox {
	
/**
 * A multilayer perceptron (a type of artificial neural network), representing a 
 * feedforward neural network, with one hidden layer. 
 * More info: http://en.wikipedia.org/wiki/Mulitlayer_perceptron
 */	
class multilayer_perceptron : public neural_network {
public:
	/// Constructor
	/**
	 * Creates a new multilayer_perceptron object, which is derived from the 
	 * neural_network class and using one hidden layer of nodes. It calls the
	 * set_weights function to initalize the weights of the neural network.
	 * \param input_nodes	the number of input nodes
	 * \param hidden_nodes	the number of nodes in the hidden layer
	 * \param output_nodes	the number of output nodes (default = 1)
	 * \param w				the weights, with which the neural network is initiated (empty by default)
	 * \return a perceptron object
	 */
	multilayer_perceptron(unsigned int input_nodes_, unsigned int hidden_nodes_, 
			unsigned int output_nodes_ = 1, const std::vector<double> &w = std::vector<double>());	

	/// Destructor
    ~multilayer_perceptron();

	/// Compute Outputs
	const std::vector<double> compute_outputs(std::vector<double> &inputs);

protected:
	// number of hidden nodes
	unsigned int	m_hidden;
};

}
#endif
