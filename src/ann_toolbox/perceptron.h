#ifndef ANN_TB_PERCEPTRON_H
#define ANN_TB_PERCEPTRON_H

#include "neural_network.h"

namespace ann_toolbox {
	
/**
 * A simple perceptron (a type of artificial neural network), representing the 
 * simplest kind of feedforward neural network. This basically refers to a linear
 * classifier. 
 * More info: http://en.wikipedia.org/wiki/Perceptron
 */	
class perceptron : public neural_network {
public:
	/// Constructors
	/**
	 * Creates a new perceptron object, which is derived from the neural_network
	 * class. If initial weights are handed over it calls the set_weights function
	 * to initalize the weights of the neural network.
	 * \param input_nodes	the number of input nodes
	 * \param output_nodes	the number of output nodes (not mandatory, default = 1)
	 * \param w				the weights, with which the neural network is initiated (not mandatory)
	 * \return a perceptron object
	 */
	perceptron(unsigned int input_nodes_, unsigned int output_nodes_ = 1, const std::vector<double> &w = std::vector<double>());

	/// Destructor
    ~perceptron();

	/// Compute Outputs
	const std::vector<double> compute_outputs(std::vector<double> &inputs);

protected:
};

}
#endif
