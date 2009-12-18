#ifndef ANN_TB_ELMAN_NETWORK_H
#define ANN_TB_ELMAN_NETWORK_H

#include "neural_network.h"

namespace ann_toolbox {
	
/**
 * An Elman network (named after Jeff Elman) is a recurrent neural network (RNN), a class of
 * artificial neural networks (ANN), where fixed connections backwards are added. This creates
 * an internal state of the network which allows it to exhibit dynamic temporal behavior. In
 * an Elman network connections from the middle (hidden) layer to so called "context units", 
 * which are added to the input nodes.
 * More info: http://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_network
 */	
class elman_network : public neural_network {
public:
	/// Constructor
	/**
	 * Creates a new elman_network object, which is derived from the neural_network class
	 * and using one hidden layer of nodes and "context units" from that hidden layer as
	 * extra inputs. It calls the set_weights function to initalize the weights of the neural
	 * network.	
	 * \param input_nodes	the number of input nodes
	 * \param hidden_nodes	the number of nodes in the hidden layer
	 * \param output_nodes	the number of output nodes (default = 1)
	 * \param w				the weights, with which the neural network is initiated (empty by default)
	 * \return a perceptron object
	 */
	elman_network(unsigned int input_nodes_, unsigned int hidden_nodes_, 
			unsigned int output_nodes_ = 1, const std::vector<double> &w = std::vector<double>());	

	/// Destructor
    ~elman_network();

	/// Compute Outputs
	const std::vector<double> compute_outputs(std::vector<double> &inputs);

protected:
	// number of hidden nodes
	unsigned int	m_hidden;
	// a vector to store the memory of the network (feedback nodes)
	std::vector<double>	m_memory;	
};

}
#endif


