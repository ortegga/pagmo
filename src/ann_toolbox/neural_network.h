#ifndef ANN_TB_NEURALNETWORK_H
#define ANN_TB_NEURALNETWORK_H

#include <vector>

namespace ann_toolbox {
	
enum { PERCEPTRON };

class neural_network {
public:
	neural_network(unsigned int input_nodes_, unsigned int output_nodes_);
	virtual ~neural_network();

//	virtual void set_inputs(std::vector<double> &inputs);
	virtual void set_weights(const std::vector<double> &chromosome); 
    virtual const std::vector<double> compute_outputs(std::vector<double> &inputs) = 0;

	//virtual void SimulationStep(unsigned n_step_number, double f_time, double f_step_interval);

	unsigned int get_number_of_input_nodes() 	{ return m_inputs; };
	unsigned int get_number_of_output_nodes()	{ return m_outputs; };
	unsigned int get_number_of_weights() 		{ return m_weights.size(); };

protected:
	const char*		m_name;
	unsigned int	m_inputs, m_outputs;
	std::vector<double>	m_weights;
};

}
#endif
