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

namespace ann_toolbox {
	
  /**
   * A multilayer perceptron (a type of artificial neural network), representing a 
   * feedforward neural network, with one hidden layer. 
   * More info: http://en.wikipedia.org/wiki/Mulitlayer_perceptron
   */	
  template <typename ty , int activ_type>
  class multilayer_perceptron : public neural_network<ty> {
  public:

    typedef typename cuda::multilayer_perceptron_task<ty, activ_type> task;

  multilayer_perceptron(unsigned int input_nodes_, unsigned int hidden_nodes_, 
			cuda::task<ty> * pTask, unsigned int output_nodes_ = 1) : 
    neural_network<ty>::neural_network(input_nodes_, output_nodes_, pTask),
      m_hidden(hidden_nodes_)
      {
	// the number of weights is equal to all the inputs (and a bias)
	// for every hidden node, plus the connections from every hidden
	// node to every output, i.e. it is fully connected
	
	this->m_weights = (this->m_inputs + 1) * this->m_hidden + (this->m_hidden + 1) * this->m_outputs;

      }

    /// Destructor
    ~multilayer_perceptron() {}
	
    /// Stream output operator.
    /*friend std::ostream &operator<<(std::ostream &, const multilayer_perceptron<ty> &)
      {}*/

    virtual bool prepare_outputs()
    {
	 return neural_network<ty>::prepare_outputs() &&  
	   this->prepare_dataset(cuda::task<ty>::hiddens, m_hidden);
    }
	
  protected:
    unsigned int m_hidden;
  };


}
#endif
