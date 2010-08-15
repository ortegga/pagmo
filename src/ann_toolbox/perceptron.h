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

#ifndef ANN_TB_PERCEPTRON_H
#define ANN_TB_PERCEPTRON_H

#include "neural_network.h"
#include "../cuda/cudatask.h"

namespace ann_toolbox {
	
  /**
   * A simple perceptron (a type of artificial neural network), representing the 
   * simplest kind of feedforward neural network. This basically refers to a linear
   * classifier. 
   * More info: http://en.wikipedia.org/wiki/Perceptron
   */	
  template <typename ty, int activ_type>
    class perceptron : public neural_network <ty> {
  public:
    typedef typename cuda::perceptron_task <ty,activ_type> task;


  perceptron(unsigned int input_nodes_, unsigned int output_nodes_ = 1, 
	     cuda::task<ty> * pTask = NULL) : 
    neural_network(input_nodes_, output_nodes_, pTask)
      {
	this->m_weights = (this->m_inputs + 1) * this->m_outputs;
      }

    ~perceptron() {}

  };

}
#endif
