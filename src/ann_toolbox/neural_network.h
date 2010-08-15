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

#ifndef ANN_TB_NEURALNETWORK_H
#define ANN_TB_NEURALNETWORK_H

#include <vector>
#include <iostream>
#include <string>
#include "../cuda/cudatask.h"
#include "../cuda/tasklet.h"

using namespace cuda;

namespace ann_toolbox {

  template <typename ty>
    class neural_network : public tasklet<ty>
  {
  public:
  neural_network(unsigned int input_nodes_, unsigned int output_nodes_, 
		 task<ty> * pTask): 
    tasklet<ty> ( pTask ) , 
      m_inputs(input_nodes_), m_outputs(output_nodes_),
      m_weights(0)
      {

      }
    virtual ~neural_network(){}

    unsigned int get_number_of_input_nodes() const{ return get_number_of_inputs(); }
    unsigned int get_number_of_inputs() const	{ return m_inputs; }	
    unsigned int get_number_of_output_nodes() const	{ return get_number_of_outputs(); }
    unsigned int get_number_of_outputs() const	{ return m_outputs; }
    unsigned int get_number_of_weights() const	{ return m_weights; }

    virtual bool set_inputs(const std::vector<ty> & inputs)
    {
        if (inputs.size() == get_number_of_inputs())
	  {
	    return tasklet<ty>::set_inputs (cuda::task<ty>::inputs, inputs);
	  }
	return false;
    }
    virtual bool set_weights(const std::vector<ty> &chromosome)
    {
        if (chromosome.size() == get_number_of_weights())
	  {
	    return tasklet<ty>::set_inputs (cuda::task<ty>::weights, chromosome);
	  }
	return false;
    }

    virtual bool get_outputs( std::vector<ty> & outputs)
    {
      return tasklet<ty>::get_outputs (cuda::task<ty>::outputs, outputs);
    }

    virtual bool prepare_outputs()
      {
	int size = get_number_of_output_nodes();
	return tasklet<ty>::prepare_dataset(cuda::task<ty>::outputs, size);
      }
	
  protected:

    const char*	  m_name;
    unsigned int  m_inputs;
    unsigned int  m_outputs;
    unsigned int  m_weights;

  };

}
#endif
