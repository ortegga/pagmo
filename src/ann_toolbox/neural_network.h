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
#include "../cuda/cudaty.h"
#include "../cuda/cudatask.h"

namespace ann_toolbox {

  class neural_network {
  public:
    neural_network(unsigned int input_nodes_, unsigned int output_nodes_, 
		   CudaTask * pTask);
    virtual ~neural_network();

    virtual const std::vector<CUDA_TY> compute_outputs(std::vector<CUDA_TY> &inputs) = 0;

    unsigned int get_number_of_input_nodes() const{ return get_number_of_inputs(); }
    unsigned int get_number_of_inputs() const	{ return m_inputs; }	
    unsigned int get_number_of_output_nodes() const	{ return get_number_of_outputs(); }
    unsigned int get_number_of_outputs() const	{ return m_outputs; }
    unsigned int get_number_of_weights() const	{ return m_weights; }
    unsigned int get_id() const { return m_id;} 

    virtual bool set_inputs(const std::vector<CUDA_TY> & inputs);
    virtual bool set_weights(const std::vector<CUDA_TY> &chromosome); 
    virtual bool get_outputs( std::vector<CUDA_TY> & outputs);
    virtual bool prepare_outputs();


    virtual void print();
    virtual void print(const char * message, std::vector<CUDA_TY> & c);

    void set_task(int id) { m_id = id;}
	
  protected:

    virtual bool prepare_dataset(int parameter, int size);

    const char*	m_name;
    unsigned int	m_inputs, m_outputs;
    unsigned int  m_id;
    unsigned int	m_weights;
    CudaTask * m_pTask;

  };

}
#endif
