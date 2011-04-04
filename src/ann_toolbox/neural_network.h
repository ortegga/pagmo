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



#ifndef ANN_TB_NEURALNETWORK_H
#define ANN_TB_NEURALNETWORK_H

#include <vector>
#include <iostream>
#include <string>
#include "../cuda/cudatask.h"
#include "../cuda/common.h"

using namespace cuda;

namespace ann_toolbox {

    template <typename ty, size_t in_, size_t out_, typename pre_exec = nop_functor<ty> , typename post_exec = nop_functor<ty> >
	class neural_network : public task<ty>
    {
    public:
    neural_network(info & in, const std::string & name, size_t islands, size_t individuals, size_t task_count): 
    task<ty> ( in, name, islands, individuals, task_count, out_) , 
	    m_weights(0)
	    {
	    }
	virtual ~neural_network(){}

	enum 
	{
	    param_inputs = 0,
	    param_weights = 1,
	    param_hiddens = 2,
	    param_outputs = 3,
	    param_output_weights = 4,
	};


	unsigned int get_number_of_input_nodes() const{ return get_number_of_inputs(); }
	unsigned int get_number_of_inputs() const	{ return in_; }	
	unsigned int get_number_of_output_nodes() const	{ return get_number_of_outputs(); }
	unsigned int get_number_of_outputs() const	{ return out_; }
	unsigned int get_number_of_weights() const	{ return m_weights; }

	virtual bool set_inputs(const data_item & item, const std::vector<ty> & inputs)
	{
	    if (inputs.size() == get_number_of_inputs())
	    {
		return task<ty>::set_inputs (item, param_inputs, inputs, get_number_of_inputs(), this->m_name + ":inputs");
	    }
	    return false;
	}
	virtual bool set_weights(const data_item & item, const std::vector<ty> &weights)
	{
	    if (weights.size() == get_number_of_weights())
	    {
		return task<ty>::set_inputs (item, param_weights, weights, get_number_of_weights(), this->m_name + ":weights");
	    }
	    return false;
	}

	virtual bool get_weights(const data_item & item, std::vector<ty> &weights)
        {
	    return task<ty>::get_outputs (item, param_weights, weights);
	}

	virtual bool get_outputs( const data_item & item, std::vector<ty> & outputs)
	{
	    return task<ty>::get_outputs (item, param_outputs, outputs);
	}
     
	virtual bool prepare_outputs()
	{ 
	    int size = get_number_of_output_nodes();
	    return task<ty>::prepare_dataset(data_item::point_mask,param_outputs, size, this->m_name + ":outputs");
	}
     
	virtual bool launch() = 0;
     
    protected:
     
	size_t  m_weights;
     
    };
  
}
#endif
