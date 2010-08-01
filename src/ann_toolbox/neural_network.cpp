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

#include <exception>
#include <cstdlib>
#include "../exceptions.h"
#include <iostream>

#include "neural_network.h"


using namespace ann_toolbox;

neural_network::neural_network(unsigned int input_nodes_, unsigned int output_nodes_, 
			       CudaTask * pTask):
  m_inputs(input_nodes_), m_outputs(output_nodes_),
  m_pTask(pTask), m_weights(0)
{


}

neural_network::~neural_network() {}

void neural_network::print()
{
  //print("weights", );
  //TODO: new print method
}

void neural_network::print(const char * message, std::vector<CUDA_TY> & c)
{
  std::cout<<message<<":";
  for (int i=0; i<c.size(); ++i)
    {
      std::cout<<c[i]<<" ";
    }
  std::cout<<std::endl;
}


bool neural_network::set_inputs(const std::vector<CUDA_TY> & inputs)
{
  if (inputs.size() == get_number_of_inputs())
    {
      if (m_pTask)
	{
	  if (!m_pTask->HasData(CudaTask::InputParam))
	    {
	      m_pTask->CreateData(CudaTask::InputParam, inputs.size(), false);
	    }
	  return m_pTask->SetData(m_id, CudaTask::InputParam, inputs);
	}
    }
  return false;
}

bool neural_network::set_weights(const std::vector<CUDA_TY> & weights)
{
  if (weights.size() == get_number_of_weights())
    {
      if (m_pTask)
	{
	  if (!m_pTask->HasData(CudaTask::WeightParam))
	    {
	      m_pTask->CreateData(CudaTask::WeightParam, weights.size(), false);
	    }
	  return m_pTask->SetData(m_id, CudaTask::WeightParam, weights);
	}
      //Place for non-cuda implementation
    }
  return false;
}


bool neural_network::get_outputs(std::vector<CUDA_TY> & outputs)
{
  outputs.clear();
  if (m_pTask)
    {
      if (!m_pTask->HasData(CudaTask::OutputParam))
	{
	  return false;
	}
      return m_pTask->GetData(m_id, CudaTask::OutputParam, outputs);
    }
  return false;
}


bool neural_network::prepare_outputs()
{
  std::cout<<"neural_network::prepare_outputs1"<<std::endl;
  int size = neural_network::get_number_of_output_nodes();
  std::cout<<"neural_network::prepare_outputs2"<<std::endl;
  return prepare_dataset(CudaTask::OutputParam, size);
}

bool  neural_network::prepare_dataset(int parameter, int size)
{
  std::cout<<"neural_network::prepare_dataset("<<parameter<<","<<size<<")"<<std::endl;
  if (m_pTask)
    {
      if (!m_pTask->HasData(parameter))
	{
	  return m_pTask->CreateData(parameter, size, false);
	}
    }
  //Place for non-cuda implementation
  return false;
}
