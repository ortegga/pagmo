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


#ifndef __PAGMO_CUDA_NNET_KERNELS__
#define __PAGMO_CUDA_NNET_KERNELS__

#include "common.h"

////////////////////////////////////////////////////////////////////////////////////////////
// activation types

template <typename cty>
struct sigmoid_functor;


template <typename cty>
struct linear_functor;


////////////////////////////////////////////////////////////////////////////////////////////
// neural networks

namespace cuda
{
    class kernel_dimensions;
}


template <typename cty, typename pre_exec, typename activ_type>
    __host__   cudaError_t cu_compute_layer(cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, 
				     size_t inputs, 
				     cuda::kernel_dimensions * dimensions_);

template <>
__host__ cudaError_t cu_compute_layer<float, nop_functor<float>, linear_functor<float> >(cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, 
							      size_t inputs, 
							      cuda::kernel_dimensions * dimensions_);
template <>
__host__ cudaError_t cu_compute_layer<float, nop_functor<float>, sigmoid_functor<float> >(cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, 
							       size_t inputs, 
							       cuda::kernel_dimensions * dimensions_);

template <>
__host__ cudaError_t cu_compute_layer<double, nop_functor<float>, linear_functor<double> >(cudaPitchedPtr *X, cudaPitchedPtr *W, cudaPitchedPtr *Y, 
								size_t inputs, 
								cuda::kernel_dimensions * dimensions_);

template <>
__host__ cudaError_t cu_compute_layer<double, nop_functor<float>, sigmoid_functor<double> >(cudaPitchedPtr *X, cudaPitchedPtr *W,  cudaPitchedPtr *Y, 
								 size_t inputs, 
								 cuda::kernel_dimensions * dimensions_);


#endif
