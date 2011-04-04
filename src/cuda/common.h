
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

#ifndef __PAGMO_CUDA_COMMON_H_
#define __PAGMO_CUDA_COMMON_H_

// common kernel and device functions/classes

template <typename fty>
struct nop_functor;

template <typename fty>
struct scale_functor;


#define ADJUSTED_TX(TPB, BX, TX) ((BX) * (TPB) + (TX))
#define RAW_TX(BDX, BX, TX) ((BX) * (BDX) + (TX))

#define JOB_ID() (threadIdx.y)

#define BLOCK_ADJUSTED_TX(TPB) ADJUSTED_TX(TPB,0,threadIdx.x)
#define GLOBAL_ADJUSTED_TX(TPB) ADJUSTED_TX(TPB,blockIdx.x,threadIdx.x)

#define BLOCK_RAW_TX() RAW_TX(blockDim.x, 0, threadIdx.x)
#define GLOBAL_RAW_TX() RAW_TX(blockDim.x, blockIdx.x, threadIdx.x)

#define BLOCK_JOBS() (blockDim.y)
#define BLOCK_TASKS(TPB) (TPB)
#define BLOCK_POINTS(TPB) (BLOCK_TASKS(TPB) )
#define BLOCK_INDIVIDUALS(TPB, PTS) (BLOCK_TASKS(TPB) / (PTS))

#define GLOBAL_INDIV_ID(TPB, PTS) (GLOBAL_ADJUSTED_TX(TPB) / (PTS))
#define BLOCK_INDIV_ID(TPB, PTS) (BLOCK_ADJUSTED_TX(TPB) / (PTS))

#define GLOBAL_POINT_ID(TPB) (GLOBAL_ADJUSTED_TX(TPB))
#define BLOCK_POINT_ID(TPB) (BLOCK_ADJUSTED_TX(TPB))

#define GLOBAL_TASK_ID(TPB) GLOBAL_ADJUSTED_TX(TPB)
#define BLOCK_TASK_ID(TPB) BLOCK_ADJUSTED_TX(TPB)

#define IS_VALID_FOR_BLOCK(TPB, PTS) (BLOCK_INDIV_ID(TPB, PTS) < BLOCK_INDIVIDUALS(TPB, PTS) && BLOCK_TASK_ID(TPB) < BLOCK_TASKS(TPB))

#endif
