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



#ifndef __PAGMO_CUDA_LOGGER__
#define __PAGMO_CUDA_LOGGER__


#ifndef CUDA_LOG_DISABLED

//#define CUDA_LOG_INFO(N, X, Y) (std::cout <<std::endl<< "[I] " << (N) <<" "<< (X) <<" "<< (Y) << std::endl)
#define CUDA_LOG_WARN(N, X, Y) (std::cout <<std::endl<< "[W] "<< (N) <<" "<< (X) <<" "<< (Y) << std::endl)
#define CUDA_LOG_ERR(N, X, Y)  (std::cout <<std::endl<< "[E] "<< (N) <<" "<< (X) <<" "<< (Y) << std::endl)
#define CUDA_LOG_INFO(n, x, y) ;
//#define CUDA_LOG_INFO(N, X, Y) (std::cout <<std::endl<< "[I] "<< (N) <<" "<< (X) <<" "<< (Y) << std::endl);
#define CUDA_LOG_DATASET(n, x) (log_dataset(n,x))

#else

//#define CUDA_LOG_INFO(x, y) ;
//#define CUDA_LOG_WARN(x, y) ;
//#define CUDA_LOG_ERR(x, y) ;

#endif


#include <iostream>
#include "pagmo_cuda.h"
#include "../exceptions.h"



std::ostream& operator << (std::ostream & logger, cudaError_t & err) ;    

namespace cuda
{
    
    struct cuda_error: public p_base_exception 
    {
        cuda_error(const std::string &s): 
	p_base_exception(s) 
	{
	    
	}

	cuda_error (cudaError_t & err) : 
	p_base_exception(cudaGetErrorString(err))
	{

	}
    };
    
    class Logger : public std::ostream
    {
    public:
	Logger (bool bActivated = true)
	{
	    m_bActivated = bActivated;
	}
	bool active() { return m_bActivated;}
	bool m_bActivated;
    };

    template <typename T>
	Logger & operator << (Logger & logger, const T & t)
    {
	if (logger.active())
	{
	    std::cout << t;
	}
	return logger;
    }

    template <typename tt>
    void log_dataset(const std::string & str, const tt & vec)
    {
	std::cout<<str<<": ";
	for (int i=0; i < vec.size(); ++i)
	{
	    std::cout<<vec[i]<<" ";
	}
	std::cout<<std::endl;
    }

    Logger& operator << (Logger & logger, cudaError_t & err) ;    
    Logger& operator << (Logger & logger, const dim3 & d) ;    

    extern Logger logger_info;
    extern Logger logger_warn;
    extern Logger logger_err;
}

#endif
