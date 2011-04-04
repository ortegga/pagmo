
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


#include "logger.h"

namespace cuda
{
 
    Logger logger_info(false);
    Logger logger_warn(true);
    Logger logger_err;

    Logger& operator << (Logger & logger, cudaError_t & err)
    {
	if (logger.active())
	{
	    logger << cudaGetErrorString(err);
	}
	return logger;
    }

    Logger& operator << (Logger& logger, const dim3 & d)
    {
	logger << "< " << d.x <<", "<< d.y<<", "<< d.z << " >";
	return logger;
    }
  
}


std::ostream& operator << (std::ostream & logger, cudaError_t & err)
{
    logger << cudaGetErrorString(err);
    return logger;
}


    std::ostream& operator << (std::ostream & logger, const dim3 & d)
    {
	logger << "< " << d.x <<", "<< d.y<<", "<< d.z << " >";
	return logger;
    }
