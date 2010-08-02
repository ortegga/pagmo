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
 *   the Free Software Foundation; either version 3 of the License, or       *
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

// 02/07/2010: Initial version by Edgar Simo.

#include <boost/python.hpp>
#include <vector>

#include "../../src/AstroToolbox/Pl_Eph_An.h"
#include "../boost_python_container_conversions.h"
#include "../exceptions.h"

using namespace boost::python;


static inline tuple Py_Planet_Ephemerides_Analytical( const double &mjd2000, const int &planet )
{
   std::vector<double> position(3), velocity(3);
   Planet_Ephemerides_Analytical( mjd2000, planet, &position[0], &velocity[0] );
   return make_tuple( position, velocity );
}

BOOST_PYTHON_MODULE(_astro_toolbox) {
	// Translate exceptions for this module.
	translate_exceptions();

	// Functions
   def("Planet_Ephemerides_Analytical", &Py_Planet_Ephemerides_Analytical);
}

