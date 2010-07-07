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

// 07/06/2010: Updated to new API by Edgar Simo.
// 05/03/2009: Initial version by Francesco Biscani.

#include <boost/python.hpp>
#include <vector>

#include "../../src/keplerian_toolbox/keplerian_toolbox.h"
#include "../boost_python_container_conversions.h"
#include "../exceptions.h"

using namespace boost::python;
using namespace kep_toolbox;


/*static inline void Py_propagate_kep( std::vector<double> &r0, std::vector<double> &v0, const double &t, const double &mu )*/
static inline tuple Py_propagate_kep( tuple r0, tuple v0, const double &t, const double &mu )
{
   double r0_d[3], v0_d[3];

   if (len(r0) != 3 || len(v0) != 3) {
		P_EX_THROW(value_error,"the size of all input/output position/velocity vectors must be 3");
	}

   /* Extract values from r0. */
   r0_d[0] = extract<double>(r0[0]);
   r0_d[1] = extract<double>(r0[1]);
   r0_d[2] = extract<double>(r0[2]);
   /* Extract values from v0. */
   v0_d[0] = extract<double>(v0[0]);
   v0_d[1] = extract<double>(v0[1]);
   v0_d[2] = extract<double>(v0[2]);

   /* Propagate Lagrangian. */
   propagate_lagrangian( r0_d, v0_d, t, mu );

   /* Return values as tuple. */
   return make_tuple( make_tuple( r0_d[0], r0_d[1], r0_d[2] ),
                      make_tuple( v0_d[0], v0_d[1], v0_d[2] ) );
}

BOOST_PYTHON_MODULE(_keplerian_toolbox) {
	//to_tuple_mapping<array_d3>();
	from_python_sequence<std::vector<double>,variable_capacity_policy>();

	// Translate exceptions for this module.
	translate_exceptions();

   // Functions
   def("__propagate_kep", &Py_propagate_kep);
}

