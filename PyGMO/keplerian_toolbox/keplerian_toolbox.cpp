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

static inline tuple Py_propagate_kep(const std::vector<double> &r0, const std::vector<double> &v0, const double &t, const double &mu )
{
	if (r0.size() != 3 || v0.size() != 3) {
		P_EX_THROW(value_error,"size of input position/velocity vectors must be 3");
	}
	std::vector<double> r0_out(r0), v0_out(v0);
	propagate_lagrangian(r0_out,v0_out,t,mu);
	return make_tuple(r0_out,v0_out);
}

BOOST_PYTHON_MODULE(_keplerian_toolbox) {
	//to_tuple_mapping<array_d3>();
	from_python_sequence<std::vector<double>,variable_capacity_policy>();

	// Translate exceptions for this module.
	translate_exceptions();

   // Functions
   def("__propagate_kep", &Py_propagate_kep);
}

