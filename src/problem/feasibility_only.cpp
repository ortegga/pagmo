/*****************************************************************************
 *   Copyright (C) 2004-2013 The PaGMO development team,                     *
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

#include <cmath>
#include <algorithm>

#include "../exceptions.h"
#include "../types.h"
#include "feasibility_only.h"

namespace pagmo { namespace problem {

/**
 * Constructor using initial constrained problem
 *
 * @param[in] problem base::problem to be transformed in feasibility only problem (substitution of the objective function with a constant objective function)
 *
 */
feasibility_only::feasibility_only(const base &problem):
	base_meta(problem,
		 problem.get_dimension(),
		 problem.get_i_dimension(),
		 problem.get_f_dimension(),
		 problem.get_c_dimension(),
		 problem.get_ic_dimension(),
		 problem.get_c_tol())
{
}

/// Clone method.
base_ptr feasibility_only::clone() const
{
	return base_ptr(new feasibility_only(*this));
}

/// Implementation of the objective functions.
/// (Wraps over the original implementation)
void feasibility_only::objfun_impl(fitness_vector &f, const decision_vector &x) const
{
		std::fill(f.begin(),f.end(), 0.);
}

/// Extra human readable info for the problem.
/**
 * Will return a formatted string containing the type of constraint handling
 */
std::string feasibility_only::human_readable_extra() const
{
	std::ostringstream oss;
	oss << m_original_problem->human_readable_extra() << std::endl;
	oss << "\n\tfeasibility_only";
	oss << std::endl;
	return oss.str();
}

std::string feasibility_only::get_name() const
{
	return m_original_problem->get_name() + " [feasibility_only]";
}

}}

BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::problem::feasibility_only)

