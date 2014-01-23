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

#ifndef PAGMO_ALGORITHM_GAME_THEORY_H
#define PAGMO_ALGORITHM_GAME_THEORY_H

#include "../config.h"
#include "../serialization.h"
#include "base.h"
#include "jde.h"
#include "../problem/decompose.h"



namespace pagmo { namespace algorithm {

//! Typedef for decomposition weights
typedef std::vector< double > weights_type;

//! Typedef for vector of decomposition weights
typedef std::vector< weights_type > weights_vector_type;

/// Game Theory
/**
 *
 * TODOThis class implement a multi-objective optimization algorithm based on parallel decomposition.
 * For each element of the population a different single objective problem is generated using
 * a decomposition method. Those single-objective problems are thus solved in parallel.
 * At the end of the evolution the population is set as the best individual for each single-objective problem.
 *
 * Game Theory assumes all the objectives need to be minimized.
 *
 * @author Jacco Geul (jacco.geul@gmail.com)
 **/

class __PAGMO_VISIBLE game_theory: public base
{
  public:
	game_theory(
		unsigned int gen = 10,
		unsigned int max_parallelism = 1,
		const pagmo::algorithm::base & = pagmo::algorithm::jde(100),
		weights_vector_type var_weights = weights_vector_type(),
		weights_vector_type obj_weights = weights_vector_type()
	);
	game_theory(const game_theory &);

	base_ptr clone() const;
	void evolve(population &) const;
	std::string get_name() const;
	weights_vector_type generate_var_weights(const unsigned int, const unsigned int) const;
	weights_vector_type generate_obj_weights(const unsigned int, const unsigned int) const;

  protected:
	std::string human_readable_extra() const;

  private:
	friend class boost::serialization::access;
	template <class Archive>
		void serialize(Archive &ar, const unsigned int)
	{
		ar & boost::serialization::base_object<base>(*this);
		ar & const_cast<int &>(m_gen);
		ar & const_cast<unsigned int &>(m_threads);
		ar & const_cast<base_ptr &>(m_solver);
		ar & m_var_weights;
		ar & m_obj_weights;
	}
	//Number of generations
	const int m_gen;
	const unsigned int m_threads;
	const base_ptr m_solver;
	weights_vector_type m_var_weights;
	weights_vector_type m_obj_weights;
};

}} //namespaces

BOOST_CLASS_EXPORT_KEY(pagmo::algorithm::game_theory)

#endif // PAGMO_ALGORITHM_GAME_THEORY_H
