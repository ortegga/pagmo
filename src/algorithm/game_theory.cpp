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


#include <string>
#include <vector>
#include <algorithm>

#include <boost/math/special_functions/binomial.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

#include "../exceptions.h"
#include "../population.h"
#include "../problem/base.h"
#include "../archipelago.h"
#include "../island.h"
#include "../population.h"
#include "../topology/fully_connected.h"
#include "../topology/custom.h"
#include "../topology/watts_strogatz.h"
#include "../problem/decompose.h"
#include "../util/discrepancy.h"
#include "../util/neighbourhood.h"
#include "../migration/worst_r_policy.h"
#include "../migration/best_s_policy.h"
#include "../types.h"
#include "base.h"
#include "game_theory.h"

namespace pagmo { namespace algorithm {

/// Constructor
 /**
 * Constructs a Game Theory algorithm
 *
 * TODO
 * @param[in] gen Number of generations to evolve.
 * @param[in] threads the amounts of threads that will be used
 * @param[in] solver the algorithm to solve the single objective problems.
 * @param[in] var_weights the weights for the decision variables.
 * @param[in] obj_weights the decomposition weights for the objective functions.
 * @throws value_error if gen is negative, weight_generation is not sane
 * @see pagmo::problem::decompose::method_type
 */
game_theory::game_theory(int gen,
	unsigned int threads,
	const pagmo::algorithm::base & solver,
	weights_vector_type var_weights,
	weights_vector_type obj_weights)
	:base(),
	 m_gen(gen),
	 m_threads(threads),
	 m_solver(solver.clone()),
	 m_var_weights(var_weights),
	 m_obj_weights(obj_weights)
{
	if (gen < 0) {
		pagmo_throw(value_error,"number of generations must be nonnegative");
	}

	//0 - Check whether method is implemented
	if(m_weight_generation != RANDOM && m_weight_generation != GRID && m_weight_generation != LOW_DISCREPANCY) {
		pagmo_throw(value_error,"non existing weight generation method");
	}
}

/// Copy constructor. Performs a deep copy. Necessary as a pointer to a base algorithm is here contained
game_theory::game_theory(const game_theory &algo):
	base(algo),
	m_gen(algo.m_gen),
	m_threads(algo.m_threads),
	m_solver(algo.m_solver->clone()),
	m_var_weights(algo.m_var_weights),
	m_obj_weights(algo.m_obj_weights)
{}

/// Clone method.
base_ptr game_theory::clone() const
{
	return base_ptr(new game_theory(*this));
}

/// Generates the decision variables weights
/**
 * Generates the decision variables weights for linking decision
 * variables to objective sub populations.
 *
 * @param[in] n_x number of decision variables.
 * @param[in] n_o number of objective functions.
 * @param[in] n_p number of populations.
 */
weights_vector_type game_theory::generate_var_weights(
	const unsigned int n_x, const unsigned int n_o, 
	const unsigned int n_p ) const {

	// Definition of combined: two or more decision variables
	// controlled by a single population.

	// Vector of decision variable weights.
	weights_vector_type var_weights;

	// Number of decision variables to be combined. 
	int n_c = n_o - n_p;

	// Number of decision variables to be combined for a single
	// occasion. No need to initialise, will differ per occasion.
	int n_c_o;

	// Index for vars, as eventually k will outrun i if n_x > n_p.
	int k = 0;

	// Run through the number of populations to fill obj_weights.
	for( int i = 0; i < n_p; i++ ){
		// Initialise with zero weights
		weights_type weights(n_x, 0.0);

		// Number of decision variables to combine for this i.
		// Where n_p - i represents the number of remaining
		// occasions.
		n_c_o = n_c / ( n_p - i );

		for( int j = 0; j < n_c_o; j++ ){

			// Set the weights
			weights[k] = 1.0;

			// Increment k
			k++;
		}

		// Reduce the number to be combined.
		n_c -= n_c_o;

		var_weights.push_back( weights );
	}

	return var_weights;
}

/// Generates the objective weights
/**
 * Generates the weights used in the problem decomposition.
 *
 * @param[in] n_x number of decision variables.
 * @param[in] n_o number of objective functions.
 * @param[in] n_p number of populations.
 */
weights_vector_type game_theory::generate_obj_weights(
	const unsigned int n_x, const unsigned int n_o, 
	const unsigned int n_p ) const {

	// Definition of combined: two or more objective functions
	// (through weighed decomposition) for a single population.

	// Vector of objective weights.
	weights_vector_type obj_weights;

	// Number of objectives to be combined
	int n_c = n_o - n_p;

	// Number of objectives to be combined for a single occasion.
	// No need to initialise, will differ per occasion.
	int n_c_o;

	// Index for objs, as eventually k will outrun i if n_o > n_p.
	int k = 0;

	// Do not combine the first objective function. For usage with
	// con2mo.
	bool do_not_combine_first_obj = true;

	// Run through the number of populations to fill obj_weights.
	for( int i = 0; i < n_p; i++ ){
		// Initialise with zero weights
		weights_type weights(n_o, 0.0);

		// Number of objective to combine for this i. Where
		// n_p - i represents the number of remaining
		// occasions.
		if( i == 0 && do_not_combine_first_obj ){
			n_c_o = 1;
		} else {
			n_c_o = n_c / ( n_p - i ); 
		}
		
		// Give out a warning.
		if( n_c_o > 1 ){
			std::cout << "Combining constraints: ";
		}

		// Calculate the fraction as total must eq 1.
		double frac = 1.0 / static_cast<double>( n_c_o );

		for( int j = 0; j < n_c_o; j++ ){

			// Append warning with the constraint numbers
			// that are combined in this iteration.
			if( n_c_o > 1 ){
				std::cout << (i+k) << " ";
			}

			// Set the weights
			weights[k] = frac;

			// Increment k
			k++;
		}
		if( n_c_o > 1 ){
			std::cout << std::endl;
		}

		// Reduce the number to be combined.
		n_c -= n_c_o;

		obj_weights.push_back( weights );
	}

	return obj_weights;
}

/// Evolve implementation.
/**
 * Run the Game Theory algorithm for the number of generations specified in the constructors.
 *
 * @param[in,out] pop input/output pagmo::population to be evolved.
 */
void game_theory::evolve(population &pop) const
{
	// Let's store some useful variables.
	const problem::base &prob = pop.problem();
	const population::size_type pop_size = pop.size();
	const problem::base::f_size_type prob_objectives = prob.get_f_dimension();
	const problem::base::size_type prob_dimension = prob.get_dimension();

	// Preform sanity checks
	if ( prob_objectives < 2 ) {
		pagmo_throw(value_error, "The problem is not multiobjective, try some other algorithm than Game Theory");
	}

	if ( prob_dimension < 2 ) {
		pagmo_throw(value_error, "The problem has only one decision variable. This is not supported by Game Theory, select a different algorithm.");
	}

	// Number of populations.
	const int NP = min( prob_objectives, prob_dimension );
	
	// Get out if there is nothing to do.
	if (m_gen == 0) {
		return;
	}
	
	// Generate the default if vector of decision variable weights
	// for linking to populations is empty.
        if(m_prm_weight.empty()){
		m_var_weights = generate_var_weights( prob_dimension, prob_objectives, NP );
	}

	// Generate the default if vector of objectives weights for
	// the decomposition is empty.
        if(m_obj_weight.empty()){
		m_obj_weights = generate_obj_weights( prob_dimension, prob_objectives, NP );
	}

	// More sanity checks
	if ( m_var_weights.size() != NP ) {
		pagmo_throw(value_error, "The vector of variable weights has an incorrect number of entries. Create an entry for each population.");
	}
	if ( m_obj_weights.size() != NP ) {
		pagmo_throw(value_error, "The vector of objective weights has an incorrect number of entries. Create an entry for each population.");
	}
	if ( m_var_weights[0].size() != prob_dimension ) {
		pagmo_throw(value_error, "The dimension of the variable weights do not match the problem. The dimension must be equal to the number of decision variables.");
	}
	if ( m_obj_weights[0].size() != prob_objectives ) {
		pagmo_throw(value_error, "The dimension of the objective weights do not match the problem. The dimension must be equal to the number of objectives.");
	}

	// Create all the decomposed problems 
	std::vector<pagmo::problem::decompose*> prob_vector;
	for( problem::base::f_size_type i = 0; i<NP; i++ ) {
		prob_vector.push_back(
			new pagmo::problem::decompose( prob,
				pagmo::problem::decompose::WEIGHTED, 
				m_obj_weights[i]));       
	}
	
	// TODO LOOK UP WHERE THIS SHOULD GO
	weights_type inverse_weight = abs( m_var_weights[i] - 1.0 );

	// Define best decision vector for fixed variables
	std::vector< double > best_vector;
	for( problem::base::f_size_type i = 0; i<NP; i++ ) {
		best_vector = best_vector + 
			prob_vector[i].best_x * m_var_weights[i];	
	}

	for( problem::base::f_size_type i = 0; i<NP; i++ ) {
		// Calculate modified lower and upper bounds
		std::vector< double > mod_lb = 
			inverse_single_weight * best_vector
			+ m_var_weights[i] * prob_lb;
		std::vector< double > mod_ub = 
			inverse_single_weight * best_vector
			+ m_var_weights[i] * prob_ub;

		// Change the bounds of the problems
		prob_vector(i).set_bounds( mod_lb, mod_ub );
	}

	// TODO
	// Create unconnected archipelago of NP islands. Each island solve a different decomposed part of the problem.
	pagmo::archipelago arch(pagmo::archipelago::broadcast);

	// Sets random number generators of the archipelago using the
	// algorithm urng to obtain a deterministic behaviour upon
	// copy.
	arch.set_seeds(m_urng());

	// Assign population to each problem
	for( problem::base::f_size_type i = 0; i<NP; i++ ) { 
		
		// Create an empty population for each decomposed
		// problem
		pagmo::population decomp_pop(*prob_vector[i], 0, m_urng()); 

		// TODO Decide fill option
		// A: Copy the original population over to each
		for ( population::size_type j = 0; j<pop.size(); j++ ) {
			decomp_pop.set_x( j, pop.get_individual(j).cur_x );
		}

		// B: Fill the population with the original population
		for ( population::size_type j = 0; j<pop.size(); j++ ) {
			decomp_pop.push_back( pop.get_individual(j).cur_x );
		}
		
		// Add the island to the archipelago
		arch.push_back(pagmo::island(*m_solver,decomposed_pop, 1.0, selection_policy, replacement_policy));
	}

	// Set topology
	topology::unconnected topo;
	arch.set_topology(topo);

	// Evolve entire archipelago once
	arch.evolve_batch(1, m_threads);

	// Evolve N time for algorithm
	for(int g = 0; g < a_gen; ++g) {
		arch.evolve_batch(1, m_threads);
	}

	pop.clear();
	
	for ( population::size_type i = 0; i<pop.size(); i++ ) {
		pop.push_back( arch.get_island(0)->get_population()->get_individual(i).cur_x );
	}
	
	// For multiple fixed best strategy, select top N
	// // Finally, we assemble the evolved population selecting from the original one + the evolved one
	// // the best NP (crowding distance)
	// population popnew(pop);
	// for(pagmo::population::size_type i=0; i<arch.get_size() ;++i) {
	// 	popnew.push_back(arch.get_island(i)->get_population().champion().x);
	// 	m_fevals += arch.get_island(i)->get_algorithm()->get_fevals();
	// }
	// std::vector<population::size_type> selected_idx = popnew.get_best_idx(NP);
	
	// // We completely clear the population (NOTE: memory of all individuals and the notion of
	// // champion is thus destroyed)
	// pop.clear();
	// // And we recreate it with the best NP among the evolved and the new population
	// for (population::size_type i=0; i < NP; ++i) {
	// 	pop.push_back(popnew.get_individual(selected_idx[i]).cur_x);
	// }
}

/// Algorithm name
std::string game_theory::get_name() const
{
	return m_solver->get_name() + "[Game Theory]";
}

/// Extra human readable algorithm info.
/**
 * Will return a formatted string displaying the parameters of the algorithm.
 */
std::string game_theory::human_readable_extra() const
{
	std::ostringstream s;
	s << "gen:" << m_gen << ' ';
	s << "threads:" << m_threads << ' ';
	s << "solver:" << m_solver->get_name() << ' ';
	s << "neighbours:" << m_T << ' ';
	s << "decomposition:";
	switch (m_method)
	{
		case pagmo::problem::decompose::BI : s << "BI" << ' ';
			break;
		case pagmo::problem::decompose::WEIGHTED : s << "WEIGHTED" << ' ';
			break;
		case pagmo::problem::decompose::TCHEBYCHEFF : s << "TCHEBYCHEFF" << ' ';
			break;
	}
	s << "weights:";
	switch (m_weight_generation)
	{
		case RANDOM : s << "RANDOM" << ' ';
			break;
		case LOW_DISCREPANCY : s << "LOW_DISCREPANCY" << ' ';
			break;
		case GRID : s << "GRID" << ' ';
			break;
	}
	s << "ref. point" << m_z;
	return s.str();
}

}} //namespaces

BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::algorithm::game_theory)
