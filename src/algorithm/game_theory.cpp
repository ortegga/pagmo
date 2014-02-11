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
#include "../topology/unconnected.h"
#include "../problem/decompose.h"
#include "../util/discrepancy.h"
#include "../util/neighbourhood.h"
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
 * @param[in] weight_generation type of weight generation.
 * @param[in] relative_tolerance for determining convergence.
 * @param[in] absolute_tolerance for determining convergence..
 * @throws value_error if gen is negative, weight_generation is not sane
 * @see pagmo::problem::decompose::method_type
 */
game_theory::game_theory(int gen,
	unsigned int threads,
	const pagmo::algorithm::base & solver,
	const weights_vector_type &var_weights,
	const weights_vector_type &obj_weights,
	weight_generation_type weight_generation,
	const std::vector< double > &relative_tolerance,
	const std::vector< double > &absolute_tolerance)
	:base(),
	 m_gen(gen),
	 m_threads(threads),
	 m_solver(solver.clone()),
	 m_var_weights(var_weights),
	 m_obj_weights(obj_weights),
	 m_weight_generation(weight_generation),
	 m_relative_tolerance(relative_tolerance),
	 m_absolute_tolerance(absolute_tolerance)
{
	if (gen < 0) {
		pagmo_throw(value_error,"number of generations must be nonnegative");
	}
	if(m_weight_generation != UNIFORM && m_weight_generation != RANDOM  ) {
		pagmo_throw(value_error,"non existing weight generation method.");
	}
}

/// Copy constructor. Performs a deep copy. Necessary as a pointer to a base algorithm is here contained
game_theory::game_theory(const game_theory &algo):
	base(algo),
	m_gen(algo.m_gen),
	m_threads(algo.m_threads),
	m_solver(algo.m_solver->clone()),
	m_var_weights(algo.m_var_weights),
	m_obj_weights(algo.m_obj_weights),
	m_weight_generation(algo.m_weight_generation),
	m_relative_tolerance(algo.m_relative_tolerance),
	m_absolute_tolerance(algo.m_absolute_tolerance)
{}

/// Clone method.
base_ptr game_theory::clone() const
{
	return base_ptr(new game_theory(*this));
}

// Sum of two vectors
template <typename T>
std::vector<T> game_theory::sum_of_vec(const std::vector<T>& a, const std::vector<T>& b) const
{
	assert(a.size() == b.size());

	std::vector<T> result;
	result.reserve(a.size());

	std::transform(a.begin(), a.end(), b.begin(), 
		std::back_inserter(result), std::plus<T>());
	return result;
}

// Hadamard product of two vectors
template <typename T>
std::vector<T> game_theory::had_of_vec(const std::vector<T>& a, const std::vector<T>& b) const
{
	assert(a.size() == b.size());

	std::vector<T> result;
	result.reserve(a.size());

	std::transform(a.begin(), a.end(), b.begin(), 
		std::back_inserter(result), std::multiplies<T>());
	return result;
}

// Inverse of a vector
template <typename T>
std::vector<T> game_theory::inv_of_vec(const std::vector<T>& a) const
{
	std::vector<T> b;
	for( int i = 0; i < a.size(); ++i )
		b.push_back(fabs( a[i] - 1 ));
	return b;
}

// Absolute difference between two vectors
template <typename T>
bool game_theory::solution_within_tolerance(const std::vector<T>& a, const std::vector<T>& b) const
{
	// Check if sizes match
	assert(m_relative_tolerance.size() == m_absolute_tolerance.size());
	assert(a.size() == b.size());

	// Create temporary vectors
	std::vector< double > rel_tol;
	std::vector< double > abs_tol;

	// Check if single tolerance value is used or one for each
	// variable.
	if( m_relative_tolerance.size() == 1){
		rel_tol = std::vector< double >( a.size(), m_relative_tolerance[0] );
		abs_tol = std::vector< double >( a.size(), m_absolute_tolerance[0] );
	} else {
		assert(a.size() == m_absolute_tolerance.size());
		rel_tol = m_relative_tolerance;
		abs_tol = m_absolute_tolerance;
	}
		
	// Check if each elements meets tolerance
	bool withintol = true;
	for( unsigned int i = 0; i < a.size(); ++i){
		withintol = withintol * ( fabs(a[i] - b[i]) < 
			std::max( abs_tol[i], 
				std::max(fabs(a[i]),fabs(b[i])) * rel_tol[i] ));
	}
	return withintol;
}

/// Generates the weights randomly
/**
 * Generates the weights used in the problem decomposition.
 *
 * @param[in] n_x length of weight vector.
 * @param[in] n_v length of vector of weight vectors.
 * @param[in] fracs if true the sum of a weight vector is always one.
 */
weights_vector_type game_theory::generate_random_weights(const unsigned int n_x, 
	const unsigned int n_v, const bool fracs ) const
{
	// Generate vectors with entries 0,1,...,(d-1).
	std::vector< double > b( n_x, 0 );
	weights_vector_type a;
	std::vector< int > m;
	
	while( true ){
		// Fill a with dummy b and reset counter
		a.clear();
		m.clear();
		for( unsigned int i = 0; i < n_v; ++i ){
		        a.push_back( b );
			m.push_back( 0 );
		}

		// Change some zeros to ones
		for( unsigned int i = 0; i < n_x; ++i ){
			// Generate random no in 0,...,(d-1)
			int r = static_cast< int >( m_drng() * n_v);

			// Assign value
			a[r][i] = 1.0;

			// Increase the counter
			m[r]++;
		}

		// See if all are represented
		bool all_unique = true;
		for( unsigned int i = 0; i < n_v; ++i ){
			all_unique *= ( m[i] > 0 );
		}
		if( all_unique )
			break;
	}

	if( fracs ){
		for( unsigned int i = 0; i < n_v; ++i ){
			for( unsigned int j = 0; j < n_x; ++j ){
				a[i][j] /= static_cast< double >( m[i] );
			}
		}
	}
	return a;
}

/// Generates the weights uniformly
/**
 * Generates the weights used in the problem decomposition.
 *
 * @param[in] n_x length of weight vector.
 * @param[in] n_v length of vector of weight vectors.
 * @param[in] fracs if true the sum of a weight vector is always one.
 */
weights_vector_type game_theory::generate_uniform_weights(
	const unsigned int n_x, const unsigned int n_v, const bool fracs ) const {

	// Definition of combined: two or more of x for one single
	// population.

	// Vector of objective weights.
	weights_vector_type vec_weights;

	// Number to be combined
	unsigned int n_c;

	// Number to be combined for a single occasion. No need to
	// initialise, will differ per occasion.
	unsigned int n_c_o;

	// Index for objs, as eventually k will outrun i if n_x > n_v.
	unsigned int k = 0;

	// Assigned weights, useful for random generation.
	std::vector< int > ass_weights(n_x, 0);

	// Run through the number of populations to fill vec_weights.
	for( unsigned int i = 0; i < n_v; i++ ){
		// Initialise with zero weights
		weights_type weights(n_x, 0.0);

		n_c = n_x - (n_v - i) + 1 - k;

		// Number of objective to combine for this i. Where
		// n_v - i represents the number of remaining
		// occasions.
		if( n_c == 1 ){
			n_c_o = 1;
		} else {
			n_c_o = static_cast<int>( ceil( static_cast<double>( n_c ) / 
					static_cast<double>( n_v - i )));
		}

		// Calculate the fraction, if fracs then the sum must be 1.
		double frac = 1.0;

		if( fracs ){
			frac = 1.0 / static_cast<double>( n_c_o );
		}
		for( unsigned int j = 0; j < n_c_o; j++ ){

			// Set the weights
			weights[k] = frac;

			// Increment k
			k++;
		}
		vec_weights.push_back( weights );
	}

	return vec_weights;
}

/// Generates the weights
/**
 * Generates the weights used in the problem decomposition.
 *
 * @param[in] n_x length of weight vector.
 * @param[in] n_v length of vector of weight vectors.
 * @param[in] fracs if true the sum of a weight vector is always one.
 * @param[in] random if true the weights are generated at random.
 */
weights_vector_type game_theory::generate_weights( const unsigned int n_x,
	const unsigned int n_v, const bool fracs, bool random ) const {

	// Preform sanity checks
	if ( n_v > n_x ) {
		pagmo_throw(value_error, "The number of weight vector cannot be longer than the number of entries in a weight vector.");
	}
	if( random ){
		return generate_random_weights( n_x, n_v, fracs );
	} else {
		return generate_uniform_weights( n_x, n_v, fracs );
	}
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
	const unsigned int subpops = std::min( prob_objectives, prob_dimension );
	
	// Get out if there is nothing to do.
	if (m_gen == 0) {
		return;
	}

	// Generate the default if vector of decision variable weights
	// for linking to populations is empty.
	weights_vector_type var_weights;
	weights_vector_type obj_weights;

	switch (m_weight_generation)
	{
		case UNIFORM : 
		        if(m_var_weights.empty()){
				var_weights = generate_weights( 
					prob_dimension, subpops, false, false );
			}else{
				var_weights = m_var_weights;
			}
			if(m_obj_weights.empty()){
				obj_weights = generate_weights( 
					prob_objectives, subpops, true, false );
			}else{
				obj_weights = m_obj_weights;
			}
			break;
		case RANDOM : 
			var_weights = generate_weights( 
				prob_dimension, subpops, false, true );
			obj_weights = generate_weights( 
				prob_objectives, subpops, true, true );
			break;
	}

	// More sanity checks
	if ( var_weights.size() != subpops ) {
		pagmo_throw(value_error, "The vector of variable weights has an incorrect number of entries. Create an entry for each population.");
	}
	if ( obj_weights.size() != subpops ) {
		pagmo_throw(value_error, "The vector of objective weights has an incorrect number of entries. Create an entry for each population.");
	}
	if ( var_weights[0].size() != prob_dimension ) {
		pagmo_throw(value_error, "The dimension of the variable weights do not match the problem. The dimension must be equal to the number of decision variables.");
	}
	if ( obj_weights[0].size() != prob_objectives ) {
		pagmo_throw(value_error, "The dimension of the objective weights do not match the problem. The dimension must be equal to the number of objectives.");
	}

	// Create all the decomposed problems 
	std::vector<pagmo::problem::decompose*> prob_vector;
	for( problem::base::f_size_type i = 0; i<subpops; i++ ) {
		prob_vector.push_back(
			new pagmo::problem::decompose( prob,
				pagmo::problem::decompose::WEIGHTED, 
				obj_weights[i]));       
	}

	// Set unconnected topology, only using arch for parallel processing
	topology::unconnected topo;
	
	// Create unconnected archipelago of subpops islands. Each island solve a different decomposed part of the problem.
	pagmo::archipelago arch(topo);

	// Sets random number generators of the archipelago using the
	// algorithm urng to obtain a deterministic behaviour upon
	// copy.
	arch.set_seeds(m_urng());

	// Assign population to each problem
	for( problem::base::f_size_type i = 0; i<subpops; i++ ) { 
		
		// Create an empty population for each decomposed
		// problem
		population decomp_pop(*prob_vector[i], 0, m_urng()); 

		// Fill the population from the original and calculate
		// the objective.
		for ( population::size_type j = 0; j<pop.size(); j++ ) {
			decomp_pop.push_back( pop.get_individual(j).cur_x );
		}

		// Put pop in island and add to arch.
		arch.push_back(pagmo::island(*m_solver, decomp_pop, 1.0));
	}

	// Create a last best vector for convergence check
	std::vector< double > last_best_vector( prob_dimension, 0.0 );

	for(int g = 0; g < m_gen; ++g) {

		// Define best decision vector for fixed variables
		std::vector< double > best_vector( prob_dimension, 0.0 );
		for( problem::base::f_size_type i = 0; i<subpops; i++ ) {
			int best_idx = arch.get_island(i)->get_population().get_best_idx();
			best_vector = sum_of_vec( best_vector,  
				had_of_vec( var_weights[i], 
					arch.get_island(i)->get_population().get_individual( best_idx ).cur_x ));
		}

		// Check if Nash equilibrium is reached
		
		if( solution_within_tolerance( best_vector, last_best_vector )){
			break;
		}

		last_best_vector = best_vector;

		// Change the boundaries of each problem, this is
		// equal to fixing!
		for( problem::base::f_size_type i = 0; i<subpops; i++ ) {
			// Get problem pointer from population from
			// island. Each get creates a clone (except
			// problem_ptr), so that they must be set back
			// later on.
			base_island_ptr isl_i = arch.get_island(i);
			population pop_i = isl_i->get_population();
			problem::base &prob_i = pop_i.problem();

			// Inverse of the decision variable weight. 
			weights_type inverse_weight = inv_of_vec( var_weights[i] );

			// Calculate modified lower and upper bounds.
			std::vector< double > mod_lb = sum_of_vec(
				had_of_vec( inverse_weight, best_vector ),
				had_of_vec( var_weights[i], prob_i.get_lb() ));
			std::vector< double > mod_ub = sum_of_vec(
				had_of_vec( inverse_weight, best_vector ),
				had_of_vec( var_weights[i], prob_i.get_ub() ));

			// Change the bounds of the problem.
			prob_i.set_bounds( mod_lb, mod_ub );

			bool reinit_pop_after_bounds_change = true;

			// Change the chromosomes according to bounds.
			for ( population::size_type j = 0; j<pop.size() && 
				      reinit_pop_after_bounds_change; j++ ) {
				pop_i.set_x(j, sum_of_vec(
						had_of_vec( inverse_weight, best_vector ),
						had_of_vec( var_weights[i], pop_i.get_individual(j).cur_x )));
			}
			
			// Set population back into island, island
			// back into arch.
			isl_i->set_population( pop_i );
			arch.set_island( i, *isl_i );
		}

		// Evolve entire archipelago once
		arch.evolve_batch(1, m_threads);
	}

	// Calculate best vector for last time and push to population.
	std::vector< double > best_vector( prob_dimension, 0.0 );
	for( problem::base::f_size_type i = 0; i<subpops; i++ ) {
		best_vector = sum_of_vec( best_vector,  
			had_of_vec( var_weights[i], 
				arch.get_island(i)->get_population().champion().x ));
	}
	pop.push_back( best_vector );

        // Fill population will all individuals.
	for (problem::base::f_size_type i=0; i <subpops; ++i) {
		for (population::size_type j=0; j < pop_size; ++j) {
			pop.push_back(arch.get_island(i)->get_population().get_individual(j).cur_x);
		}
		// Add the number of fevals
		prob.add_fevals( arch.get_island(i)->get_population().problem().get_fevals() );
		prob.add_cevals( arch.get_island(i)->get_population().problem().get_cevals() );
	}

	// Get sorted list from best to worst
	std::vector<population::size_type> best_idx = pop.get_best_idx( pop.size() );

	// Get worst pop.size() - NP
	std::vector<population::size_type> worst_idx( best_idx.begin() + pop_size, best_idx.end() );

	// Sort list in descending order
	std::sort( worst_idx.begin(), worst_idx.end(), std::greater<int>());

	// Remove worst from population (back to front)
	for (population::size_type i = 0; i < worst_idx.size(); ++i) {
		pop.erase( worst_idx[i] );
	}
	
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
	s << "decomposition: WEIGHTED" << ' ';
	s << "weights:";
	switch (m_weight_generation)
	{
		case UNIFORM : 
			s << "UNIFORM" << ' ';
			break;
		case RANDOM : 
			s << "RANDOM" << ' ';
			break;
	}
	return s.str();
}

}} //namespaces

BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::algorithm::game_theory)
