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


#include "beume3d.h"

namespace pagmo { namespace util { namespace hv_algorithm {

// Copy constructor
beume3d::beume3d(const beume3d &orig) : m_initial_sorting(orig.m_initial_sorting) { }

/// Constructor
/**
 * Constructor of the algorithm object.
 * In the very first step, algorithm requires the inital set of points to be sorted ASCENDING in the third dimension.
 * If the input is already sorted, user can skip this step using "initial_sorting = false" option, saving some extra time.
 *
 * @param[in] initial_sorting when set to true (default), algorithm will sort the points ascending by third dimension
 */
beume3d::beume3d(bool initial_sorting) : m_initial_sorting(initial_sorting) { }

/// Compute hypervolume 
/**
 * This method should be used both as a solution to 3D cases, and as a general termination method for algorithms that reduce D-dimensional problem to 3-dimensional one.
 *
 * This is the implementation of the Beume3D algorithm for computing hypervolume.
 * The implementation uses std::multiset (which is based on red-black tree data structure) as a container for the sweeping front.
 * Original implementation by Beume et. al uses AVL-tree.
 * The difference is insiginificant as the important characteristics (maintaining order, self-balancing) of both structures and the asymptotical times (O(log n) updates) are guaranteed.
 * Computational complexity: O(n*log(n))
 *
 * @see "On the Complexity of Computing the Hypervolume Indicator", Nicola Beume, Carlos M. Fonseca, Manuel Lopez-Ibanez,
 * Luis Paquete, Jan Vahrenhold. IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 13, NO. 5, OCTOBER 2009
 *
 * @param[in] points vector of points containing the 3-dimensional points for which we compute the hypervolume
 * @param[in] r_point reference point for the points
 *
 * @return hypervolume.
 */
double beume3d::compute(std::vector<fitness_vector> &points, const fitness_vector &r_point)
{
	if (m_initial_sorting) {
		sort(points.begin(), points.end(), fitness_vector_cmp(2,'<'));
	}
	double V = 0.0; // hypervolume
	double A = 0.0; // area of the sweeping plane
	std::multiset<fitness_vector, fitness_vector_cmp> T(fitness_vector_cmp(0, '>'));

	// sentinel points (r_point[0], -INF, r_point[2]) and (-INF, r_point[1], r_point[2])
	const double INF = std::numeric_limits<double>::max();
	fitness_vector sA(r_point.begin(), r_point.end()); sA[1] = -INF;
	fitness_vector sB(r_point.begin(), r_point.end()); sB[0] = -INF;

	T.insert(sA);
	T.insert(sB);
	double z3 = points[0][2];
	T.insert(points[0]);
	A = fabs((points[0][0] - r_point[0]) * (points[0][1] - r_point[1]));

	std::multiset<fitness_vector>::iterator p;
	std::multiset<fitness_vector>::iterator q;
	for(std::vector<fitness_vector>::size_type idx = 1 ; idx < points.size() ; ++idx) {
		p = T.insert(points[idx]);
		q = (p);
		++q; //setup q to be a successor of p
		if ( (*q)[1] <= (*p)[1] ) { // current point is dominated
			T.erase(p); // disregard the point from further calculation
		} else {
			V += A * fabs(z3 - (*p)[2]);
			z3 = (*p)[2];
			std::multiset<fitness_vector>::reverse_iterator rev_it(q);
			++rev_it;

			std::multiset<fitness_vector>::reverse_iterator erase_begin (rev_it);
			std::multiset<fitness_vector>::reverse_iterator rev_it_pred;
			while((*rev_it)[1] >= (*p)[1] ) {
				rev_it_pred = rev_it;
				++rev_it_pred;
				A -= fabs(((*rev_it)[0] - (*rev_it_pred)[0])*((*rev_it)[1] - (*q)[1]));
				++rev_it;
			}
			A += fabs(((*p)[0] - (*(rev_it))[0])*((*p)[1] - (*q)[1]));
			T.erase(rev_it.base(),erase_begin.base());
		}
	}
	V += A * fabs(z3 - r_point[2]);

	return V;
}

/// comparator method for hycon3d algorithm's tree structure
bool beume3d::hycon3d_tree_cmp::operator()(const std::pair<fitness_vector, int> &a, const std::pair<fitness_vector, int> &b) {
	return a.first[0] > b.first[0];
}

/// box_volume method
/**
 * Returns the volume of the box3d object
 */
double beume3d::box_volume(const box3d &b) {
	return fabs((b.ux - b.lx) * (b.uy - b.ly) * (b.uz - b.lz));
}

// comparator method for the hycon3d algorithm's sorting procedure
bool beume3d::hycon3d_sort_cmp(const std::pair<fitness_vector, unsigned int> &a, const std::pair<fitness_vector, unsigned int> &b) {
	return a.first[2] < b.first[2];
}

/// least_contributor method
/**
 * This method overloads the virtual method in pagmo::util::hv_algorithm::base.
 * This method relies on the HyCon3D algorithm as it was presented in the article by Emmerich and Fonseca.
 *
 * @see "Computing hypervolume contribution in low dimensions: asymptotically optimal algorithm and complexity results", Michael T. M. Emmerich, Carlos M. Fonseca
 *
 * @param[in] points vector of points containing the 3-dimensional points for which we compute the hypervolume
 * @param[in] r_point reference point for the points
 *
 * @return index of the least contributor.
 */
unsigned int beume3d::least_contributor(std::vector<fitness_vector> &points, const fitness_vector &r_point) {
	return extreme_contributor(points, r_point, base::cmp_least);
}

/// greatest_contributor method
/**
 * This method overloads the virtual method in pagmo::util::hv_algorithm::base.
 * This method relies on the HyCon3D algorithm as it was presented in the article by Emmerich and Fonseca.
 *
 * @see "Computing hypervolume contribution in low dimensions: asymptotically optimal algorithm and complexity results", Michael T. M. Emmerich, Carlos M. Fonseca
 *
 * @param[in] points vector of points containing the 3-dimensional points for which we compute the hypervolume
 * @param[in] r_point reference point for the points
 *
 * @return index of the greatest contributor.
 */
unsigned int beume3d::greatest_contributor(std::vector<fitness_vector> &points, const fitness_vector &r_point) {
	return extreme_contributor(points, r_point, base::cmp_greatest);
}

/// extreme_contributor method
/**
 * This method elicites an "extreme" contributor (either min or max item) from a set of contributions computed by HyCon3D algorithm.
 */
unsigned int beume3d::extreme_contributor(std::vector<fitness_vector> &points, const fitness_vector &r_point, bool (*cmp_func)(double, double)) {

	const std::vector<double> contributions = hycon3d(points, r_point);

	unsigned int extr_i = 0;
	double extr_c = contributions[extr_i];
	for(unsigned int i = 1 ; i < contributions.size() ; ++i) {
		if (cmp_func(contributions[i], extr_c)) {
			extr_i = i;
			extr_c = contributions[i];
		}
	}

	return extr_i;
}

/// HyCon3D algorithm
/*
 * This algorithm computes the exclusive contribution to the hypervolume object by every point, using an efficient HyCon3D algorithm by Emmerich and Fonseca.
 */
std::vector<double> beume3d::hycon3d(std::vector<fitness_vector> &points, const fitness_vector &r_point) {
	std::vector<std::pair<fitness_vector, unsigned int> > point_pairs;
	point_pairs.reserve(points.size());
	for(unsigned int i = 0 ; i < points.size() ; ++i) {
		point_pairs.push_back(std::make_pair(points[i], i));
	}
	sort(point_pairs.begin(), point_pairs.end(), hycon3d_sort_cmp);
	for(unsigned int i = 0 ; i < points.size() ; ++i) {
		points[i] = point_pairs[i].first;
	}

	typedef std::multiset<std::pair<fitness_vector, int>, hycon3d_tree_cmp > tree_t;

	unsigned int n = points.size();
	const double INF = std::numeric_limits<double>::max();

	// Placeholder value for undefined lower z value.
	const double NaN = INF;

	// Contributions
	std::vector<double> c(n, 0.0);

	// Sentinel points
	fitness_vector s_x(3, -INF); s_x[0] = r_point[0]; // (r,oo,oo)
	fitness_vector s_y(3, -INF); s_y[1] = r_point[1]; // (oo,r,oo)
	fitness_vector s_z(3, -INF); s_z[2] = r_point[2]; // (oo,oo,r)

	points.push_back(s_z); // points[n]
	points.push_back(s_x); // points[n + 1]
	points.push_back(s_y); // points[n + 2]

	tree_t T;
	T.insert(std::make_pair(points[0], 0));
	T.insert(std::make_pair(s_x, n + 1));
	T.insert(std::make_pair(s_y, n + 2));

	// Boxes
	std::vector<std::deque<box3d> > L(n + 3);

	box3d b(r_point[0], r_point[1], NaN, points[0][0], points[0][1], points[0][2]);
	L[0].push_front(b);

	for (unsigned int i = 1 ; i < n + 1 ; ++i) {
		std::pair<fitness_vector, int> pi(points[i], i);

		tree_t::iterator it = T.lower_bound(pi);

		// Point is dominated
		if (points[i][1] >= (*it).first[1]) {
			continue;
		}

		tree_t::reverse_iterator r_it(it);

		std::vector<int> d;

		while((*r_it).first[1] > points[i][1]) {
			d.push_back((*r_it).second);
			++r_it;
		}
		tree_t::iterator erase_begin = r_it.base();
		tree_t::iterator erase_end(it);

		T.erase(erase_begin, erase_end);

		int r = (*it).second;
		int t = (*r_it).second;

		// Process right neighbor region, region R
		while(!L[r].empty()) {
			box3d& b = L[r].front();
			if(b.ux >= points[i][0]) {
				b.lz = points[i][2];
				c[r] += box_volume(b);
				L[r].pop_front();
			} else if(b.lx > points[i][0]) {
				b.lz = points[i][2];
				c[r] += box_volume(b);
				b.lx = points[i][0];
				b.uz = points[i][2];
				b.lz = NaN;
				break;
			} else {
				break;
			}
		}

		// Process dominated points, region M
		double xleft = points[t][0];
		std::vector<int>::reverse_iterator r_it_idx = d.rbegin();
		std::vector<int>::reverse_iterator r_it_idx_e = d.rend();
		for(;r_it_idx != r_it_idx_e ; ++r_it_idx) {
			int jdom = *r_it_idx;
			while(!L[jdom].empty()) {
				box3d& b = L[jdom].front();
				b.lz = points[i][2];
				c[jdom] += box_volume(b);
				L[jdom].pop_front();
			}
			L[i].push_back(box3d(xleft, points[jdom][1], NaN, points[jdom][0], points[i][1], points[i][2]));
			xleft = points[jdom][0];
		}
		L[i].push_back(box3d(xleft, points[r][1], NaN, points[i][0], points[i][1], points[i][2]));
		xleft = points[t][0];

		// Process left neighbor region, region L
		while(!L[t].empty()) {
			box3d &b = L[t].back();
			if(b.ly > points[i][1]) {
				b.lz = points[i][2];
				c[t] += box_volume(b);
				xleft = b.lx;
				L[t].pop_back();
			} else {
				break;
			}
		}
		if (xleft > points[t][0]) {
			L[t].push_back(box3d(xleft, points[i][1], NaN, points[t][0], points[t][1], points[i][2]));
		}
		T.insert(std::make_pair(points[i], i));
	}

	// Fix the indices
	std::vector<double> contribs;
	contribs.reserve(n);
	for(unsigned int i=0;i < c.size();++i) {
		contribs.push_back(c[point_pairs[i].second]);
	}
	return contribs;
}

// verify_before_compute
/**
 * Verifies whether given algorithm suits the requested data.
 *
 * @param[in] points vector of points containing the d dimensional points for which we compute the hypervolume
 * @param[in] r_point reference point for the vector of points
 *
 * @throws value_error when trying to compute the hypervolume for the dimension other than 3 or non-maximal reference point
 */
void beume3d::verify_before_compute(const std::vector<fitness_vector> &points, const fitness_vector &r_point) {
	if (r_point.size() != 3) {
		pagmo_throw(value_error, "Algorithm Beume3D works only for 3-dimensional cases");
	}

	base::assert_maximal_reference_point(points, r_point);
}

/// Clone method.
base_ptr beume3d::clone() const
{
	return base_ptr(new beume3d(*this));
}

/// Algorithm name
std::string beume3d::get_name() const {
	return "Beume3D algorithm";
}

} } }

BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::util::hv_algorithm::beume3d);
