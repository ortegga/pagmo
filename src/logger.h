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

// Created by Juxi Leitner on 2010-11-08.

#ifndef LOGGER_H
#define LOGGER_H

#include <vector>
#include <iostream>
#include <boost/thread/mutex.hpp>

class my_logger {
private: 
	std::string logstring;
	double max_value;
	boost::mutex m_mutex;
public:
	my_logger() {
		logstring = "";
		max_value = 99.9;
	}
	~my_logger() {}
	
	void log(std::string &str, double value) {
		// mutexing here!
		m_mutex.lock();
		
		if(value < max_value) { 
//			std::cout << "newtesting ";
			logstring = str; 
			max_value = value;
		}
		
		m_mutex.unlock();
	}

	double best_value(void) {
		return max_value;
	}

	void add_string(std::string &str) {
		logstring = str;
	}
	
	std::string get_string() {
		return logstring;
	}
	
	
	/// Stream output operator.
	friend __PAGMO_VISIBLE_FUNC std::ostream &operator<<(std::ostream &s, const my_logger &log)
	{
		s << "LoggeR:" << std::endl;
		s << log.logstring;
		return s;
	}
	
};

#endif
