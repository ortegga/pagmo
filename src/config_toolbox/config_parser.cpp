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

// Created by Juxi Leitner on 2010-03-15.

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>        
#include <exception>


#include "../exceptions.h"
#include "config_parser.h"

#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;

// Constructor
config_parser::config_parser(char *fname) : 
    filename(fname)
{
	if(filename != NULL) read_file();
}

void config_parser::read_file() {
	if(filename == NULL) {
//		cout << "No filename specified!" << endl;
		return;
	}

	fstream file;
	file.open(filename, ios::in);
	if ( file.fail() ) {
		std::string error = "Error opening "; error += filename;
		pagmo_throw(value_error, error.c_str());
		exit(0);
	}
	char h[256];
	std::string line;
	// add for
	while(! file.eof()) {
		file.getline(h, 256);
		line = h;
		boost::algorithm::trim(line);
		if((! line.empty()) &&  (line.at(0) != '#')) {
			// no empty lines or commented lines are added
			read_lines.push_back(line);
		}
	}
    file.close();
	std::cout << "Config file loaded!" << std::endl;
}

//
void config_parser::read_parameter(const char *param, int &value) {
	if(filename == NULL) return;
	// loop thru all lines
	int pos; std::string h;
	for(int i = 0;i < read_lines.size(); i++) {
		pos = read_lines[i].find(":");
		if((pos != string::npos) &&
			read_lines[i].substr(0, pos).compare(param) == 0) {
			// we found it =)
			// parse the value!
			h = read_lines[i].substr(pos + 1);
			boost::algorithm::trim(h);
			value = boost::lexical_cast<int>(h);
			// delete it from the lines
			read_lines.erase(read_lines.begin() + i);
			return;
		}
	}
}

void config_parser::read_parameter(const char *param, double &value) {
	if(filename == NULL) return;
	// loop thru all lines
	int pos; std::string h;
	for(int i = 0;i < read_lines.size(); i++) {
		pos = read_lines[i].find(":");
		if((pos != string::npos) &&
			read_lines[i].substr(0, pos).compare(param) == 0) {
			// we found it =)
			// parse the value!
			h = read_lines[i].substr(pos + 1);
			boost::algorithm::trim(h);
			value = boost::lexical_cast<double>(h);
			// delete it from the lines
			read_lines.erase(read_lines.begin() + i);
			return;
		}
	}
}


/*void config_parser::read_parameter(char *parameter, double &value) {
	if(filename == NULL) return;
	// loop thru all lines
}*/

void config_parser::report() {
	// if read_lines still has members!
	if(read_lines.size() > 0) {
		cout << "config warning: the following values are defined in the conf file but not read:" <<endl;
		cout << "\t";
		while(read_lines.size() > 0) {
			cout << read_lines.back() << ", ";
			read_lines.pop_back();
		}
		cout << endl;cout << endl;
	}
	cout << "Finished reading config parameters!" << endl;
}


// Destructor
config_parser::~config_parser() {}