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



#ifndef __PAGMO_CUDA_TIMER_H__
#define __PAGMO_CUDA_TIMER_H__

#include <string>
#include <ostream>
#include "pagmo_cuda.h"
#include "boost/shared_ptr.hpp"


namespace cuda
{
    class timer;


    ///times_keeper class
/*
  Singleton class thats used by the scoped_timer class to save time
  used
 */
    class times_keeper 
    {
    public:
	~times_keeper();
	static times_keeper & get_instance();
	bool add_time(timer & t);
    private:
	times_keeper(std::ostream & os);
	times_keeper(times_keeper & ); // no copy
	std::ostream & m_ostream;
	static boost::shared_ptr<times_keeper> m_ptr;
    };

    /// timer class
    /*
      uses cuda events to record time elapsed
     */
    class timer
    {
    public: 
	timer(const std::string & description, cudaStream_t st = 0);
	virtual bool start();
	virtual bool stop();
	virtual bool started() { return m_started;} 
	virtual float get_elapsed() { return m_elapsed;} 
	virtual ~timer();
	friend std::ostream & operator <<(std::ostream & os, const timer & t);
    protected:

	cudaEvent_t m_start;
	cudaEvent_t  m_stop;
	bool m_started;
	std::string m_description;
	float m_elapsed;
	cudaStream_t m_stream;

    };

    /// scoped_timer
    /*
      log on destroy timer
     */
    class scoped_timer : public timer
    {
    public:
    scoped_timer(const std::string & description, cudaStream_t st = 0) 
	: timer(description, st)
	{
	    start();
	}
	~scoped_timer();
    };
}
#endif
