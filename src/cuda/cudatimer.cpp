

#include "cudatimer.h"
#include <iostream>



namespace cuda
{
    timer::timer(const std::string & desc, cudaStream_t st) : 
	m_start(0), m_stop(0), m_started(false), 
	m_description(desc), m_elapsed(0.0f), m_stream(st)
    {
  
    }

    timer::~timer()
    {
	if (m_started)
	{
	    stop();
	}
    }

    bool timer::start()
    {
	if (!m_started)
	{
	    cudaEventCreate(&m_start);
	    cudaEventCreate(&m_stop);
	    cudaEventRecord(m_start, 0);
	    m_elapsed = 0.0f;
	    return m_started = true;
	}
	return false;
    }

    bool timer::stop()
    {
	if (m_started)
	{
	    cudaEventRecord(m_stop, 0);
	    cudaEventSynchronize(m_stop);
	    cudaEventElapsedTime(&m_elapsed, m_start, m_stop);
	    cudaEventDestroy(m_start);
	    cudaEventDestroy(m_stop);
	    m_started = false;
	    return true;
	}
	return false;
    }


    std::ostream & operator <<(std::ostream & os, const timer & t)
    {
	if (t.m_started)
	{
	    os<<"Event: "<<t.m_description<<" "<<" invalid state "<<std::endl;
	}
	os<<"Event: "<<t.m_description<<" "<<t.m_elapsed<<" ms elapsed"<<std::endl;
	return os;
    }


    scoped_timer::~scoped_timer()
    {
	stop();
	times_keeper::get_instance().add_time(*this);
    }


    times_keeper::times_keeper(std::ostream & os) 
	: m_ostream(os)
    {
  
    }

    times_keeper::~times_keeper() {}

    bool times_keeper::add_time(timer & t)
    {
	m_ostream<<t;
	return true;
    }

    boost::shared_ptr<times_keeper> times_keeper::m_ptr;

    times_keeper & times_keeper::get_instance()
    {
	if (!m_ptr)
	{
	    m_ptr =  boost::shared_ptr< times_keeper> (new times_keeper(std::cout));
	}
	return *m_ptr;
    }

}
