

#ifndef __PAGMO_CUDA_TIMER_H__
#define __PAGMO_CUDA_TIMER_H__

#include <string>
#include <ostream>
#include "pagmo_cuda.h"
#include "boost/shared_ptr.hpp"


namespace cuda
{
    class timer;

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
