#ifndef PAGMO_CUDA_ISLAND_H
#define PAGMO_CUDA_ISLAND_H

#include <set>
#include <string>

#include "island.h"
#include "config.h"
#include "algorithm/base.h"
#include "migration/base_r_policy.h"
#include "migration/base_s_policy.h"
#include "migration/best_s_policy.h"
#include "migration/fair_r_policy.h"
#include "population.h"
#include "problem/base.h"




namespace pagmo
{

    template <typename fty>
	class cuda_island<fty> : public island
    {

    public:
    cuda_island(const cuda_island<fty> & isl) : island(isl)
	{
	
	}
	explicit cuda_island(const cuda_problem<fty> & p, const algorithm::base & a, int n,
			     const double & migr_prob,
			     const migration::base_s_policy & s_policy,
			     const migration::base_r_policy & r_policy) :
	island(p,a,n,migr_prob,s_policy,r_policy)
	{

	
	}

	explicit cuda_island(const population &pop, const algorithm::base & a,
			     const double & migr_prob,
			     const migration::base_s_policy & s_policy,
			     const migration::base_r_policy & r_policy) :
	base_island(pop,a,migr_prob,s_policy,r_policy)
	{

	
	}

	cuda_island<fty> &operator = (const cuda_island<fty> & isl)
	{
	    base_island::operator=(isl);
	    return *this;
	}
	base_island_ptr clone() const
	{
	    return base_island_ptr(new cuda_island<fty>(*this));
	}
    protected:
	/** @name Evolution.
	 * Methods related to island evolution.
	 */
	//@{
	bool is_blocking_impl() const
	{
	    return false;	    
	}
	void perform_evolution(const algorithm::base &, population & pop) const
	{
	    //
	    const boost::shared_ptr<population> pop_copy(new population(pop));
	    const algorithm::base_ptr algo_copy = algo.clone();
	    const std::pair<const boost::shared_ptr<population>, const algorithm::base_ptr> out(pop_copy,algo_copy);
	    MPI_Status status;
	    int processor, flag, size;
	    {
		std::stringstream ss;
		boost::archive::text_oarchive oa(ss);
		oa << out;
		const std::string buffer_str(ss.str());
		std::vector<char> buffer_char(buffer_str.begin(),buffer_str.end());
		size = boost::numeric_cast<int>(buffer_char.size());
		lock_type lock(m_mutex);
		processor = acquire_processor();
		MPI_Send(static_cast<void *>(&size),1,MPI_INT,processor,0,MPI_COMM_WORLD);
		MPI_Send(static_cast<void *>(&buffer_char[0]),size,MPI_CHAR,processor,1,MPI_COMM_WORLD);
	    }
	    std::vector<char> buffer_char;
	    while (true) {
		{
		    lock_type lock(m_mutex);
		    MPI_Iprobe(processor,0,MPI_COMM_WORLD,&flag,&status);
		    if (flag) {
			MPI_Recv(static_cast<void *>(&size),1,MPI_INT,processor,0,MPI_COMM_WORLD,&status);
			buffer_char.resize(boost::numeric_cast<std::vector<char>::size_type>(size),0);
			MPI_Recv(static_cast<void *>(&buffer_char[0]),size,MPI_CHAR,processor,1,MPI_COMM_WORLD,&status);
			release_processor(processor);
			break;
		    }
		}
		boost::this_thread::sleep(boost::posix_time::milliseconds(100));
	    }
	    const std::string buffer_str(buffer_char.begin(),buffer_char.end());
	    std::stringstream ss(buffer_str);
	    boost::archive::text_iarchive ia(ss);
	    std::pair<boost::shared_ptr<population>,algorithm::base_ptr> in;
	    ia >> in;
	    pop = *in.first;

	}
	//@}

    private:
	
    };
}


#endif
