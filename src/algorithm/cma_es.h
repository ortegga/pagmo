/* --------------------------------------------------------- */
/* --- File: cmaes.h ----------- Author: Nikolaus Hansen --- */
/* ---------------------- last modified: IX 2010         --- */
/* --------------------------------- by: Nikolaus Hansen --- */
/* --------------------------------------------------------- */
/*   
     CMA-ES for non-linear function minimization. 

     Copyright (C) 1996, 2003-2010  Nikolaus Hansen. 
     e-mail: nikolaus.hansen (you know what) inria.fr
      
     License: see file cmaes.c
   
*/

/* C++-ified by Juxi Leitner <juxi.leitner@gmail.com>
 * Date: 11/01/20
 */

#ifndef PAGMO_ALGORITHM_CMA_ES_H
#define PAGMO_ALGORITHM_CMA_ES_H

#include "../config.h"
#include "../problem/base.h"
#include "../serialization.h"
#include "base.h"


#include <time.h>
#include <string>

typedef struct 
/* random_t 
 * sets up a pseudo random number generator instance 
 */
{
  /* Variables for Uniform() */
  long int startseed;
  long int aktseed;
  long int aktrand;
  long int *rgrand;
  
  /* Variables for Gauss() */
  short flgstored;
  double hold;
} random_t;

typedef struct 
/* timings_t 
 * time measurement, used to time eigendecomposition 
 */
{
  /* for outside use */
  double totaltime; /* zeroed by calling re-calling timings_start */
  double totaltotaltime;
  double tictoctime; 
  double lasttictoctime;
  
  /* local fields */
  clock_t lastclock;
  time_t lasttime;
  clock_t ticclock;
  time_t tictime;
  short istic;
  short isstarted; 

  double lastdiff;
  double tictoczwischensumme;
} timings_t;

typedef struct 
/* readpara_t
 * collects all parameters, in particular those that are read from 
 * a file before to start. This should split in future? 
 */
{
  /* input parameter */
  int N; /* problem dimension, must stay constant */
  unsigned int seed; 
  double * xstart; 
  double * typicalX; 
  int typicalXcase;
  double * rgInitialStds;
  double * rgDiffMinChange; 

  /* termination parameters */
  double stopMaxFunEvals; 
  double facmaxeval;
  double stopMaxIter; 
  struct { int flg; double val; } stStopFitness; 
  double stopTolFun;
  double stopTolFunHist;
  double stopTolX;
  double stopTolUpXFactor;

  /* internal evolution strategy parameters */
  int lambda;          /* -> mu, <- N */
  int mu;              /* -> weights, (lambda) */
  double mucov, mueff; /* <- weights */
  double *weights;     /* <- mu, -> mueff, mucov, ccov */
  double damps;        /* <- cs, maxeval, lambda */
  double cs;           /* -> damps, <- N */
  double ccumcov;      /* <- N */
  double ccov;         /* <- mucov, <- N */
  double diagonalCov;  /* number of initial iterations */
  struct { int flgalways; double modulo; double maxtime; } updateCmode;
  double facupdateCmode;

  /* supplementary variables */

  char *weigkey; 
  char resumefile[99];
  char **rgsformat;
  void **rgpadr;
  char **rgskeyar;
  double ***rgp2adr;
  int n1para, n1outpara;
  int n2para;
} readpara_t;

typedef struct 
/* cmaes_t 
 * CMA-ES "object" 
 */
{
  random_t rand; /* random number generator */

  double sigma;  /* step size */

  double *rgxmean;  /* mean x vector, "parent" */
  double *rgxbestever; 
  double **rgrgx;   /* range of x-vectors, lambda offspring */
  int *index;       /* sorting index of sample pop. */
  double *arFuncValueHist;

  short flgIniphase; /* not really in use anymore */
  short flgStop; 

  double chiN; 
  double **C;  /* lower triangular matrix: i>=j for C[i][j] */
  double **B;  /* matrix with normalize eigenvectors in columns */
  double *rgD; /* axis lengths */

  double *rgpc;
  double *rgps;
  double *rgxold; 
  double *rgout; 
  double *rgBDz;   /* for B*D*z */
  double *rgdTmp;  /* temporary (random) vector used in different places */
  double *rgFuncValue; 
  double *publicFitness; /* returned by cmaes_init() */

  double gen; /* Generation number */
  double countevals;
  double state; /* 1 == sampled, 2 == not in use anymore, 3 == updated */

  double maxdiagC; /* repeatedly used for output */
  double mindiagC;
  double maxEW;
  double minEW;

  short flgEigensysIsUptodate;
  short flgCheckEigen; /* control via signals.par */
  double genOfEigensysUpdate; 
  timings_t eigenTimings;
 
  double dMaxSignifKond; 				     
  double dLastMinEWgroesserNull;

  short flgresumedone; 

  time_t printtime; 
  time_t writetime; /* ideally should keep track for each output file */
  time_t firstwritetime;
  time_t firstprinttime; 

} cmaes_t; 



/* --------------------------------------------------------- */
/* ------------------ Class Definition --------------------- */
/* --------------------------------------------------------- */

class cmaes  {
public:
	/* --- initialization, constructors, destructors --- */
	cmaes(int dimension = 0, double *xstart = NULL, 
		  double *stddev = NULL, long seed = 0, int lambda = 0,
		  const char *input_parameter_filename = NULL);
	~cmaes();
	void exit(void);
	
	void resume_distribution(char *filename);
	
	
	/* --- core functions --- */
	double * const * SamplePopulation(void);
	double *         UpdateDistribution(const double *rgFitnessValues);
	const char *     TestForTermination(void);
	
	/* --- additional functions --- */
	double * const * ReSampleSingle(int index);
	double const *   ReSampleSingle_old(double *rgx); 
	double *         SampleSingleInto(double *rgx);
	void             UpdateEigensystem(int flgforce);
	
	/* --- getter functions --- */
	double         Get(char const *keyword);
	const double * GetPtr(char const *keyword); /* e.g. "xbestever" */
	double *       GetNew(char const *keyword); 
	double *       GetInto(char const *keyword, double *mem); 
	int			   GetDimension(void);
	
	double *	   GetPublicFitness();
	
	/* --- online control and output --- */
	void           ReadSignals(char const *filename);
	void           WriteToFile(const char *szKeyWord,
							   const char *output_filename); 
	const char *   SayHello(void);
	
private:
	cmaes_t *t;
	readpara_t rp;
	
	std::string version;
	std::string outstr;
	
	
	void init(int dimension, double *inxstart,
                double *inrgstddev, /* initial stds */
                long int inseed, int lambda, 
			   const char *input_parameter_filename);
		
	void ReadFromFilePtr( FILE *fp );
	void WriteToFileAW(const char *key, const char *name, 
							 char * append);
	void WriteToFilePtr(const char *key, FILE *fp);

	double const * SetMean(const double *xmean);

	void Adapt_C2(int hsig);
	void TestMinStdDevs(void);
	
	double * PerturbSolutionInto(double *xout, double const *xin, double eps);

};

/* --- misc --- */
double *       cmaes_NewDouble(int n);  
void           cmaes_FATAL(char const *s1, char const *s2, char const *s3, 
						   char const *s4);



	namespace pagmo { namespace algorithm {

	/**
	 * TODO
	 */

	class __PAGMO_VISIBLE cma_es : public base
	{
	public:

	/*	/// Crossover operator info
		struct crossover {
			/// Crossover type, binomial or exponential
			enum type {BINOMIAL = 0, EXPONENTIAL = 1};
		};*/
		cma_es( /*int gen  = 1, const double &cr = .5, const double &m = .5, int elitism = 1,
		    mutation::type mut  = mutation::GAUSSIAN, double width = 0.05,
		    selection::type sel = selection::ROULETTE,
		    crossover::type cro = crossover::EXPONENTIAL*/);
		base_ptr clone() const;
		void evolve(population &) const;
		std::string get_name() const;
	protected:
		std::string human_readable_extra() const;
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive &ar, const unsigned int)
		{
	// TODO
	/*		ar & boost::serialization::base_object<base>(*this);
			ar & const_cast<int &>(m_gen);
			ar & const_cast<double &>(m_cr);
			ar & const_cast<double &>(m_m);
			ar & const_cast<int &>(m_elitism);
			ar & const_cast<mutation &>(m_mut);
			ar & const_cast<selection::type &>(m_sel);
			ar & const_cast<crossover::type &>(m_cro);*/
		}  
		
		mutable cmaes *cma;
		
	};

	}} //namespaces

	BOOST_CLASS_EXPORT_KEY(pagmo::algorithm::cma_es);



#endif 
