//-*-c++-*-
//Basic kernel to compute the result of a layer's operation

#ifndef __PAGMO_CUDA_KERNELS__
#define  __PAGMO_CUDA_KERNELS__

#include "stdio.h"
#include "kernel.h"


template <typename cuda_type>
struct nop_functor 
{
  __device__ __forceinline__ cuda_type operator() ( cuda_type val )
  {
    return val;
  }
};

template <typename cuda_type>
struct scale_functor 
{
  __device__ __forceinline__ cuda_type operator ()(cuda_type val, const cuda_type max_val)
  {
    return ( val - 0.5f )* 2 * max_val;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////
// computes y += alpha * x1
template <typename cuda_type>
__device__ __forceinline__ void increment(cuda_type *Y,  cuda_type *X,  const cuda_type alpha) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] += alpha * X[idx];
}

// computes y = x1 - x2
template <typename cuda_type>
__device__ __forceinline__ void assign_diff(cuda_type *Y,  cuda_type *X1,  cuda_type * X2) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = X1[idx] - X2[idx];
}

// computes y = x1 + alpha * x2
template <typename cuda_type>
__device__ __forceinline__ void assign_sum(cuda_type *Y,  cuda_type *X1,  
				      cuda_type* X2, const cuda_type alpha) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = X1[idx] + alpha * X2[idx];
}

// computes y = alpha1 * ( x1 + x2 + beta*x3 )
template <typename cuda_type>
__device__ __forceinline__ void increment_sum_sum(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
					    cuda_type* X3, const cuda_type alpha, 
					     const cuda_type beta) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = alpha*(X1[idx] + X2[idx] + beta*X3[idx]);
}

// computes y = x1 + alpha * x2 ; x2 += x3
template <typename cuda_type>
__device__ __forceinline__ void assign_sum_increment(cuda_type *Y,  cuda_type *X1,  cuda_type* X2, 
					       cuda_type* X3, cuda_type alpha) 
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Y[idx] = X1[idx] + alpha*X2[idx];
  X2[idx] += X3[idx];
}


///////////////////////////////////////////////////////////////////////
// hills equation

template <typename cuda_type, typename preprocessor>
struct hills_dynamical_system 
{
  static size_t get_size () {return 2;} 

  __device__ void operator () (cuda_type *S,  cuda_type *D,  cuda_type* O, 
			       cuda_type t, cuda_type max_val, preprocessor prec = preprocessor () )
  {

    const cuda_type nu = 0.08, mR = (1.5 * 0.5);	
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int sstride = 6*idx;
    unsigned int ostride = 2*idx;

    cuda_type x = S[sstride];
    cuda_type vx = S[++sstride];
    cuda_type y = S[++sstride];
    cuda_type vy = S[++sstride];
    cuda_type theta = S[++sstride];	
    cuda_type omega = S[++sstride];
	
    cuda_type distance = sqrt(x * x + y * y);

    if(theta < -M_PI) theta += 2 * M_PI;
    if(theta > M_PI) theta -= 2 * M_PI;
	
    cuda_type ul = prec (O[ostride], max_val);
    cuda_type ur = prec (O[++ostride], max_val);
       
    D[sstride] = (ul - ur) * 1/mR;
    D[--sstride] = omega;
    D[--sstride] = -2 * nu * vx + (ul + ur) * sin(theta);
    D[--sstride] = vy;
    D[--sstride] = 2 * nu * vy + 3 * nu * nu * x + (ul + ur) * cos(theta);
    D[--sstride] = vx;
  }
};


/////////////////////////////////////////////////////////////////////////
// runge kutta integrator
//


template <typename cuda_type, typename DynamicalSystem>

__global__ void cu_runge_kutta_integrate (cuda_type  * X , cuda_type * O, cuda_type t , cuda_type dt ,  
					  const cuda_type max_val, const size_t value_size, 
					  const size_t block_size,  DynamicalSystem system = DynamicalSystem())
{
  const size_t datasize = 6;

  cuda_type dxdt [ datasize ] ;

  cuda_type  dh = cuda_type( 0.5 ) * dt;
  cuda_type th = t + dh;

  const cuda_type val2 = cuda_type( 2.0 );

  system(X, dxdt, O, t, max_val);

  cuda_type xt [datasize];

  assign_sum( xt , X , dxdt , dh );

  cuda_type dxt [datasize];

  system( xt , dxt , O, th , max_val);

  assign_sum( xt, X, dxt , dh );

  cuda_type dxm [datasize];
  system( xt , dxm , O, th , max_val);

  assign_sum_increment( xt, X, dxm,dxt, dt );

  system( xt , dxt , O, cuda_type( t + dt ) , max_val);

  increment_sum_sum( X, dxdt, dxt,  dxm, 
		     dt /  cuda_type( 6.0 ) , val2 );

}


template <typename cuda_type, typename dynamicalsystem>
void runge_kutta_integrate (cuda_type  * X , cuda_type * O, cuda_type t , cuda_type dt , cuda_type max_val, size_t value_size, size_t block_size, dim3 g, dim3 b)
  {
    cu_runge_kutta_integrate <cuda_type, dynamicalsystem ><<<g, b>>>(X , O, t , dt, max_val, value_size, block_size);
  } 


template <>
void runge_kutta_integrate <float, hills_dynamical_system <float, scale_functor<float> > > (float  * X , float * O, float t , float dt , 
											    float max_val, size_t value_size, 
											    size_t block_size, dim3 g, dim3 b)
{
    cu_runge_kutta_integrate <float, hills_dynamical_system <float, scale_functor<float> > ><<<g, b>>>(X , O, t , dt, max_val, value_size, block_size);
  } 


template <>
void runge_kutta_integrate <double, hills_dynamical_system <double, scale_functor<double> > > (double  * X , double * O, double t , double dt , 
											    double max_val, size_t value_size, 
											    size_t block_size, dim3 g, dim3 b)
  {
    cu_runge_kutta_integrate <double, hills_dynamical_system <double, scale_functor<double> > ><<<g, b>>>(X , O, t , dt, max_val, value_size, block_size);
  } 


///////////////////////////////////////////////////////////////////////
// fitness kernels


template <typename ty, typename preprocessor>
__global__ void cu_compute_fitness_mindis_kernel(ty *S , ty *O, ty width, 
						 preprocessor prep = preprocessor())
{

  //  unsigned int bx = blockIdx.x, by = blockIdx.y;
  //Todo remove S[1] as its not used
  unsigned int tx = threadIdx.x;
  unsigned int offset = tx * 4;

  for (size_t i = 0; i < 4; ++i)
    {
       S[offset + i] = prep( S[offset + i] );
    }

  ty distance = sqrt(S[offset] * S[offset] + S[offset + 2] * S[offset + 2]);
  ty speed    = sqrt(S[offset + 1] * S[offset + 1] + S[offset + 3] * S[offset + 3]);
  O[tx] =  1/( 1 + distance );
}


template <typename ty, typename preprocessor>
void cu_compute_fitness_mindis(ty *S , ty *O, ty width, dim3 g, dim3 b )
{
  cu_compute_fitness_mindis_kernel<ty, preprocessor><<<g, b>>>(S , O, width);
}

template <>
void cu_compute_fitness_mindis<float, nop_functor<float> >(float *S , float *O, float width, dim3 g, dim3 b )
{
  cu_compute_fitness_mindis_kernel<float, nop_functor<float> ><<<g, b>>>(S , O, width);
}


template <>
void cu_compute_fitness_mindis<double, nop_functor<double> >(double *S , double *O, double width, dim3 g, dim3 b )
{
  cu_compute_fitness_mindis_kernel<double, nop_functor<double> ><<<g, b>>>(S , O, width);
}


/*template <typename ty>
__global__ void cu_compute_fitness_mindis_theta_kernel(ty *S , ty *O , ty width)
{

  //unsigned int bx = blockIdx.x, by = blockIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int offset = tx * 5;

  ty distance = sqrt(S[offset ] * S[offset] + S[offset + 2] * S[offset + 2]);
  ty speed    = sqrt(S[offset + 1] * S[offset + 1] + S[offset + 3] * S[offset + 3]);		// sqrt(vx^2 + vy^2)
  ty theta = S[offset + 4];
  // keep theta between -180 and +180°

  //Todo optimise
  if(theta < -M_PI) theta += 2 * M_PI;
    if(theta > M_PI) theta -= 2 * M_PI;	
  ty fitness =  1/( (1+distance) * (1+speed) * (1+fabs(theta)) );
}

template <typename ty>
__global__ void cu_compute_fitness_mindis_simple_kernel(ty *S , ty *O , 
							ty * init_distance, ty width)
{
  unsigned int tx = threadIdx.x;
  unsigned int offset = tx * 3;

  ty distance = sqrt(S[offset] * S[offset] + S[offset + 2] * S[offset + 2]);
  O[tx] =  0; 
  if (distance < init_distance[tx])
    {
      O[tx] = 1/( 1 + distance );
    }     
}

template <typename ty>
__global__ void cu_compute_fitness_mindis_noatt_kernel(ty *S , ty *O , 
						       ty * init_distance,ty vic_distance,  
						       ty vic_speed, ty max_dock_time, ty t,
						       ty width)
{

  unsigned int tx = threadIdx.x;
  unsigned int offset = tx * 4;

  ty distance = sqrt(S[offset] * S[offset] + S[offset + 2] * S[offset + 2]);
  ty speed    = sqrt(S[offset + 1] * S[offset + 1] + S[offset + 3] * S[offset + 3]);		// sqrt(vx^2 + vy^2)
  O[tx] = 0.0;
  if(distance < init_distance[tx]/2) {
    O[tx] = 1.0/((1 + distance) * (1 + speed));				
    if(distance < vic_distance && speed < 0.1)
      O[tx] += O[tx] * (max_dock_time - t)/max_dock_time;
  }		
}


template <typename ty>
__global__ void cu_compute_fitness_twodee1_kernel(ty *S , ty *O , 
						  ty * init_distance, ty max_docking_time, ty t,
						  ty width)
{

  unsigned int tx = threadIdx.x;
  unsigned int offset = tx * 5;

  ty distance = sqrt(S[offset] * S[offset] + S[offset + 2] * S[offset + 2]);
  ty speed    = sqrt(S[offset + 1] * S[offset + 1] + S[offset + 3] * S[offset + 3]);		// sqrt(vx^2 + vy^2)
  ty theta = S[offset + 4];
  if(theta < -M_PI) theta += 2 * M_PI;
  if(theta > M_PI) theta -= 2 * M_PI;	
  ty timeBonus = (max_docking_time - t)/max_docking_time;
  O[tx] = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
  if (init_distance[tx] > distance/2) {
    if(O[tx] > 0.87)
      O[tx] += O[tx] * timeBonus;	
  } 
  else
    O[tx] = 0;	
}

template <typename ty>
__global__ void cu_compute_fitness_twodee2_kernel(ty *S0 , ty *O , 
						  ty *init_distance,ty vic_distance,  
						  ty vic_speed, ty vic_orientation, ty max_dock_time, ty t,
						  ty width)
{

  unsigned int tx = threadIdx.x;
  ty * S = &S0[tx*5];

  ty distance = sqrt(S[0] * S[0] + S[2] * S[2]);
  ty speed    = sqrt(S[1] * S[1] + S[3] * S[3]);		// sqrt(vx^2 + vy^2)
  ty theta = S[4];
  // keep theta between -180 and +180°
  if(theta < -M_PI) theta += 2 * M_PI;
  if(theta > M_PI) theta -= 2 * M_PI;	
  // based on Christos' TwoDee function
  ty timeBonus = (max_dock_time - t)/max_dock_time;
  ty alpha = 1.0/((1+distance)*(1+fabs(theta))*(speed+1));
  if (init_distance[tx] > distance/2) {
    if (distance < vic_distance && fabs(theta) < vic_orientation && speed < vic_speed)
      O[tx] = alpha + alpha * timeBonus;	
    else
      O[tx] = alpha;
  } else
    O[tx] = 0;

}

template <typename ty>
__global__ void cu_compute_fitness_twodee3_kernel(ty *S0 , ty *O , 
						  ty *init_distance,ty vic_distance,  
						  ty vic_speed, ty vic_orientation, ty max_dock_time, ty t,
						  ty width)
{

  unsigned int bx = blockIdx.x, by = blockIdx.y;
  unsigned int tx = threadIdx.x, tty = threadIdx.y;
  ty *  S = &S0[tx*5];

  ty distance = sqrt(S[0] * S[0] + S[2] * S[2]);
  ty speed    = sqrt(S[1] * S[1] + S[3] * S[3]);		// sqrt(vx^2 + vy^2)
  ty theta = S[4];
  // keep theta between -180 and +180°
  if(theta < -M_PI) theta += 2 * M_PI;
  if(theta > M_PI) theta -= 2 * M_PI;	
  // christo's but as soon as we reach the vicinity the
  // individual gets 1.00 as fitness + then the timeBonus
  ty timeBonus = (max_dock_time - t)/max_dock_time;
  ty alpha = 1.0/((1+ distance )*(1+fabs(theta))*(speed+1));
  if (init_distance[tx] > distance / 2) {
    if (distance < vic_distance && fabs(theta) < vic_orientation && speed < vic_speed)
      O[tx] = 1 + timeBonus;	
    else
      O[tx] = alpha;
  } else
    O[tx] = 0;
}*/


/////////////////////////////////////////////////////////////////////////////
//Misc kernels


#endif 
