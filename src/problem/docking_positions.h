#ifndef __PAGMO_DOCKING_POSITIONS__
#define __PAGMO_DOCKING_POSITIONS__

#include <vector>
#include "../rng.h"
///////////////////////////////////////////////////////////////////////
// Finish the random sources code

namespace pagmo
{
  namespace docking_points
  {
    template <typename ty>
      class base_source
      {
      public:
	typedef typename std::vector<ty> positions_vector;
	base_source(){}
	virtual ~base_source() {};
	virtual void operator (positions_vector & results) = 0;
      };

    template <typename ty, ty r1, ty r2, bool half >
    class spoke_source : public base_source<ty>
    {
      //	  generate_spoke_positions(2.0, 2.0);
      //generate_spoke_positions(1.8, 2.0, -1);
    public:
    spoke_source() : 
      base_source()
	{
	}
      virtual void operator (positions_vector & results)
      {
	rng_double drng = rng_double(static_rng_uint32()());
	ty r, theta, x, y;	
		
	for(ty a = 0; random_start.size() < random_starting_positions; 
	    a += (2*M_PI)/random_starting_positions) 
	  {
	    r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
	    x = r * cos(a);
	    // if we select a half the points should be in that half!
	    if( (half == -1 && x > 0.0) || 
		(half == 1  && x < 0.0)  )  x = -x;		 
	    y = r * sin(a);
	    theta = drng() * 2 * M_PI;	// theta between 0-2Pi
	    // Start Condt:  x,  vx, y,  vy, theta, omega
	    ty cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
	    random_start.push_back(std::vector<ty> (cnd, cnd + 6)); //ann->get_number_of_inputs()
	  }
      }
    };

    template <typename ty, ty r1, ty r2>
    class random_source : public base_source<ty>
    {
      //	  generate_random_positions(1.8, 2.0);
    public:
    random_source() : base_source<ty>(){}
      virtual void operator (positions_vector & results)
      {
	rng_double drng = rng_double(static_rng_uint32()());
	ty r, a, theta, x, y;	
	
	while(random_start.size() < random_starting_positions) {
	  r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
	  a = drng() * 2 * M_PI;	// alpha between 0-2Pi
	  x = r * cos(a);
	  y = r * sin(a);
	  theta = drng() * 2 * M_PI;	// theta between 0-2Pi
	  // Start Condt:  x,  vx, y,  vy, theta, omega
	  ty cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
	  random_start.push_back(std::vector<ty> (cnd, cnd + ann->get_number_of_inputs()));
	}
      }
    };
    
    template <typename ty, ty r1, ty r2>
    class random_facing_orig_source : public base_source<ty>
    {
      //generate_random_positions_facing_origin(1.8, 2.0);
    public:
      random_facing_orig_source (): base_source<ty>()
	{}
      virtual void operator (positions_vector & results)
      {
	rng_double drng = rng_double(static_rng_uint32()());
	ty r, a, theta, x, y;	
	
	while(random_start.size() < random_starting_positions) {
	  r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
	  a = drng() * 2 * M_PI;	// alpha between 0-2Pi
	  x = r * cos(a);
	  y = r * sin(a);
	  theta = atan2(-y, -x);	// theta is facing 0/0
	  if(theta < 0) theta += 2 * M_PI;
	  
	  // Start Condt:  x,  vx, y,  vy, theta, omega
	  ty cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
	  random_start.push_back(std::vector<ty> (cnd, cnd + ann->get_number_of_inputs()));
	}
      }
    };
    template <typename ty, ty d, ty angle, ty rin>
      class cloud_source : public base_source<ty>
    {
      //generate_cloud_positions(2.0, M_PI, 0.1);
    public:
    cloud_source() : base_source<ty>()
	{
	}
      virtual void operator (positions_vector & results)
      {
	rng_double drng = rng_double(static_rng_uint32()());
	ty r, theta, a, x, y;
	
	ty x_start = d * cos(angle);
	ty y_start = d * sin(angle);
	
	while(random_start.size() < random_starting_positions) 
	  {
	    r = rin * drng();       // between 0 and rin
	    a = drng() * 2 * M_PI;  // alpha between 0-2Pi
	    x = x_start + r * cos(a);
	    y = y_start + r * sin(a);
	    theta = drng() * 2 * M_PI;      // theta between 0-2Pi
	    // Start Condt:  x,  vx, y,  vy, theta, omega
	    ty cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
	    random_start.push_back(std::vector<ty> (cnd, cnd + 6));
	  }
      }
    };
    
    template <typename ty, ty r1, ty r2, size_t spokes>
      class multi_spoke_source : public base_source<ty>
    {
      //	  generate_multi_spoke_positions(1.8, 2.0, 8);
    public:
    multi_spoke_source() : base_source<ty> (){}
	virtual void operator (positions_vector & results)
	{
	  rng_double drng = rng_double(static_rng_uint32()());
	  ty r, theta, x, y;	
	  
	  for(ty a = 0; random_start.size() < random_starting_positions; a += (2*M_PI)/spokes) 
	    {
	      r = r1 + (r2-r1) * drng();	// radius between 1.5 and 2
	      x = r * cos(a);
	      y = r * sin(a);
	      theta = drng() * 2 * M_PI;	// theta between 0-2Pi
	      // Start Condt:  x,  vx, y,  vy, theta, omega
	      ty cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
	      random_start.push_back(std::vector<ty> (cnd, cnd + 6)); //ann->get_number_of_inputs()
	    }
	}
    };
    
    template <typename ty, size_t h, size_t v>
      class full_grid_source : public base_source<ty>
    {
    public:
      //	  generate_full_grid_positions(5, 5);
    full_grid_source() : base_source <ty> (){}
      virtual void operator (positions_vector & results)
      {
	ty r, theta, x, y;	
	ty minx = -2, maxx = 2;
	ty miny = -2, maxy = 2;	
	for(int i = 0; i < h; i++) {
	  x = i * (maxx-minx) / (h - 1) + minx;
	  for(int j = 0; j < v; j++) {
	    y = j * (maxy-miny) / (v - 1) + miny;
	    theta = 0;//drng() * 2 * M_PI;	// theta between 0-2Pi
	    // Start Condt:  x,  vx, y,  vy, theta, omega
	    ty cnd[] = { x, 0.0, y, 0.0, theta, 0.0 };
	    random_start.push_back(std::vector<ty> (cnd, cnd + ann->get_number_of_inputs()));
	  }
	}
      }
    };

    template <typename ty, size_t points>
      class fixed_point_source : public base_source <ty>
    {
    public:
      fixed_point_source () : base_source <ty> (){}
      virtual void operator (position_vector & results)
      {
	if(random_starting_positions >= 1) 
	  {
	    ty cnd[] = { -2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
	    random_start.push_back(std::vector<ty> (cnd, cnd + ann->get_number_of_inputs()));
	  }

	if(random_starting_positions >= 2) 
	  {
	    ty cnd[] = { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 };		
	    random_start.push_back(std::vector<ty> (cnd, cnd + ann->get_number_of_inputs()));
	  }
		
	if(random_starting_positions >= 3) 
	  {
	    ty cnd[] = { -1.0, 0.0, -1.0, 0.0, 0.0, 0.0 };		
	    random_start.push_back(std::vector<ty> (cnd, cnd + ann->get_number_of_inputs()));
	  }
      }
    };
    
  }
}


#endif
