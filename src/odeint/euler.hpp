/* Boost odeint/euler.hpp header file
 
 Copyright 2009 Karsten Ahnert
 Copyright 2009 Mario Mulansky
 Copyright 2009 Andre Bergner
 
 This file includes the explicit euler solver for ordinary differential equations.

 It solves any ODE dx/dt = f(x,t) via
 x(t+dt) = x(t) + dt*f(x,t)
*/

#ifndef BOOST_NUMERIC_ODEINT_EULER_HPP
#define BOOST_NUMERIC_ODEINT_EULER_HPP

#include "detail/iterator_algebra.hpp"
#include "resizer.hpp"



namespace odeint {

    template<
        class Container ,
        class Time = double ,
        class Resizer = resizer< Container >
        >
    class ode_step_euler
    {


        // provide basic typedefs
    public:

        typedef Container container_type;
        typedef Resizer resizer_type;
        typedef Time time_type;
	typedef const unsigned short order_type;
        typedef typename container_type::value_type value_type;
        typedef typename container_type::iterator iterator;




        // private members
    private:

        container_type m_dxdt;
        resizer_type m_resizer;




        // public interface
    public:

        order_type order() const { return 1; }



        template< class DynamicalSystem >
        void next_step( DynamicalSystem &system ,
                        container_type &x ,
                        const container_type &dxdt ,
                        time_type t ,
                        time_type dt )
        {
            detail::it_algebra::increment( x.begin() , x.end() , dxdt.begin() , dt );
        }

        template< class DynamicalSystem >
        void next_step( DynamicalSystem &system ,
                        container_type &x ,
                        time_type t ,
                        time_type dt )
        {
            m_resizer.adjust_size( x , m_dxdt );
            system( x , m_dxdt , t );
            next_step( system , x , m_dxdt , t , dt );
        }
    };



} // namespace odeint


#endif // BOOST_NUMERIC_ODEINT_EULER_HPP
