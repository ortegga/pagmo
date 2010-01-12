/*
 boost header: numeric/odeint/integrator_constant_step.hpp

 Copyright 2009 Karsten Ahnert
 Copyright 2009 Mario Mulansky
 Copyright 2009 Andre Bergner
*/

#ifndef BOOST_NUMERIC_ODEINT_INTEGRATOR_CONSTANT_STEP_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_INTEGRATOR_CONSTANT_STEP_HPP_INCLUDED

#include <stdexcept>

namespace odeint {


    template<
        class Stepper ,
        class DynamicalSystem ,
        class Observer
        >
    void integrate(
                   Stepper &stepper ,
                   DynamicalSystem &system ,
                   typename Stepper::time_type start_time ,
                   typename Stepper::time_type dt ,
                   typename Stepper::container_type &state ,
                   typename Stepper::time_type end_time ,
                   Observer &observer
                   )
    {
        if( start_time > end_time )
            throw std::invalid_argument( "integrate() : start_time > end_time" );

        observer( start_time , state , system );
        while( start_time < end_time ) {
            stepper.next_step( system , state , start_time , dt );
            start_time += dt;
            observer( start_time , state , system );
        }
    }
    
   

} // odeint

#endif //BOOST_NUMERIC_ODEINT_INTEGRATOR_CONSTANT_STEP_HPP_INCLUDED
