# -*- coding: iso-8859-1 -*-
# Copyright (C) 2004-2009 The PaGMO development team,
# Advanced Concepts Team (ACT), European Space Agency (ESA)
# http://apps.sourceforge.net/mediawiki/pagmo
# http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Developers
# http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Credits
# act@esa.int
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the
# Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
  
## @package environment
#  This module contains the ALifeEnvironment classes. 
#  The environment loads the robot and asteroid models and puts them into an ODE world.
#  The ODE world handles all physics simulation.
#  Code is based on the ode.environment module from PyBrain
#
#  @author John Glover
from pybrain.rl.environments.ode import sensors, actuators
import ode
import xode.parser
import numpy as np
        

## ALifeEnvironment class
#
#  Handles all Physics simulation. 
#  It loads the robot and asteroid models and puts them into an ODE world. 
class ALifeEnvironment(object):
    ## Constructor
    def __init__(self):
        ## @var world XODE world node
        self.world = ode.World() 
        ## @var space:  XODE space node, defined in load_xode
        self.space = ode.Space()  
        ## @var body_geom A list with (body, geom) tuples
        self.body_geom = []
        ## @var robot The robot object
        self.robot = None
        ## @var asteroid_geom The asteroid geometry
        self.asteroid_geom = None
        ## @var asteroid The asteroid object
        self.asteroid = None
        ## @var contactgroup A joint group for the contact joints that 
        # are generated whenever two bodies collide
        self.contactgroup = ode.JointGroup()
        ##
        self.passpairs = []
        ## @var grav_constant The Gravitational constant (G) used to calculate
        #  the force of gravity between bodies and the asteroid. Uses a simplified
        #  model of gravity based on Newton's law, but the forces are only applied
        #  to the bodies in the world, they do not exert any force on the asteroid.
        self.grav_constant = 1.0
        ## @var max_distance_from_asteroid The maximum distance that a body can be from
        #  the asteroid. An exception is raised by the step() function if this value is
        #  exceeded.
        self.max_distance_from_asteroid = 200
        ## @var max_force The maximum force of gravity that can be applied to a body.
        #  There seems to be a problem with ODE collision detection if the gravitational
        #  force is too large. This value is just based on trial and error.
        self.max_force = 290 
        ## @var friction Coulomb friction coefficient for contact joints 
        self.friction = 8.0
        ## @var step_count The current step number
        self.step_count = 0
        
    ## Get the list of body and geometry objects.
    #  @return list of (body, geometry) tuples.
    def get_objects(self):
        return self.body_geom
        
    def set_robot(self, robot):
        self.robot = robot
        for body_geom in robot.bodies_geoms:
            self.body_geom.append(body_geom)
        for passpair in robot.passpairs:
            self.passpairs.append(passpair)
        
    ## Adds an asteroid geometry to the environment
    #  @param asteroid
    def set_asteroid(self, asteroid):
        self.asteroid = asteroid
            
    ## Callback function for the space.collide() method.
    #  This function checks if the given geoms do collide and 
    #  creates contact joints if they do.
    #  @param args Arguments given to the space.collide() method, which calls this function.
    #  @param geom1 Geometry object that may be colliding with geom2.
    #  @param geom2 A geometry object that may be colliding with geom1.
    def _near_callback(self, args, geom1, geom2):
        # only check parse list, if objects have name
        if geom1.name != None and geom2.name != None:
            # Preliminary checking, only collide with certain objects
            for p in self.passpairs:
                g1 = False
                g2 = False
                for x in p:
                    g1 = g1 or (geom1.name.find(x) != -1)
                    g2 = g2 or (geom2.name.find(x) != -1)
                if g1 and g2:
                    return
        
        # Check if the objects do collide
        contacts = ode.collide(geom1, geom2)
        
        # Create contact joints
        world, contactgroup = args
        for c in contacts:
            p = c.getContactGeomParams()
            # parameters from Niko Wolf (taken from PyBrain code)
            c.setBounce(0.2)
            c.setBounceVel(0.05) #Set the minimum incoming velocity necessary for bounce
            c.setSoftERP(0.6) #Set the contact normal "softness" parameter
            c.setSoftCFM(0.00005) #Set the contact normal "softness" parameter
            c.setSlip2(0.02) #Set the coefficient of force-dependent-slip (FDS) for friction direction 2
            c.setMu(self.friction) #Set the Coulomb friction coefficient
            j = ode.ContactJoint(world, contactgroup, c)
            j.name = None
            j.attach(geom1.getBody(), geom2.getBody())
#            print c.getContactGeomParams() 
            
    ## Get the distance between the given body and the centre of the asteroid
    #  @param body the body
    #  @return distance between the given body and the centre of the asteroid (float)
    def _distance_to_asteroid(self, body):
        a = self.asteroid.geom.getPosition()
        b = body.getPosition()
        # change the body position, keeping it the same relative to the asteroid, 
        # but with the asteroid centred at the origin
        b = (b[0]+a[0], b[1]+a[1], b[2]+a[2])
        # get the length of the body vector
        return np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    
    ## Get the direction from the given body to the centre of the asteroid,
    #  represented as a unit vector
    #  @param body the body
    #  @return direction from the given body to the centre of the asteroid,
    #          represented as a unit vector (float x, float y, float z)
    def _direction_of_asteroid(self, body):
        a = self.asteroid.geom.getPosition()
        b = body.getPosition()
        # update the asteroid position, keeping its position the same relative to
        # the body, but with the body centred at the origin
        a = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
        # return a unit vector in the direction of the asteroid
        length = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        # check that length is non-zero
        if length:
            return (a[0]/length, a[1]/length, a[2]/length)
        else:
            return (0, 0, 0)
    
    ## Resets the environment
    def reset(self):
        self.step_count = 0
        if self.robot:
            self.robot.reset()
            
    ## Calculate the next step in the ODE environment.
    #  @param dt The step size. 
    #  @return The current step count
    def step(self, dt=0.04):    
        # update gravity
        for (body, geom) in self.body_geom:
            if body:
                # calculate the distance to the centre of the asteroid
                distance = self._distance_to_asteroid(body)
                if distance > self.max_distance_from_asteroid:
                    # todo: more detail on this exception
                    raise Exception("BodyTooFarFromAsteroid")
                # get the direction of the force
                direction = self._direction_of_asteroid(body)
                # calculate the force of gravity, based on the 
                # this distance and the masses of the asteroid and body
                m1 = self.asteroid.mass
                m2 = body.getMass().mass
                f = self.grav_constant * m1 * m2 / distance**2
                # check that the forces generated by body and asteroid masses are
                # not too large (seems to be a problem with ODE collision detection)
                # The maximum force value is currently just based on trial and error.
                # todo: can collision detected be fixed for large forces?
                #       smaller ODE step size?
                if f > self.max_force:
                    print "Warning: Gravitational force", f, "is larger than the",
                    print "current maximum force", "(" + str(self.max_force) + ")"
                    f = self.max_force
                # apply this force to the body
                force = (f*direction[0], f*direction[1], f*direction[2])
                body.addForce(force)
                # damping
                #f = body.getAngularVel()
                #scale = m2 * 0.01
                #body.addTorque((-f[0]*scale, -f[1]*scale, -f[2]*scale))
                
        # Detect collisions and create contact joints
        self.space.collide((self.world, self.contactgroup), self._near_callback)
        # Simulation step
        self.world.step(dt)
        # Remove all contact joints
        self.contactgroup.empty()
        # increase step counter
        self.step_count += 1
        return self.step_count


## APlane class
#
#  A simplification of the ALifeEnvironment that uses an infinitely flat 2D plane
#  instead of an asteroid for the ground. 
# 
#  It also uses standard ODE gravity instead of the custom gravity calculation
#  performed in ALifeEnvironment.
class ALifePlane(ALifeEnvironment):
    ## Constructor
    def __init__(self):
        super(ALifePlane, self).__init__()
        # Add a 2D plane as the ground
        p = ode.GeomPlane(self.space, (0,1,0), 0)
        p.name = "ground"
        self.body_geom.append((None, p))
        # Use standard ODE gravity
        self.world.setGravity((0.0, -4.9, 0.0))
        
    ## Calculate the next step in the ODE environment.
    #  @param dt The step size. 
    #  @return The current step count
    def step(self, dt=0.04):   
        # Detect collisions and create contact joints
        self.space.collide((self.world, self.contactgroup), self._near_callback)
        # Simulation step
        self.world.step(dt)
        # Remove all contact joints
        self.contactgroup.empty()
        # increase step counter
        self.step_count += 1
        return self.step_count