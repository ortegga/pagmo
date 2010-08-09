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
from pybrain.rl.environments.ode import sensors, actuators
import ode
import xode.parser
import numpy as np
        
## ConfigGrabber class
#
# A replacement for the pybrain.rl.environments.ode.tools.configgrab ConfigGrabber
# class, that uses a string instead of a file.
#
# @author John Glover
class ConfigGrabber:
    ## Constructor
    # @param data The XODE data
    # @param sectionId start looking for parameters only after this string has
    # been encountered in the file.
    # @param delim tuple of delimiters to identify tags
    def __init__(self, data, sectionId="", delim=("[", "]")):
        ## @var data The XODE data
        self._data = data
        ## @var sectionId start looking for parameters only after this string
        self._sectionId = sectionId.strip()
        ## @var delim tuple of delimiters to identify tags
        self._delim = delim
    
    ## Get the value for the named parameter
    #  @param name The parameter name
    #  @return: The value of the parameter 
    def getValue(self, name):
        output = []
        start = 0
        # find section if one is given
        if self._sectionId:
            start = self._data.find(self._sectionId)
            if start < 0:
                return output
            start += len(self._sectionId)
        
        # find tag with given name
        parameter_tag = self._data.find(self._delim[0]+name+self._delim[1]+"\n", start)
        if parameter_tag == -1:
            return output
        start = parameter_tag + len(self._delim[0]+name+self._delim[1]+"\n")
        # find the next delimiter
        end = self._data.find(self._delim[0], start)
        if end == -1:
            end = len(self._data)
        # get every line between start and end of delimiters
        for line in self._data[start:end].split("\n"):
            if line:
                output.append(line.strip())
        return output
        

## ALifeEnvironment class
#
#  Handles all Physics simulation. 
#  It loads the robot and asteroid models and puts them into an ODE world. 
#  Code is based on the ode.environment module from PyBrain
#
#  @author John Glover
class ALifeEnvironment(object):
    ## Constructor
    def __init__(self, robot=None, asteroid=None):
        ## @var root XODE root node, defined in load_xode
        self.root = None
        ## @var world XODE world node, defined in load_xode
        self.world = None 
        ## @var space:  XODE space node, defined in load_xode
        self.space = None  
        ## @var body_geom A list with (body, geom) tuples
        self.body_geom = []
        ## @var robot_body The robot body object, defined in _parseBodies
        self.robot_body = None
        ## @var joints The robot's joints
        self.joints = []
        ## @var robot The robot object
        self.robot = robot
        if self.robot:
            self.load_robot(robot.get_xode())
        ## @var asteroid_geom The asteroid geometry
        self.asteroid_geom = None
        ## @var asteroid The asteroid object
        self.asteroid = asteroid
        if self.asteroid:
            self.load_asteroid(asteroid.get_xode())
        ## @var contactgroup A joint group for the contact joints that 
        # are generated whenever two bodies collide
        self.contactgroup = ode.JointGroup()
        ## @var asteroid_mass The mass of the asteroid.
        #  Used when calculating the force of gravity that the asteroid exerts
        #  on the bodies in the ODE space.
        self.asteroid_mass = 100000.0
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
    
    ## Loads the robot XODE data (xml format) and parses it.
    #  @param robot_xode The XODE data for the robot.
    def load_robot(self, robot_xode):
        p = xode.parser.Parser()
        self.root = p.parseString(robot_xode)
        try:
            # filter all xode "world" objects from root, take only the first one
            world = filter(lambda x: isinstance(x, xode.parser.World), self.root.getChildren())[0]
        except IndexError:
            # malicious format, no world tag found
            raise Exception("No <world> tag found in XODE data")
        self.world = world.getODEObject()
        try:
            # filter all xode "space" objects from world, take only the first one
            space = filter(lambda x: isinstance(x, xode.parser.Space), world.getChildren())[0]
        except IndexError:
            # malicious format, no space tag found
            raise Exception("no <space> tag found in XODE data")
        self.space = space.getODEObject()
                
        # load bodies and geoms for painting
        self.body_geom = [] 
        self.joints = []
        self._parseBodies(self.root)
        
        # now parse the additional parameters at the end of the xode file
        self._loadConfig(robot_xode)
        
    ## Adds an asteroid geometry to the environment
    #  @param asteroid_xode XODE string containing the asteroid geometry
    def load_asteroid(self, asteroid_xode):
        # check for existing ODE space
        if not self.space:
            # todo: more detail on this exception
            raise Exception("NoSpace")
        
        # parse asteroid xode, adding to environment
        parser = xode.parser.Parser()
        self._parseBodies(parser.parseString(asteroid_xode))
        
        # check that asteroid geometry was created successfully
        # todo: check triangle count: self.asteroid_geom.getTriangleCount()
        # todo: give more detail in exception
        if not self.asteroid_geom:
            raise Exception("NoAsteroid")
        # add asteroid to current space
        self.asteroid_geom.getSpace().remove(self.asteroid_geom)
        self.space.add(self.asteroid_geom)
    
    ## Load the XODE config.
    #  @param data The XODE data string
    def _loadConfig(self, data):
        # parameters are given in (our own brand of) config-file syntax
        self.config = ConfigGrabber(data, sectionId="<!--odeenvironment parameters", delim=("<", ">"))

        # <passpairs>
        self.passpairs = []
        for passpairstring in self.config.getValue("passpairs")[:]:
            self.passpairs.append(eval(passpairstring))

        # <affixToEnvironment>
        for jointName in self.config.getValue("affixToEnvironment")[:]:
            try:
                # find first object with that name
                obj = self.root.namedChild(jointName).getODEObject()
            except IndexError:
                print "ERROR: Could not affix object '" + jointName + "' to environment!"
                sys.exit(1)
            if isinstance(obj, ode.Joint):
                # if it is a joint, use this joint to fix to environment
                obj.attach(obj.getBody(0), ode.environment)
            elif isinstance(obj, ode.Body):
                # if it is a body, create new joint and fix body to environment
                j = ode.FixedJoint(self.world)
                j.attach(obj, ode.environment)
                j.setFixed()

        # <colors>
        for coldefstring in self.config.getValue("colors")[:]:
            objname, coldef = eval(coldefstring)
            for (body, _) in self.body_geom:
                if hasattr(body, 'name'):
                    if objname == body.name:
                        body.color = coldef
                        break

    ## Parse the given xode node and all children (recursively), creating ODE body and geometry objects.
    #  @param node The XODE node.
    def _parseBodies(self, node):
        # body (with nested geom)
        if isinstance(node, xode.body.Body):
            body = node.getODEObject()
            body.name = node.getName()
            if body.name == "robot_body":
                self.robot_body = body
            try:
                # filter all xode geom objects and take the first one
                xgeom = filter(lambda x: isinstance(x, xode.geom.Geom), node.getChildren())[0]
            except IndexError:
                return() # no geom object found, skip this node
            # get the real ode object
            geom = xgeom.getODEObject()
            # if geom doesn't have own name, use the name of its body
            geom.name = node.getName()
            self.body_geom.append((body, geom))
        
        # geom on its own without body
        elif isinstance(node, xode.geom.Geom):
            try:
                node.getFirstAncestor(ode.Body)
            except xode.node.AncestorNotFoundError:
                body = None
                geom = node.getODEObject()
                geom.name = node.getName()
                self.body_geom.append((body, geom))
                if geom.name == "Asteroid":
                    self.asteroid_geom = geom
        
        # special cases for joints: universal, fixed, amotor
        elif isinstance(node, xode.joint.Joint):
            joint = node.getODEObject()
            self.joints.append(joint)

        # recursive call for all child nodes
        for c in node.getChildren():
            self._parseBodies(c)
            
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
            
    ## Get the distance between the given body and the centre of the asteroid
    #  @param body the body
    #  @return distance between the given body and the centre of the asteroid (float)
    def _distance_to_asteroid(self, body):
        a = self.asteroid_geom.getPosition()
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
        a = self.asteroid_geom.getPosition()
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
            
    ## Get the list of body and geometry objects.
    #  @return list of (body, geometry) tuples.
    def get_objects(self):
        return self.body_geom
    
    ## Get the robot body object
    # @return robot body object
    def get_robot_body(self):
        return self.robot_body
    
    ## Get the position of the robot body
    # @return position of the robot body
    def get_robot_position(self):
        return self.robot_body.getPosition()
    
    ## Get the robot's joints
    #  @return list of robot's joints
    def get_robot_joints(self):
        return self.joints
    
    ## Resets the environment
    def reset(self):
        self.step_count = 0
        self.body_geom = []
        if self.robot:
            self.load_robot(self.robot.get_xode())
        if self.asteroid:
            self.load_asteroid(self.asteroid.get_xode())
            
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
                m1 = self.asteroid_mass
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
#
#  @author John Glover
class ALifePlane(ALifeEnvironment):
    ## Loads the robot XODE data (xml format) and parses it.
    #  @param robot_xode The XODE data for the robot.
    def load_robot(self, robot_xode):
        super(ALifePlane, self).load_robot(robot_xode)
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