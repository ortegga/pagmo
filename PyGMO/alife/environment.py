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
#  This module contains the ALifeEnvironment class and the Robot class. 
#  The environment loads the robot and asteroid models and puts them into an ODE world.
#  The ODE world handles all physics simulation.
#  Code is based on the ode.environment module from PyBrain
from pybrain.rl.environments.ode import sensors, actuators
from pybrain.rl.environments.ode.tools.xodetools import XODEfile
import ode
import xode.parser
import xml.dom.minidom as md
import numpy as np


## StringWriter class
# 
#  Utility class that is basically just a string with a write method.
#  This means it can be passed to a function expecting a file object,
#  and instead of writing to a file the function will write to the string.
#
#  @author John Glover
class StringWriter(object):
    def __init__(self):
        ## @var string The string being written to
        self.string = ""
        
    ## Append text to this object's string
    #  @param s The string to append to this object's string 
    def write(self, s):
        self.string += s
        
        
## ConfigGrabber class
#
# A replacement for the pybrain.rl.environments.ode.tools.configgrab ConfigGrabber
# class, that uses a string instead of a file.
#
# @author John Glover
class ConfigGrabber:
    # @param data The XODE data
    # @param sectionId start looking for parameters only after this string has
    #                  been encountered in the file.
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


## Robot class
#
#  Defines a robot that will be used in the ALife simulation.
#  For now this is just a body (box) with 4 cylindrical legs. 
#  Each leg is attached to the body with a hinge joint.
#
#  The constructor takes a optional 3D position vector, which 
#  the body will be centred on.
#
#  Adds the ability to just return an XML-formatted string containing
#  the XODE file, rather than having to write it to disc then read it.
#   
#  Code is based on the ode.tools.xodetools module from PyBrain
#
#  @author John Glover
class Robot(XODEfile):
    def __init__(self, name, position=[0.0, 150.0, 0.0]):
        XODEfile.__init__(self, name)
        # position of the body
        # position of legs is defined relative to this
        body_position = position
        # mass of the body
        body_mass = 1.0
        # the density of the body
        body_density = 3.0
        # the size of the body
        body_size = [4.0, 3.0, 4.0]
        # radius of the legs
        leg_radius = 0.25
        # length of the legs
        leg_length = 3.8
        # density of the legs
        leg_density = 3.0
        # mass of the legs
        leg_mass = 1.0
        # offset used to calculate leg y-axis coordinate
        # the last term makes the legs recede into the body slightly, looks
        # a bit better
        leg_y_offset = (leg_length/2) + (body_size[1]/2) - min(leg_radius*2, 1.0)
        
        # add body objects
        self.insertBody('robot_body', 'box', body_size, body_density, 
                        pos=body_position, mass=body_mass, 
                        passSet=["robot_body_leg1",
                                 "robot_body_leg2",
                                 "robot_body_leg3",
                                 "robot_body_leg4"])
        self.insertBody('robot_leg1', 'cappedCylinder', [leg_radius, leg_length], 
                        leg_density, euler=[90, 0, 0], mass=leg_mass, 
                        passSet=["robot_body_leg1"],
                        pos=[body_position[0]+1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]-1.2])
        self.insertBody('robot_leg2', 'cappedCylinder', [leg_radius, leg_length],
                        leg_density, euler=[90, 0, 0], mass=leg_mass, 
                        passSet=["robot_body_leg2"],
                        pos=[body_position[0]-1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]-1.2])
        self.insertBody('robot_leg3', 'cappedCylinder', [leg_radius, leg_length], 
                        leg_density, euler=[90, 0, 0], mass=leg_mass, 
                        passSet=["robot_body_leg3"],
                        pos=[body_position[0]+1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]+1.2])
        self.insertBody('robot_leg4', 'cappedCylinder', [leg_radius, leg_length],
                        leg_density, euler=[90, 0, 0], mass=leg_mass, 
                        passSet=["robot_body_leg4"],
                        pos=[body_position[0]-1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]+1.2])
        
        # add joints
        self.insertJoint('robot_body', 'robot_leg1', 'hinge', 
                         anchor=(body_position[0]+1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]-1.2),
                         axis={'x':-1, 'y':0, 'z':0, 'HiStop':1.2, 'LowStop':-1.2})
        self.insertJoint('robot_body', 'robot_leg2', 'hinge', 
                         anchor=(body_position[0]-1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]-1.2),
                         axis={'x':-1, 'y':0, 'z':0, 'HiStop':1.2, 'LowStop':-1.2})
        self.insertJoint('robot_body', 'robot_leg3', 'hinge', 
                         anchor=(body_position[0]+1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]+1.2),
                         axis={'x':-1, 'y':0, 'z':0, 'HiStop':1.2, 'LowStop':-1.2})
        self.insertJoint('robot_body', 'robot_leg4', 'hinge', 
                         anchor=(body_position[0]-1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]+1.2),
                         axis={'x':-1, 'y':0, 'z':0, 'HiStop':1.2, 'LowStop':-1.2})
        
        self.centerOn('robot_body')
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None
        
    ## Get the XODE (xml-formatted) data for this robot as a string
    #  @return XODE (xml-formatted) data for this robot as a string
    def get_xode(self):
        # get main XML string containing body/geometry and joint information
        xml = StringWriter()
        self.write(xml)
        # get custom parameters (passpairs, etc)
        custom = StringWriter()
        self.writeCustomParameters(custom)
        # format xml
        xode = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xode += '<xode version="1.0r23" name="' + self._xodename + '"\n'
        xode += 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' 
        xode += 'xsi:noNamespaceSchemaLocation='
        xode += '"http://tanksoftware.com/xode/1.0r23/xode.xsd">\n\n'
        xode += xml.string
        xode += '</xode>\n'
        xode += custom.string
        return xode


## ALifeEnvironment class
#
#  Handles all Physics simulation. 
#  It loads the robot and asteroid models and puts them into an ODE world. 
#  Code is based on the ode.environment module from PyBrain
#
#  @author John Glover
class ALifeEnvironment(object):
    def __init__(self):
        ## @var root XODE root node, defined in load_xode
        self.root = None
        ## @var world XODE world node, defined in load_xode
        self.world = None 
        ## @var space:  XODE space node, defined in load_xode
        self.space = None  
        ## @var body_geom A list with (body, geom) tuples
        self.body_geom = []
        ## @var asteroid The asteroid geometry
        self.asteroid = None
        ## @var robot_body The robot body object, defined in _parseBodies
        self.robot_body = None
        ## @var textures the textures dictionary
        self.textures = {}
        ## @var sensors sensor list
        self.sensors = []
        ## @var excludesensors list of sensors to exclude
        self.excludesensors = []
        ## @var actuators actuators list
        self.actuators = []
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
        ## @var friction Coulomb friction coefficient for contact joints 
        self.friction = 8.0
        ## @var step_count The current step number
        self.step_count = 0
    
    ## Loads the robot XODE data (xml format) and parses it.
    #  @param data The XODE data for the robot.
    #  @param reload Whether or not to reload sensor data .
    def load_robot(self, data, reload=False):
        p = xode.parser.Parser()
        self.root = p.parseString(data)
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
        self._parseBodies(self.root)

        # now parse the additional parameters at the end of the xode file
        self._loadConfig(data, reload)
        
    ## Loads the asteroid X3D file (XML) and parses it. The resulting
    #  geometry is a xode trimesh object, stored in the variable self.asteroid.
    #  The asteroid is then added to the ALife ODE space.
    #
    #  Note: this file must be called after load_xode, so that the member variable
    #        'space' already exists. An exception is raised if space does not exist.
    #
    #  Other X3D file restraints:
    #  - The file must contain at least 1 object called 'Asteroid'. If not,
    #    an exception is raised
    #  - If more than 1 object called 'Asteroid' exists, only the first one
    #    is processed, the rest are ignored.
    #  - The file must consist of faces stored as triangles. If the file 
    #    cannot be read or it does not contain any triangles an exception
    #    is raised.
    #     
    #  @param file_name The file path to the .x3d file for the asteroid.
    def load_asteroid(self, file_name):
        # check for existing ODE space
        if not self.space:
            # todo: more detail on this exception
            raise Exception("NoSpace")
        dom = md.parse(file_name)
        root = dom.createElement('xode')
        root.attributes['version'] = '1.0r23'
        root.attributes['name'] = 'alife'
        root.attributes['xmlns:xsi'] = 'http://www.w3.org/2001/XMLSchema-instance'
        root.attributes['xsi:noNamespaceSchemaLocation'] = 'http://tanksoftware.com/xode/1.0r23/xode.xsd'
        world = dom.createElement('world')
        root.appendChild(world)
        space = dom.createElement('space')
        world.appendChild(space)
        
        geom = dom.createElement('geom')
        geom.attributes['name'] = 'Asteroid'
        trimesh = dom.createElement('trimesh')
        geom.appendChild(trimesh)
        trimesh_triangles = dom.createElement('triangles')
        trimesh.appendChild(trimesh_triangles)
        trimesh_vertices = dom.createElement('vertices')
        trimesh.appendChild(trimesh_vertices)
        space.appendChild(geom)
        
        for node in dom.getElementsByTagName('Transform'):
            if 'DEF' in node.attributes.keys():
                # take the first transform node defined as 'Asteroid'
                if node.attributes['DEF'].value == 'Asteroid':
                    # get scale information from the model
                    if 'scale' in node.attributes.keys():
                        scale_string = node.attributes['scale'].value
                        scale = scale_string.split()
                        scale = [float(s) for s in scale]
                    else:
                        scale = (1, 1, 1)
                        
                    # todo: get translation information from the model
                    # todo: get rotation information from the model
                    
                    if node.getElementsByTagName('IndexedFaceSet'):
                        ifs = node.getElementsByTagName('IndexedFaceSet')[0]
                        # form triangles from the coordIndex
                        coord_index = ifs.attributes['coordIndex'].value
                        for face in coord_index.split(','):
                            # make sure that the given faces are triangles
                            # there should 4 indicies, the last one equal to -1
                            indicies = face.split()
                            if len(indicies) == 4 and int(indicies[3]) == -1:
                                # x3d indices count from zero but xode indices
                                # count form one, so add one to each index value
                                t = dom.createElement('t')
                                t.attributes['ia'] = str(int(indicies[0])+1)
                                t.attributes['ib'] = str(int(indicies[1])+1)
                                t.attributes['ic'] = str(int(indicies[2])+1)
                                trimesh_triangles.appendChild(t)
                        # form vertices from the Coordinate point attribute
                        coordinate = ifs.getElementsByTagName('Coordinate')[0]
                        coord_points = coordinate.attributes['point'].value
                        for points in coord_points.split(','):
                            # each vertex should have 3 points
                            points = points.split()
                            if len(points) == 3:
                                v = dom.createElement('v')
                                v.attributes['x'] = str(float(points[0]) * scale[0])
                                v.attributes['y'] = str(float(points[1]) * scale[1])
                                v.attributes['z'] = str(float(points[2]) * scale[2])
                                trimesh_vertices.appendChild(v)
                        break
    
        # parse, adding to environment
        parser = xode.parser.Parser()
        self._parseBodies(parser.parseString(root.toxml()))
        
        # check that asteroid geometry was created successfully
        # todo: check triangle count: self.asteroid.getTriangleCount()
        # todo: give more detail in exception
        if not self.asteroid:
            raise Exception("NoAsteroid")
        # add asteroid to current space
        self.asteroid.getSpace().remove(self.asteroid)
        self.space.add(self.asteroid)
    
    ## Load the XODE config.
    #  @param data The XODE data string
    #  @param reload Whether or not to reload sensor data.  
    def _loadConfig(self, data, reload=False):
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
            # ('name', (0.3,0.4,0.5))
            objname, coldef = eval(coldefstring)
            for (body, _) in self.body_geom:
                if hasattr(body, 'name'):
                    if objname == body.name:
                        body.color = coldef
                        break
                
        if not reload:
            # add the JointSensor as default
            self.sensors = [] 
            ## self.addSensor(self._jointSensor)
            
            # <sensors>
            # expects a list of strings, each of which is the executable command to create a sensor object
            # example: DistToPointSensor('legSensor', (0.0, 0.0, 5.0))
            sens = self.config.getValue("sensors")[:]
            for s in sens:
                try:
                    self.addSensor(eval('sensors.' + s))
                except AttributeError:
                    print dir(sensors)
                    warnings.warn("Sensor name with name " + s + " not found. skipped.")
        else:
            for s in self.sensors:
                s._connect(self)
            for a in self.actuators:
                a._connect(self)

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
            # todo: do we need to call geom.setBody(body)?
            #       http://pyode.sourceforge.net/tutorials/tutorial3.html
            geom.setBody(body)
        
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
                    self.asteroid = geom
        
        # special cases for joints: universal, fixed, amotor
        elif isinstance(node, xode.joint.Joint):
            joint = node.getODEObject()
            
            if type(joint) == ode.UniversalJoint:
                # insert an additional AMotor joint to read the angles from and to add torques
                # amotor = ode.AMotor(self.world)
                # amotor.attach(joint.getBody(0), joint.getBody(1))
                # amotor.setNumAxes(3)
                # amotor.setAxis(0, 0, joint.getAxis2())
                # amotor.setAxis(2, 0, joint.getAxis1())
                # amotor.setMode(ode.AMotorEuler)
                # xode_amotor = xode.joint.Joint(node.getName() + '[amotor]', node.getParent())
                # xode_amotor.setODEObject(amotor)
                # node.getParent().addChild(xode_amotor, None)
                pass
            if type(joint) == ode.AMotor:
                # do the euler angle calculations automatically (ref. ode manual)
                joint.setMode(ode.AMotorEuler)
                
            if type(joint) == ode.FixedJoint:
                # prevent fixed joints from bouncing to center of first body
                joint.setFixed()

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
            # parameters from Niko Wolf
            c.setBounce(0.2)
            c.setBounceVel(0.05) #Set the minimum incoming velocity necessary for bounce
            c.setSoftERP(0.6) #Set the contact normal "softness" parameter
            c.setSoftCFM(0.00005) #Set the contact normal "softness" parameter
            c.setSlip1(0.02) #Set the coefficient of force-dependent-slip (FDS) for friction direction 1
            c.setSlip2(0.02) #Set the coefficient of force-dependent-slip (FDS) for friction direction 2
            c.setMu(self.friction) #Set the Coulomb friction coefficient
            j = ode.ContactJoint(world, contactgroup, c)
            j.name = None
            j.attach(geom1.getBody(), geom2.getBody())
            
    ## Get the distance between the given body and the centre of the asteroid
    #  @param body the body
    #  @return distance between the given body and the centre of the asteroid (float)
    def _distance_to_asteroid(self, body):
        b = body.getPosition()
        if not self.asteroid.getPosition() == (0, 0, 0):
            # if the asteroid is not at the origin, change the body position, keeping
            # it the same relative to the asteroid, but with the asteroid centred at
            # the origin
            a = self.asteroid.getPosition()
            b = (b[0]+a[0], b[1]+a[1], b[2]+a[2])
        # get the length of the body vector
        return np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    
    ## Get the direction from the given body to the centre of the asteroid,
    #  represented as a unit vector
    #  @param body the body
    #  @return direction from the given body to the centre of the asteroid,
    #          represented as a unit vector (float x, float y, float z)
    def _direction_of_asteroid(self, body):
        a = self.asteroid.getPosition()
        b = body.getPosition()
        # update the asteroid position, keeping its position the same relative to
        # the body, but with the body centred at the origin
        a = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
        # return a unit vector in the direction of the asteroid
        length = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
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
            
    ## Calculate the next step in the ODE environment.
    #  @param dt The step size. 
    def step(self, dt=0.04):
        """ Here the ode physics is calculated by one step. """
        # Detect collisions and create contact joints
        self.space.collide((self.world, self.contactgroup), self._near_callback)
        
        # update gravity
        for (body, geom) in self.body_geom:
            if body:
                # calculate the distance to the centre of the asteroid
                distance = self._distance_to_asteroid(body)
                # get the direction of the force
                direction = self._direction_of_asteroid(body)
                # calculate the force of gravity, based on the 
                # this distance and the masses of the asteroid and body
                m1 = self.asteroid_mass
                m2 = body.getMass().mass
                f = self.grav_constant * m1 * m2 / distance**2
                force = (f*direction[0], f*direction[1], f*direction[2])
                # apply this force to the body
                body.addForce(force)
        
        # Simulation step
        self.world.step(dt)
        
        # Remove all contact joints
        self.contactgroup.empty()
            
        # update all sensors
        for s in self.sensors:
            s._update()
        
        # increase step counter
        self.step_count += 1
        return self.step_count