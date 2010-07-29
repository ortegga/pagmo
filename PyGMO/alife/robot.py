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
  
## @package robot
#  This module contains the ALife Robot classes. 
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
    ## Constructor
    def __init__(self):
        ## @var string The string being written to
        self.string = ""
        
    ## Append text to this object's string
    #  @param s The string to append to this object's string 
    def write(self, s):
        self.string += s
        

## XODEObject class
#
#  Extends XODEfile objects, adding the ability to return the XODE
#  data as a string so that it does not need to be written to disc.
#
#  @author John Glover
class XODEObject(XODEfile):
    ## Constructor
    def __init__(self, name):
        XODEfile.__init__(self, name)
        
    ## Get the XODE (xml-formatted) data for this object as a string
    #  @return XODE (xml-formatted) data for this object as a string
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
class Robot(XODEObject):
    ## Constructor
    #  @param name The string containing the name of the robot
    #  @param position A 3-tuple giving the initial position of the
    #                  robot body
    def __init__(self, name, position=[0.0, 150.0, 0.0]):
        XODEObject.__init__(self, name)
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
                         name="robot_body_leg1",
                         anchor=(body_position[0]+1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]-1.2),
                         axis={'x':-1, 'y':0, 'z':0,
                               'HiStop':1.2, 'LowStop':-1.2, 'FMax': 10.0})
        self.insertJoint('robot_body', 'robot_leg2', 'hinge', 
                         name="robot_body_leg2",
                         anchor=(body_position[0]-1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]-1.2),
                         axis={'x':-1, 'y':0, 'z':0,
                               'HiStop':1.2, 'LowStop':-1.2, 'FMax': 10.0})
        self.insertJoint('robot_body', 'robot_leg3', 'hinge',
                         name="robot_body_leg3", 
                         anchor=(body_position[0]+1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]+1.2),
                         axis={'x':-1, 'y':0, 'z':0,
                               'HiStop':1.2, 'LowStop':-1.2, 'FMax': 10.0})
        self.insertJoint('robot_body', 'robot_leg4', 'hinge',
                         name="robot_body_leg4", 
                         anchor=(body_position[0]-1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]+1.2),
                         axis={'x':-1, 'y':0, 'z':0,
                               'HiStop':1.2, 'LowStop':-1.2, 'FMax': 10.0})
        
        self.centerOn('robot_body')
