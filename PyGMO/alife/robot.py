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
class Robot(object):
    ## Constructor
    #  @param name The string containing the name of the robot
    #  @param position A 3-tuple giving the initial position of the
    #                  robot body
    def __init__(self, world, space, body_position=[0.0, 150.0, 0.0], name=""):
        self.name = name
        # position of the body
        # position of legs is defined relative to this
        # mass of the body
        body_mass = 1.0
        # the density of the body
        body_density = 1.0
        # the size of the body
        body_size = [4.0, 3.0, 4.0]
        # radius of the legs
        leg_radius = 0.25
        # length of the legs
        leg_length = 3.8
        # density of the legs
        leg_density = 1.0
        # mass of the legs
        leg_mass = 1.0
        # rotation of the legs
        leg_rotation = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
        # offset used to calculate leg y-axis coordinate
        # the last term makes the legs recede into the body slightly, looks
        # a bit better
        leg_y_offset = (leg_length/2) + (body_size[1]/2) - min(leg_radius*2, 1.0)
        
        self.bodies_geoms = []
        self.joints = []
        self.passpairs = [('robot_body', 'robot_leg1'),
                          ('robot_body', 'robot_leg2'),
                          ('robot_body', 'robot_leg3'),
                          ('robot_body', 'robot_leg4')]
        self.center_obj = "robot_body"
        
        # Body
        self.body = ode.Body(world)
        self.body.name = "robot_body"
        self.body.setPosition(body_position)
        body_mass = ode.Mass()
        body_mass.setBox(body_density, body_size[0], body_size[1], body_size[2])
        self.body.setMass(body_mass)
        body_geom = ode.GeomBox(space, lengths=body_size)
        body_geom.name = "robot_body"
        body_geom.setBody(self.body)
        self.bodies_geoms.append((self.body, body_geom))
        
        # Leg 1
        leg1 = ode.Body(world)
        leg1.name = "robot_leg1"
        leg1.setPosition((body_position[0]+1.2,
                          body_position[1]-leg_y_offset,
                          body_position[2]-1.2))
        leg1_mass = ode.Mass()
        leg1_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg1.setMass(leg1_mass)
        leg1_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg1_geom.name = "robot_leg1"
        leg1_geom.setBody(leg1)
        self.bodies_geoms.append((leg1, leg1_geom))
        leg1.setRotation(leg_rotation)
        
        # Leg 2
        leg2 = ode.Body(world)
        leg2.name = "robot_leg2"
        leg2.setPosition((body_position[0]-1.2,
                          body_position[1]-leg_y_offset,
                          body_position[2]-1.2))
        leg2_mass = ode.Mass()
        leg2_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg2.setMass(leg2_mass)
        leg2_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg2_geom.name = "robot_leg2"
        leg2_geom.setBody(leg2)
        self.bodies_geoms.append((leg2, leg2_geom))
        leg2.setRotation(leg_rotation)

        
        # Leg 3
        leg3 = ode.Body(world)
        leg3.name = "robot_leg3"
        leg3.setPosition((body_position[0]+1.2,
                          body_position[1]-leg_y_offset,
                          body_position[2]+1.2))
        leg3_mass = ode.Mass()
        leg3_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg3.setMass(leg3_mass)
        leg3_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg3_geom.name = "robot_leg3"
        leg3_geom.setBody(leg3)
        self.bodies_geoms.append((leg3, leg3_geom))
        leg3.setRotation(leg_rotation)
        
        # Leg 4
        leg4 = ode.Body(world)
        leg4.name = "robot_leg4"
        leg4.setPosition((body_position[0]-1.2,
                          body_position[1]-leg_y_offset,
                          body_position[2]+1.2))
        leg4_mass = ode.Mass()
        leg4_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg4.setMass(leg4_mass)
        leg4_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg4_geom.name = "robot_leg4"
        leg4_geom.setBody(leg4)
        self.bodies_geoms.append((leg4, leg4_geom))
        leg4.setRotation(leg_rotation)
        
        # Joint 1
        joint1 = ode.HingeJoint(world)
        joint1.name="robot_body_leg1"
        joint1.attach(self.body, leg1)
        joint1.setAnchor((body_position[0]+1.2, 
                          body_position[1]-(body_size[1]/2), 
                          body_position[2]-1.2))
        joint1.setAxis((1,0,0))
        joint1.setParam(ode.ParamLoStop, -1.2)
        joint1.setParam(ode.ParamHiStop, 1.2)
        joint1.setParam(ode.ParamFMax, 10)
        self.joints.append(joint1)
        
        # Joint 2
        joint2 = ode.HingeJoint(world)
        joint2.name="robot_body_leg2"
        joint2.attach(self.body, leg2)
        joint2.setAnchor((body_position[0]-1.2, 
                          body_position[1]-(body_size[1]/2), 
                          body_position[2]-1.2))
        joint2.setAxis((1,0,0))
        joint2.setParam(ode.ParamLoStop, -1.2)
        joint2.setParam(ode.ParamHiStop, 1.2)
        joint2.setParam(ode.ParamFMax, 10)
        self.joints.append(joint2)
        
        # Joint 3
        joint3 = ode.HingeJoint(world)
        joint3.name="robot_body_leg3"
        joint3.attach(self.body, leg3)
        joint3.setAnchor((body_position[0]+1.2, 
                          body_position[1]-(body_size[1]/2), 
                          body_position[2]+1.2))
        joint3.setAxis((1,0,0))
        joint3.setParam(ode.ParamLoStop, -1.2)
        joint3.setParam(ode.ParamHiStop, 1.2)
        joint3.setParam(ode.ParamFMax, 10)
        self.joints.append(joint3)
        
        # Joint 4
        joint4 = ode.HingeJoint(world)
        joint4.name="robot_body_leg4"
        joint4.attach(self.body, leg4)
        joint4.setAnchor((body_position[0]-1.2, 
                          body_position[1]-(body_size[1]/2), 
                          body_position[2]+1.2))
        joint4.setAxis((1,0,0))
        joint4.setParam(ode.ParamLoStop, -1.2)
        joint4.setParam(ode.ParamHiStop, 1.2)
        joint4.setParam(ode.ParamFMax, 10)
        self.joints.append(joint4)
