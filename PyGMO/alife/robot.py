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
from pybrain.rl.environments.ode import sensors, actuators
import ode
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
#  @author John Glover
class Robot(object):
    ## Constructor
    #  @param world The ODE world that bodies are added to
    #  @param space The ODE space that geometries are added to
    #  @param body_position A 3-tuple giving the initial position of the
    #                       robot body
    #  @param name The string containing the name of the robot
    def __init__(self, world, space, body_position=[0.0, 150.0, 0.0], name=""):
        ##
        self.bodies_geoms = []
        ##
        self.joints = []
        ##
        self.name = name
        ## The density of the body
        self._body_density = 0.35
        # the size of the body
        body_size = [4.0, 3.0, 4.0]
        # radius of the legs
        leg_radius = 0.25
        # length of the legs
        leg_length = 3.8
        # density of the legs
        leg_density = 0.25
        ## Offset used to calculate leg y-axis coordinate.
        #  The last term makes the legs recede into the body slightly, looks
        #  a bit better
        self._leg_y_offset = (leg_length/2) + (body_size[1]/2) - min(leg_radius*2, 1.0)
        # The rotation of the legs
        self._leg_rotation = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
        ##
        self.passpairs = [('robot_body', 'robot_leg1'),
                          ('robot_body', 'robot_leg2'),
                          ('robot_body', 'robot_leg3'),
                          ('robot_body', 'robot_leg4')]
        ##
        self.center_obj = "robot_body"
        
        # Body
        self.body = ode.Body(world)
        self.body.name = "robot_body"
        self.body.setPosition(body_position)
        self.body.initial_position = body_position
        body_mass = ode.Mass()
        body_mass.setBox(self._body_density, body_size[0], body_size[1], body_size[2])
        self.body.setMass(body_mass)
        body_geom = ode.GeomBox(space, lengths=body_size)
        body_geom.name = "robot_body"
        body_geom.setBody(self.body)
        self.body.initial_rotation = self.body.getRotation()
        self.bodies_geoms.append((self.body, body_geom))
        
        # Leg 1
        leg1 = ode.Body(world)
        leg1.name = "robot_leg1"
        leg1_position = (body_position[0]+1.2,
                         body_position[1]-self._leg_y_offset,
                         body_position[2]-1.2)
        leg1.setPosition(leg1_position)
        leg1.initial_position = leg1_position
        leg1_mass = ode.Mass()
        leg1_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg1.setMass(leg1_mass)
        leg1_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg1_geom.name = "robot_leg1"
        leg1_geom.setBody(leg1)
        leg1.setRotation(self._leg_rotation)
        leg1.initial_rotation = self._leg_rotation
        self.bodies_geoms.append((leg1, leg1_geom))
        
        # Leg 2
        leg2 = ode.Body(world)
        leg2.name = "robot_leg2"
        leg2_position = (body_position[0]-1.2,
                         body_position[1]-self._leg_y_offset,
                         body_position[2]-1.2)
        leg2.setPosition(leg2_position)
        leg2.initial_position = leg2_position
        leg2_mass = ode.Mass()
        leg2_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg2.setMass(leg2_mass)
        leg2_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg2_geom.name = "robot_leg2"
        leg2_geom.setBody(leg2)
        leg2.setRotation(self._leg_rotation)
        leg2.initial_rotation = self._leg_rotation
        self.bodies_geoms.append((leg2, leg2_geom))
        
        # Leg 3
        leg3 = ode.Body(world)
        leg3.name = "robot_leg3"
        leg3_position = (body_position[0]+1.2,
                         body_position[1]-self._leg_y_offset,
                         body_position[2]+1.2)
        leg3.setPosition(leg3_position)
        leg3.initial_position = leg3_position
        leg3_mass = ode.Mass()
        leg3_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg3.setMass(leg3_mass)
        leg3_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg3_geom.name = "robot_leg3"
        leg3_geom.setBody(leg3)
        leg3.setRotation(self._leg_rotation)
        leg3.initial_rotation = self._leg_rotation
        self.bodies_geoms.append((leg3, leg3_geom))
        
        # Leg 4
        leg4 = ode.Body(world)
        leg4.name = "robot_leg4"
        leg4_position = (body_position[0]-1.2,
                         body_position[1]-self._leg_y_offset,
                         body_position[2]+1.2)
        leg4.setPosition(leg4_position)
        leg4.initial_position = leg4_position
        leg4_mass = ode.Mass()
        leg4_mass.setCappedCylinder(leg_density, 3, leg_radius, leg_length)
        leg4.setMass(leg4_mass)
        leg4_geom = ode.GeomCCylinder(space, length=leg_length, radius=leg_radius)
        leg4_geom.name = "robot_leg4"
        leg4_geom.setBody(leg4)
        leg4.setRotation(self._leg_rotation)
        leg4.initial_rotation = self._leg_rotation
        self.bodies_geoms.append((leg4, leg4_geom))
        
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
        
    def get_position(self):
        return self.body.getPosition()    
        
    def get_joints(self):
        return self.joints

    def reset(self):
        for body, geom in self.bodies_geoms:
            body.setPosition(body.initial_position)
            body.setRotation(body.initial_rotation)
            body.setLinearVel((0, 0, 0))
            body.setAngularVel((0, 0, 0))
        for joint in self.joints:
            joint.setParam(ode.ParamVel, 0)
