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

from pybrain.rl.environments.ode.tools.xodetools import XODEfile

class Robot(XODEfile):
    def __init__(self, name, **kwargs):
        XODEfile.__init__(self, name, **kwargs)
        # position of the body
        # position of legs is defined relative to this
        body_position = [0.0, 150, 0.0]
        # mass of the body
        body_mass = 100.0
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
        leg_y_offset = (leg_length/2) + (body_size[1]/2) 
        
        # add body objects
        self.insertBody('robot_body', 'box', body_size, body_density, 
                        pos=body_position, mass=body_mass)
        self.insertBody('robot_leg1', 'cappedCylinder', [leg_radius, leg_length], 
                        leg_density, euler=[90, 0, 0], mass=leg_mass,
                        pos=[body_position[0]+1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]-1.2])
        self.insertBody('robot_leg2', 'cappedCylinder', [leg_radius, leg_length],
                        leg_density, euler=[90, 0, 0], mass=leg_mass,
                        pos=[body_position[0]-1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]-1.2])
        self.insertBody('robot_leg3', 'cappedCylinder', [leg_radius, leg_length], 
                        leg_density, euler=[90, 0, 0], mass=leg_mass,
                        pos=[body_position[0]+1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]+1.2])
        self.insertBody('robot_leg4', 'cappedCylinder', [leg_radius, leg_length],
                        leg_density, euler=[90, 0, 0], mass=leg_mass,
                        pos=[body_position[0]-1.2, 
                             body_position[1]-leg_y_offset, 
                             body_position[2]+1.2])
        
        # add joints
#        self.insertJoint('robot_body', 'robot_leg1', 'ball', 
#                         anchor=(body_position[0]+1.2, 
#                                 body_position[1]-(body_size[1]/2), 
#                                 body_position[2]-1.2))
#        self.insertJoint('robot_body', 'robot_leg2', 'ball', 
#                         anchor=(body_position[0]-1.2, 
#                                 body_position[1]-(body_size[1]/2), 
#                                 body_position[2]-1.2))
#        self.insertJoint('robot_body', 'robot_leg3', 'ball', 
#                         anchor=(body_position[0]+1.2, 
#                                 body_position[1]-(body_size[1]/2), 
#                                 body_position[2]+1.2))
#        self.insertJoint('robot_body', 'robot_leg4', 'ball', 
#                         anchor=(body_position[0]-1.2, 
#                                 body_position[1]-(body_size[1]/2), 
#                                 body_position[2]+1.2))
        self.insertJoint('robot_body', 'robot_leg1', 'fixed', 
                         anchor=(body_position[0]+1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]-1.2))
        self.insertJoint('robot_body', 'robot_leg2', 'fixed', 
                         anchor=(body_position[0]-1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]-1.2))
        self.insertJoint('robot_body', 'robot_leg3', 'fixed', 
                         anchor=(body_position[0]+1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]+1.2))
        self.insertJoint('robot_body', 'robot_leg4', 'fixed', 
                         anchor=(body_position[0]-1.2, 
                                 body_position[1]-(body_size[1]/2), 
                                 body_position[2]+1.2))
        
        self.centerOn('robot_body')
#        self.insertFloor(y= -12.7)
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None
        
      
if __name__ == "__main__":
    alife = Robot('robot')
    alife.writeXODE()