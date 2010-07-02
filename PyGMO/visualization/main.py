#!/usr/bin/env python
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

# Version information
__version__ = "0.1"


import traj3d


"""
Class:
"""
class TrajectoryVisualizer:

   def __init__( self, data ):
      """
      """
      self.__axes = None
      self.__origin = None
      self.__data = data

      # Create opengl context
      self.engine = traj3d.traj3d( "3D Trajectory Test", 800, 600 )
      self.engine.add( traj3d.Trajectory( data ) )

   def start( self ):
      self.engine.start()

   def axes( self, enable ):
      if enable and self.__axes is None:
         #self.__axes = traj3d.Axes()
         self.engine.add( self.__axes )
      elif not enable and self.__axes is not None:
         self.engine.remove( self.__axes )
         self.__axes = None

   def origin( self, enable ):
      if enable and self.__origin is None:
         self.__origin = traj3d.Origin()
         self.engine.add( self.__origin )
      elif not enable and self.__origin is not None:
         self.engine.remove( self.__origin )
         self.__origin = None

# Run some tests
if __name__ == "__main__":
   data = ( 0.,
               135299631.153314,
                  -67433330.1118738,
                  0.,
               13.3117666043686,
                  30.2608967735421,
                  -1.21905509181638,
               0.,
                  0.,
                  0.,
            350.0,
               -242539545.423682,
                  -36777066.1865885,
                  5209166.99174208,
               5.46701901228146,
                  -19.7530014959265,
                  0.562626144359047,
               0.,
                  0.,
                  0. )

   # Create the engine
   traj = TrajectoryVisualizer( data )

   # Create some stuff
   traj.origin( True )
   #traj.axes( True )


   # Start the engine
   traj.start()






