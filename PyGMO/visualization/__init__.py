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
import traj3d_traj
import traj3d_planet


"""
Class:
"""
class Trajectory3D:

   def __init__( self, data, width=800, height=600, conv_t=1., conv_r=1., conv_v=1., conv_dv=1. ):
      """
      Constructor for the TrajectoryVisualizer
      """
      self.__axes    = None
      self.__origin  = None
      self.__data    = data
      self.__playing = True

      # Create opengl context
      self.engine = traj3d.traj3d( "3D Trajectory Test", width, height )
      self.traj   = traj3d_traj.Trajectory( data, conv_t, conv_r, conv_v, conv_dv )
      self.engine.add( self.traj )

      # Bind keyboard input
      self.engine.inputKeyboard( self.__keyboard )

   def start( self ):
      """
      Starts the engine.
      """
      self.engine.start()

   def resize( self, width, height ):
      """
      Resizes the window.
      """
      self.engine.reshape( width, height )

   def addPlanets( self, mjd2000, planets ):
      """
      Adds planets.
      """
      self.traj.addPlanets( mjd2000, planets )

   def vectors( self, enable ):
      self.traj.showVectors( enable )

   def axes( self, enable ):
      """
      Shows axes.
      """
      self.traj.axes( enable )

   def origin( self, enable ):
      """
      Sets visual indicator of the origin.
      """
      if enable and self.__origin is None:
         self.__origin = traj3d_traj.Origin( self.traj.size() )
         self.engine.add( self.__origin )
      elif not enable and self.__origin is not None:
         self.engine.remove( self.__origin )
         self.__origin = None

   def __keyboard( self, pressed, key, x, y ):
      """
      Handles keyboard input.
      """
      if key is 'p' and pressed:
         self.traj.pause( not self.traj.ispaused() )
         self.__playing = not self.__playing
      if key is 'r' and pressed:
         self.traj.restart()
      if key is 'f' and pressed:
         self.traj.faster()
      if key is 's' and pressed:
         self.traj.slower()

   def duration( self, duration ):
      "Sets the duration of the animation."
      self.traj.duration = duration

   def repeat( self, enable ):
      "Sets animation repetition."
      self.traj.repeat( enable )

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
   mjd2000 = (8273.26728762578, 8623.26728762578)
   planets = { 3 : (1.,0.,0.), 4: (0.,1.,0.) }

   # Create the engine
   traj = Trajectory3D( data, 640, 480,
         24.*3600., 1000., 1000., 1. ) # Unit conversions: days->s, km->m
   traj.addPlanets( mjd2000, planets )

   # Create some stuff
   traj.origin( True )
   traj.vectors( True )
   traj.repeat( True )
   traj.axes( True )

   # Start the engine
   traj.start()






