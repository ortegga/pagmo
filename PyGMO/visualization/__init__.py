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

import csv


class Trajectory3D:
   """
   Class that allows easy 3D visualization of interplanetary trajectories.
   """

   def __init__( self, data, width=800, height=600, conv_t=1., conv_r=1., conv_v=1., conv_dv=1. ):
      """
      Constructor for the Trajectory3D.
      """
      # Handle data
      if type(data).__name__ == 'str':
         data_csv = csv.reader( open(data, 'r') )
         data     = []
         for row in data_csv:
            data.extend( row )
         data     = map( lambda x: float(x), data )
         data_csv = None

      # Defaults
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

   def setUnits( self, distance, time, velocity ):
      """
      Sets the units to display.
      """
      self.traj.setUnits( distance, time, velocity )

   def start( self ):
      """
      Starts the engine.

      When you start the engine it will block until the window is closed.
      """
      self.engine.start()

   def resize( self, width=800, height=600 ):
      """
      Resizes the window.

      Allows you to define the size of the window in pixels.
      """
      self.engine.reshape( width, height )

   def __planetFromName( self, name ):
      if "mercury" == name.lower():
         return { 'colour' : (1.0,0.8,0.0), 'period' : 87.96, 'num' : 1, 'name' : 'Mercury' }
      elif "venus" == name.lower():
         return { 'colour' : (0.1,0.1,0.9), 'period' : 224.68, 'num' : 2, 'name' : 'Venus' }
      elif "earth" == name.lower():
         return { 'colour' : (0.1,0.9,0.1), 'period' : 365.26, 'num' : 3, 'name' : 'Earth' }
      elif "mars" == name.lower():
         return { 'colour' : (0.9,0.1,0.1), 'period' : 686.98, 'num' : 4, 'name' : 'Mars' }
      elif "jupiter" == name.lower():
         return { 'colour' : (1.0,0.5,0.1), 'period' : 11.862*365.26, 'num' : 5, 'name' : 'Jupiter' }
      elif "saturn" == name.lower():
         return { 'colour' : (1.0,0.8,0.6), 'period' : 29.456*365.26, 'num' : 6, 'name' : 'Saturn' }
      elif "uranus" == name.lower():
         return { 'colour' : (0.0,0.8,0.6), 'period' : 84.07*365.26, 'num' : 7, 'name' : 'Uranus' }
      elif "neptune" == name.lower():
         return { 'colour' : (0.3,0.3,0.6), 'period' : 164.81*365.26, 'num' : 8, 'name' : 'Neptune' }

   def addPlanets( self, planets_data=None ):
      """
      Adds planets.

      The mjd2000 should be the take off date for the mission.

      Planets should be a list of planet names like:
       [ 'mercury', 'earth' ]

      Alternatively if you pass a filename as mjd2000 it will load the csv
       file and use that.
      """
      if type(planets_data).__name__ == 'str':
         data        = planets_data
         data_csv    = csv.reader( open(data, 'r') )
         planets     = {}
         for row in data_csv:

            # Extract data
            name  = row[0]
            date  = dateutil.parser.parse( row[1] )
            mjd2000 = convert_date( date.year, date.month, date.day )
            dv    = float(row[2])

            # Process data
            p     = self.__planetFromName( name )
            num   = p['num']
            if planets.has_key(num):
               planets[num]['flyby'].append( { 'mjd2000' : mjd2000*24.*3600., 'date' : row[1], 'dv' : dv } )
            else:
               p['flyby']     = [ { 'mjd2000' : mjd2000*24.*3600., 'date' : row[1], 'dv' : dv } ]
               planets[num]   = p
         
         # Final touches
         data_csv    = None

      else:
         planets     = {}
         for planet in planets_data:
            p  = self.__planetFromName( planet )
            planets[ p['num'] ] = p

      self.traj.addPlanets( planets )

   def vectors( self, enable ):
      """
      Enables visualization of vectors.

      These vectors are represent position from origin and velocity.
      """
      self.traj.showVectors( enable )

   def controls( self, enable ):
      """
      Hides or shows the playback controls.
      """
      self.traj.controls( enable )

   def axes( self, enable ):
      """
      Shows axes.

      These axes help calculate distances visually.
      """
      self.traj.axes( enable )

   def origin( self, enable ):
      """
      Sets visual indicator of the origin.

      Displays a large line at each of the axes.
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
      if pressed:
         if key is 'p':
            self.traj.pause( not self.traj.ispaused() )
            self.__playing = not self.__playing
         elif key is 'r':
            self.traj.restart()
         elif key is 'f':
            self.traj.faster()
         elif key is 's':
            self.traj.slower()
         elif key is 'h':
            self.traj.controls()

   def duration( self, duration=30. ):
      """
      Sets the duration of the animation.

      The duration is in seconds and represents how long the total animation should last.
      """
      self.traj.duration = duration

   def repeat( self, enable ):
      """
      Sets animation repetition.

      This just makes the animation loop all the time.
      """
      self.traj.repeat( enable )


def convert_date( Y, M, D, HR=0., MIN=0., SEC=0. ):
   JDN      = (1461. * (Y + 4800. + (M - 14.)/12.))/4. +(367. * (M - 2. - 12. * ((M - 14.)/12.)))/12. - (3. * ((Y + 4900. + (M - 14.)/12.)/100.))/4. + D - 32075.
   JD       = JDN + (HR-12.)/24. + MIN/1440. + SEC/86400.
   MJD      = JD - 2400000.5
   MJD2000  = MJD - 51544.5
   return MJD2000


import dateutil.parser

# Run some tests
if __name__ == "__main__":

   # Create the engine
   traj = Trajectory3D( "EarthMarsDSM.txt", 640, 480,
         24.*3600., 1000., 1000., 1000. ) # Unit conversions: days->s, km->m
   traj.addPlanets( "EarthMarsDSM_flybyinfo.txt" )
   traj.setUnits( "km", "d", "km/s" )

   # Create some stuff
   traj.origin( True )
   traj.vectors( True )
   traj.repeat( True )
   traj.axes( True )

   # Start the engine
   traj.start()






