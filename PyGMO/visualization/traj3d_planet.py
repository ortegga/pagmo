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


# General imports
import math

# General OpenGL imports
try:
   from OpenGL.GL   import *
   from OpenGL.GLU  import *
   from OpenGL.GLUT import *
   from OpenGL.arrays import ArrayDatatype as ADT
except ImportError:
   print( "Warning: The python-opengl bindings are missing, you won't be able to use the visualization module." )
   raise ImportError

# FTGL
try:
   import FTGL
except ImportError:
   print( "Warning: The python-ftgl bindings are missing, you won't be able to use the visualization module." )
   raise ImportError

# NumPy
try:
   from numpy import *
except ImportError:
   print( "Warning: The numpy bindings are missing, you won't be able to use the visualization module." )

# Local imports
from frange import *
from traj3d_path import *

# Misc PaGMO imports
from PyGMO import keplerian_toolbox, astro_toolbox


###############################################################################
class Planet(Object):
   """
   Represents a planet.
   """

   def __init__( self, mjd2000, planet, info, mu=1.32712428e20 ):
      Object.__init__( self )
      mjd2000  = [ mjd2000, mjd2000 + info['period'] ]
      self.__mjd2000 = mjd2000
      self.__planet  = planet
      self.__info    = info
      self.__mu      = mu

      # Generate data
      start    = mjd2000[0]
      data     = []
      for mjd in mjd2000:
         r, v = astro_toolbox.Planet_Ephemerides_Analytical( mjd, planet )
         data.extend( [ mjd-start, r[0], r[1], r[2], v[0], v[1], v[2], 0., 0., 0. ] )

      # Create path
      self.__path    = Path( data, 24.*3600., 1000., 1000., 1., mu, info['colour'] )

   def showVectors( self, enable ):
      self.__path.showVectors( enable )

   def setScale( self, zoom ):
      "Gives an indication of the current scale size."
      self.__path.setScale( zoom )

   def center( self ):
      "Gets the center of the object."
      return self.__path.center()

   def size( self ):
      "Gets the size of the object."
      return self.__path.size()
  
   def setPosition( self, t ):
      "Sets the position."
      self.__path.setPosition( t )

   def position( self, t ):
      "Gets the position and velocity vectors of the trajectory at a given instant."
      return self.__path.position( t )

   def display( self ):
      "Displays the trajectory."
      self.__path.display()



