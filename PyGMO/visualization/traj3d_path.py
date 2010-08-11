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

# NumPy
try:
   from numpy import *
except ImportError:
   print( "Warning: The numpy bindings are missing, you won't be able to use the visualization module." )

# Local imports
from frange import *
from traj3d_object import *

# Misc PaGMO imports
from PyGMO import keplerian_toolbox


###############################################################################
class Path(Object):
   """
   Represents a 3D path.
   """

   def __init__( self, data, conv_t=1., conv_r=1., conv_v=1., conv_dv=1., mu = 1.32712428e20, colour = (1., 1., 1.) ):
      Object.__init__( self )
      self.data = data # Store data
      self.mu   = mu # Store MU, defaults to ASTRO_MU_SUN from astro_constants.h
      self.__vbo = None
      self.__vboDV = None
      self.__zoom = 1.
      self.__colour = colour

      # Make sure data matches
      if len( data ) % 10 != 0:
         raise AssertionError

      # Initial data processing
      self.__t    = []
      self.__r    = []
      self.__v    = []
      self.__dv   = []
      center      = array( [ 0., 0., 0. ] )
      self.__maxv = 0.
      for i in range( 0, len(data), 10 ):
         # Unit conversion
         t  = data[ i+0 ] * conv_t
         r  = array( [ data[i+1], data[i+2], data[i+3] ] ) * conv_r
         v  = array( [ data[i+4], data[i+5], data[i+6] ] ) * conv_v
         dv = array( [ data[i+7], data[i+8], data[i+9] ] ) * conv_dv
         # Add value
         self.__t.append(  t  )
         self.__r.append(  r  )
         self.__v.append(  v  )
         self.__dv.append( dv )
         # Calculate center
         center += r
         # Calculate maximum velocity
         v_n = linalg.norm( v )
         if v_n > self.__maxv:
            self.__maxv = v_n
      # Center
      self.__center = center / (len(data) / 10)

      # Calculate size
      dmax = 0.
      for r in self.__r:
         dist = linalg.norm( r - self.__center )
         if dist > dmax:
            dmax = dist
      self.__size = dmax

      # Animation stuff
      self.playing   = True
      self.playspeed = (self.__t[ -1 ] - self.__t[ 0 ]) / 10.
      self.curt    = self.__t[ 0 ]
      self.__rad     = dmax / 100.
      self.__showvec = False
      self.__repeat  = False
      self.update( 0. )

      # Generate VBO
      self.__genTraj()
      self.__genDV()

   def __del__( self ):
      "Cleans up after the trajectory."
      # Delete the VBO
      if self.__vbo != None:
         glDeleteBuffers( 1, GLuint( self.__vbo ) )
      if self.__vboDV != None:
         glDeleteBuffers( 1, GLuint( self.__vboDV ) )

   def showVectors( self, enable ):
      self.__showvec = enable

   def setScale( self, zoom ):
      "Gives an indication of the current scale size."
      self.__zoom = zoom
      self.__rad  = 5. / zoom # 10 pixel diameter
      self.__genDV()

   def center( self ):
      "Gets the center of the object."
      return self.__center

   def size( self ):
      "Gets the size of the object."
      return self.__size

   def subdivide( self, subdivide = 1000 ):
      "Adjusts how many subdivisions to use."
      self.__genTraj( subdivide )

   def __genDV( self ):
      """
      Generates the DV markers from the data.
      """

      # First pass to get largest DV
      max_dv = 0
      for i in range( len( self.__t ) ):
         cur_dv = linalg.norm( self.__dv[i] )
         if cur_dv > max_dv:
            max_dv = cur_dv
      self.__dvMax = max_dv

      # Figure out how to normalize
      if max_dv == 0.: # Must actually have dv
         return
      norm_dv = 1./max_dv * (25. / self.__zoom)
      
      # Second pass to set data
      self.__vertexDV = []
      self.__valueDV  = []
      for i in range( len( self.__t ) ):
         if linalg.norm( self.__dv[i] ) > 0.:
            r  = self.__r[i]
            dv = self.__dv[i]
            self.__vertexDV.append( r )
            self.__vertexDV.append( r + dv*norm_dv )
            self.__valueDV.append( linalg.norm(dv) )

      # Convert to numpy
      self.__vertexDV = array( self.__vertexDV, dtype = float32 )

      # Create the vbo
      if self.__vboDV == None:
         self.__vboDV = glGenBuffers( 1 )
      glBindBuffer( GL_ARRAY_BUFFER_ARB, self.__vboDV )
      glBufferData( GL_ARRAY_BUFFER_ARB,
            self.__vertexDV,
            GL_STATIC_DRAW )

   def __genTraj( self, subdivide = 1000 ):
      """
      Generates the vertex trajectory from the data.
      """
      self.__vertex = []
      center = array( (0., 0., 0.) )
      step = (self.__t[-1]-self.__t[0]) / subdivide

      # Create vertex
      for i in range( len( self.__t )-1 ):

         # Calculate how to chop up
         delta = self.__t[ i+1 ] - self.__t[ i+0 ]
         #step  = delta / subdivide

         # Add first point
         r = self.__r[ i+0 ]
         self.__vertex.append( [ r[0], r[1], r[2] ] )
         center += r

         # Take into account dv
         _r = self.__r[ i ]
         _v = self.__v[ i ] + self.__dv[ i ]

         # Add interpolated points
         for j in frange( 0., delta, step ):
            r, v = keplerian_toolbox.propagate_kep( _r, _v, j, self.mu )
            self.__vertex.append( [ r[0], r[1], r[2] ] )
            center += r

      # Add final point
      r = self.__r[ -1 ]
      self.__vertex.append( [ r[0], r[1], r[2] ] )
      center += r

      # Convert to numpy
      self.__vertex = array( self.__vertex, dtype = float32 )

      # Create the VBO
      if self.__vbo == None:
         self.__vbo = glGenBuffers( 1 )
      glBindBuffer( GL_ARRAY_BUFFER_ARB, self.__vbo )
      glBufferData( GL_ARRAY_BUFFER_ARB,
            self.__vertex,
            GL_STATIC_DRAW )

      # Calculate center
      self.__center = center / len( self.__vertex )

      # Calculate size
      dmax = 0.
      for r in self.__vertex:
         dist = linalg.norm( r - self.__center )
         if dist > dmax:
            dmax = dist
      self.__size = dmax


   def set( self, t ):
      self.curt = t
      if self.curt > self.__t[-1]:
         self.curt = self.__t[-1]
      elif self.curt < self.__t[0]:
         self.curt = self.__t[0]
   
   def position( self, t ):
      "Gets the position and velocity vectors of the trajectory at a given instant."
      i = 0
      while i < len( self.__t ):
         if self.__t[i] > t:
            break
         i += 1

      if i > 0:
         i -= 1

      # Calculate point
      t  = t - self.__t[i]
      r  = self.__r[i]
      v  = self.__v[i]
      dv = self.__dv[i]
      r, v = keplerian_toolbox.propagate_kep( r, v+dv, t, self.mu )

      return array(r), array(v), dv


   def setPosition( self, t ):
      self.curt = t

   
   def interval( self ):
      return array( [self.__t[0], self.__t[-1]] )


   def display( self ):
      "Displays the trajectory."
      # Render the trajectory VBO
      glColor3d( *self.__colour )
      glEnableClientState(GL_VERTEX_ARRAY)
      glBindBuffer( GL_ARRAY_BUFFER_ARB, self.__vbo )
      glVertexPointer( 3, GL_FLOAT, 0, None )
      glDrawArrays( GL_LINE_STRIP, 0, len( self.__vertex ) )
      glDisableClientState( GL_VERTEX_ARRAY )

      # Render DV for DSM
      if self.__vboDV != None:
         glEnableClientState(GL_VERTEX_ARRAY)
         glColor3d( 0.0, 0.8, 0.8 )
         glBindBuffer( GL_ARRAY_BUFFER_ARB, self.__vboDV )
         glVertexPointer( 3, GL_FLOAT, 0, None )
         glDrawArrays( GL_LINES, 0, len( self.__vertexDV ) )
         glDisableClientState( GL_VERTEX_ARRAY )

      r, v, dv = self.position( self.curt )

      if self.__showvec:
         # Render position vector
         glColor3d( 1., 0., 0. )
         glBegin( GL_LINES )
         glVertex( 0., 0., 0. )
         glVertex( r[0], r[1], r[2] )
         glEnd()

         # Render velocity vector.
         glColor3d( 0., 1., 0. )
         glBegin( GL_LINES )
         rv = r + v * (1. / self.__maxv) * self.size() / 5.
         glVertex( rv[0], rv[1], rv[2] )
         glVertex( r[0], r[1], r[2] )
         glEnd()

      # Display current position.
      glColor3d( *self.__colour )
      glPushMatrix()
      glTranslatef( r[0], r[1], r[2] )
      glutSolidSphere( self.__rad, 10, 10 )
      glPopMatrix()


