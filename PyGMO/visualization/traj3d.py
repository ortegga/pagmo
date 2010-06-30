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
import sys
import time
import math

# General OpenGL imports
from OpenGL.GL   import *
from OpenGL.GLU  import *
from OpenGL.GLUT import *

# OpenGL VBO imports
#from OpenGL.arrays.arraydatatype import ArrayDatatype
#from OpenGL.arrays.formathandler import FormatHandler 
#from OpenGL.GL.ARB import vertex_buffer_object 


###############################################################################
#
#     ENGINE OBJECTS
#
###############################################################################
class Object:
   """
   Core object
   """

   def __init__( self ):
      if self.__class__ is Object:
         raise NotImplementedError

   def display( self ):
      if self.__class__ is Object:
         raise NotImplementedError

   def update( self, dt ):
      if self.__class__ is Object:
         raise NotImplementedError


###############################################################################
class ObjectGroup:
   """
   Group of objects.
   """

   def __init__( self ):
      self.__objs = []

   def display( self ):
      for obj in self.__objs:
         obj.display()

   def update( self, dt ):
      for obj in self.__objs:
         obj.update( dt )

   def add( self, obj ):
      self.__objs.append( obj )

   def remove( self, obj ):
      self.__objs.remove( obj )


###############################################################################
class Trajectory(Object):
   """
   Represents a 3D trajectory.
   """

   def __init( self ):
      Object.__init__( self )

   # Conics
   #  A x^2 + B x y + C y^2 + D x + E y + F = 0
   def display( self ):
      glColor3d( 1., 1., 1. )
      glBegin( GL_LINE_STRIP )

      e        = 0.5
      p        = (1 + e) / e
      ep       = e * p
      theta    = 0.
      finish   = math.pi * 2.
      step     = math.pi * 2. / 100.
      while theta < finish:
         r = ep / ( 1. + e * math.cos(theta) )
         glVertex( r*math.cos(theta), r*math.sin(theta), 0. )
         theta += step

      glEnd()


###############################################################################
class Origin(Object):
   """
   Represents 3 axes on the origin.
   """

   xColor = 1.0, 0.0, 0.0
   yColor = 0.0, 1.0, 0.0
   zColor = 0.0, 0.0, 1.0
   stipple = 0x0f0f

   def __init__( self, expanse=20 ):
      Object.__init__( self )
      self.expanse = expanse

   def display( self ):
      glLineStipple(1, self.stipple)
      glDisable(GL_LINE_STIPPLE)
      glBegin(GL_LINES)
      glColor3d(*self.xColor)
      glVertex3d(0, 0, 0)
      glVertex3d(self.expanse, 0, 0)
      glEnd()
      glEnable(GL_LINE_STIPPLE)
      glBegin(GL_LINES)
      glVertex3d(0, 0, 0)
      glVertex3d(-self.expanse, 0, 0)
      glEnd()
      glDisable(GL_LINE_STIPPLE)
      glBegin(GL_LINES)
      glColor3d(*self.yColor)
      glVertex3d(0, 0, 0)
      glVertex3d(0, self.expanse, 0)
      glEnd()
      glEnable(GL_LINE_STIPPLE)
      glBegin(GL_LINES)
      glVertex3d(0, 0, 0)
      glVertex3d(0, -self.expanse, 0)
      glEnd()
      glDisable(GL_LINE_STIPPLE)
      glBegin(GL_LINES)
      glColor3d(*self.zColor)
      glVertex3d(0, 0, 0)
      glVertex3d(0, 0, self.expanse)
      glEnd()
      glEnable(GL_LINE_STIPPLE)
      glBegin(GL_LINES)
      glVertex3d(0, 0, 0)
      glVertex3d(0, 0, -self.expanse)
      glEnd()
      glDisable(GL_LINE_STIPPLE)



###############################################################################
#
#        ENGINE ITSELF
#
###############################################################################
class traj3d:
   """
   Core class representing the entire 3d trajectory engine.
   """

   def __init__( self, title="3D Trajectory Visualizer", width=640, height=480 ):
      """
      Initalizes the engine.
      """

      # Initialize GLUT
      glutInit()

      # Set variables
      self.objects = []  # No objects yet
      self.width  = width
      self.height = height
      self.title  = title

      # Misc variables
      self.__time = 0
      self.__quit = False  # Do not exit yet
      self.__mindt = 0.

      # Set up some stuff
      glShadeModel( GL_FLAT )
      glClearColor( 1., 1., 1., 0. )
      glEnable( GL_DEPTH_TEST )
      #glEnable( GL_COLOR_MATERIAL )
      #glEnable( GL_LIGHTING )
      #glEnable( GL_LIGHT0 )

      # Create window
      glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE )
      glutInitWindowSize( self.width, self.height )
      self.window = glutCreateWindow( self.title )
      glutSetWindow( self.window )

      # Resize window
      self.reshape( width, height )

      # Set the callbacks
      glutDisplayFunc(     self.__display )
      glutIdleFunc(        self.__idle )
      glutReshapeFunc(     self.__reshape )
      #glutKeyboardFunc(    self.__keyboard )
      #glutSpecialFunc(     self.__special )
      #glutMouseFunc(       self.__mouse )
      #glutMotionFunc(      self.__motion )
      #glutPassiveMotionFunc( self.__passive )
      #glutVisibilityFunc(  self.__visibility )
      #glutEntryFunc(       self.__entry )

      # Clear the window
      self.clear()


   def start( self ):
      """
      Starts the main loop.
      """
      self.__time = time.time()
      glutMainLoop()


   def terminate( self ):
      """
      Terminates the engine.
      """
      sys.exit()


   def add( self, obj ):
      """
      Adds an object to the engine.
      """
      self.objects.append( obj )


   def remove( self, obj ):
      """
      Removes an object from the engine.
      """
      self.objects.remove( obj )


   def clear( self ):
      """
      Clears the screen.
      """
      glClear( GL_COLOR_BUFFER_BIT )


   def flush( self ):
      """
      Flushes data and swaps buffers.
      """
      glFlush()
      glutSwapBuffers()


   def __display( self ):
      """
      Updates the display.
      """
      self.clear()
      for obj in self.objects:
         obj.display()
      self.flush()


   def __idle( self ):
      """
      Handles object updating.
      """
      if self.__quit:
         self.terminate()

      # See if must update
      t  = time.time()
      dt = t - self.__time
      if self.__mindt > 0. and dt < self.__mindt:
         return;
      self.__time = t

      # Update objects
      for obj in self.objects:
         obj.update( dt )


   def __reshape( self, width=640, height=480 ):
      """
      Handles window resizes.
      """
      glViewport( 0, 0, width, height )
      glMatrixMode( GL_PROJECTION )
      glLoadIdentity()
      #gluPerspective( 60.0, float(width)/height, .1, 1000. )
      glMatrixMode( GL_MODELVIEW )
      glLoadIdentity()

      # Redraw
      self.__display()

   def reshape( self, width, height ):
      """
      Provokes a window resize.
      """
      self.__reshape( width, height )



