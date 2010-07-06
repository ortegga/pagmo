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
from OpenGL.GLUT.freeglut import *

# OpenGL VBO imports
#from OpenGL.arrays.arraydatatype import ArrayDatatype
#from OpenGL.arrays.formathandler import FormatHandler 
#from OpenGL.GL.ARB import vertex_buffer_object 

# Local imports
from vector import *
from frange import *

# Misc PaGMO imports
from PyGMO import kep_toolbox


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

   def __init__( self, data ):
      Object.__init__( self )
      self.data = data # Store data

      # Make sure data matches
      if type( data ).__name__ != 'tuple':
         raise TypeError
      for v in data:
         if type( v ).__name__ != 'float':
            raise TypeError
      if len( data ) % 10 != 0:
         raise AssertionError

      # Initial data processing
      self.__t    = []
      self.__r    = []
      self.__v    = []
      self.__dv   = []
      for i in range( 0, len(data), 10 ):
         # Unit conversion
         t  = data[ i+0 ] * 24. * 3600. # days -> seconds
         r  = Vec3( (data[i+1], data[i+2], data[i+3]) ) * 1000. # km -> m
         v  = Vec3( (data[i+4], data[i+5], data[i+6]) ) * 1000. # km/s -> m/s
         dv = Vec3( (data[i+7], data[i+8], data[i+9]) ) # ??
         # Add value
         self.__t.append(  t  )
         self.__r.append(  r  )
         self.__v.append(  v  )
         self.__dv.append( dv )
      self.__t    = tuple( self.__t )
      self.__r    = tuple( self.__r )
      self.__v    = tuple( self.__v )
      self.__dv   = tuple( self.__dv )

   def display( self ):
      glColor3d( 1., 1., 1. )
      glBegin( GL_LINE_STRIP )

      mu    = 1.32712428e20 # ASTRO_MU_SUN from astro_constants.h
      s     = 1e10

      for i in range( len( self.__t )-1 ):

         delta = self.__t[ i+1 ] - self.__t[ i+0 ]
         step  = delta/50

         r = self.__r[ i+0 ]
         glVertex( r[0]/s, r[1]/s, r[2]/s )

         for j in frange( 0., delta, step ):
            r, v = kep_toolbox.propagate_kep( self.__r[ i+0 ].data, self.__v[ i+0 ].data, j, mu )
            glVertex( r[0]/s, r[1]/s, r[2]/s )

      r = self.__r[ -1 ]
      glVertex( r[0]/s, r[1]/s, r[2]/s )

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
#        CAMERA
#
###############################################################################
class Camera:
   
   def __init__( self, rho=1., center=(0., 0., 0.), up=(0., 0., 1.), zoom=1. ):
      self.center = center
      self.up     = up
      self.zoom   = zoom

      # Internal eye stuff
      self.rho    = rho
      self.theta  = 0.
      self.phi    = 0.

      # Calculate
      self.__calc()

   def __calc( self ):
      r = self.rho * math.cos( self.phi )
      z = self.rho * math.sin( self.phi )
      self.eye = ( r * math.cos( self.theta ),
                   r * math.sin( self.theta ),
                   z )

   def zoomIn( self, factor=math.sqrt(2.) ):
      self.zoom *= factor

   def zoomOut( self, factor=math.sqrt(2.) ):
      self.zoom /= factor

   def move( self, x, y ):
      cam = self.eye - self.center
      self.center = ( self.center[0] + x, self.center[1] + y, 0 )
      self.__calc()

   def rotate( self, theta, phi ):
      self.theta += theta
      self.phi   += phi
      self.__calc()

   def absolute( self, theta, phi ):
      self.theta  = theta
      self.phi    = phi
      self.__calc()

   def refresh( self ):
      self.__calc()

   def view( self ):
      glMatrixMode( GL_PROJECTION )
      glLoadIdentity()
      gluLookAt( self.center[0], self.center[1], self.center[2],
            self.eye[0], self.eye[1], self.eye[2],
            self.up[0], self.up[1], self.up[2] )
      glScalef( self.zoom, self.zoom, self.zoom )

   def update( self, dt ):
      return


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
      self.__buttons = []
      self.__camera = Camera()

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
      #self.reshape( width, height )

      # Set the callbacks
      glutDisplayFunc(     self.__display )
      glutIdleFunc(        self.__idle )
      glutReshapeFunc(     self.__reshape )
      glutKeyboardFunc(    self.__keyboard )
      #glutSpecialFunc(     self.__special )
      glutMouseFunc(       self.__mouse )
      glutMouseWheelFunc(  self.__wheel )  # From FreeGLUT, not standard GLUT
      glutMotionFunc(      self.__motion )
      #glutPassiveMotionFunc( self.__passive )
      #glutVisibilityFunc(  self.__visibility )
      #glutEntryFunc(       self.__entry )

      # Set keymap
      self.keymap = {}
      self.keymap[ 'q' ]      = self.__key_exit
      self.keymap[ '\033' ]   = self.__key_exit
      self.keymap[ 'z' ]      = self.__key_zoomIn
      self.keymap[ 'Z' ]      = self.__key_zoomOut

      # Clear the window
      self.clear()


   # Keybindings
   def __key_exit( self, x, y ):
      self.terminate()
   def __key_zoomIn( self, x, y ):
      self.__camera.zoomIn()
      self.redisplay()
   def __key_zoomOut( self, x, y ):
      self.__camera.zoomOut()
      self.redisplay()


   def start( self ):
      """
      Starts the main loop.
      """
      if self.__camera == None:
         raise "No camera"
      self.__time = time.time()
      glutMainLoop()


   def terminate( self ):
      """
      Terminates the engine.
      """
      sys.exit()


   def camera( self, cam ):
      """
      Sets the camera
      """
      self.camera = cam


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
      # Camera
      self.__camera.view()
      # Objects
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

      # Update camera
      if self.__camera != None:
         self.__camera.update( dt )

      # Update objects
      for obj in self.objects:
         obj.update( dt )


   def __reshape( self, width=640, height=480 ):
      """
      Handles window resizes.
      """
      glutSetWindow( self.window )
      glutReshapeWindow( width, height )
      self.width  = width
      self.height = height
      glViewport( 0, 0, width, height )
      # Update camera
      self.__camera.refresh()
      # Redraw
      self.redisplay()


   def __keyboard( self, key, x, y ):
      """
      Handles key presses.
      """
      if self.keymap.has_key( key ):
         self.keymap[key]( x, y )


   def __mouse( self, button, state, x, y ):
      """
      Handles mouse events.
      """
      if state == GLUT_DOWN:
         self.__buttons.append( button )
         self.__posx = x
         self.__posy = y
      elif state == GLUT_UP:
         self.__buttons.remove( button )
         # Hack because otherwise __wheel doesn't seem to run...
         if button == 3:
            self.__wheel( 0, -1, x, y )
         elif button == 4:
            self.__wheel( 0, +1, x, y )

   def __wheel( self, wheel, direction, x, y ):
      if direction > 0:
         self.__camera.zoomOut()
         self.redisplay()
      elif direction < 0:
         self.__camera.zoomIn()
         self.redisplay()

   def __motion( self, x, y ):
      """
      Handles mouse motion events.
      """
      mod = glutGetModifiers()
      if GLUT_MIDDLE_BUTTON in self.__buttons:
         sensitivity = 0.005
         delta    =  x - self.__posx, y - self.__posy
         if (mod & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL:
            self.__camera.move( delta[0] * sensitivity, delta[1] * sensitivity )
         else:
            self.__camera.rotate( delta[0] * sensitivity, delta[1] * sensitivity )
         self.__posx = x
         self.__posy = y
         self.redisplay()


   def redisplay( self ):
      glutPostRedisplay()


   def reshape( self, width, height ):
      """
      Provokes a window resize.
      """
      self.__reshape( width, height )



