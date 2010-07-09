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
from OpenGL.raw   import *
from OpenGL.GL   import *
from OpenGL.GLU  import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
from OpenGL.arrays import ArrayDatatype as ADT


# NumPy
from numpy import *

# Local imports
from frange import *

# Misc PaGMO imports
from PyGMO import keplerian_toolbox


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

   def size( self ):
      if self.__class__ is Object:
         raise NotImplementedError

   def center( self ):
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

   def __init__( self, data, mu = 1.32712428e20 ):
      Object.__init__( self )
      self.data = data # Store data
      self.mu   = mu # Store MU, defaults to ASTRO_MU_SUN from astro_constants.h
      self.__vbo = None

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
      center      = array( [ 0., 0., 0. ] )
      for i in range( 0, len(data), 10 ):
         # Unit conversion
         t  = data[ i+0 ] * 24. * 3600. # days -> seconds
         r  = array( [ data[i+1], data[i+2], data[i+3] ] ) * 1000. # km -> m
         v  = array( [ data[i+4], data[i+5], data[i+6] ] ) * 1000. # km/s -> m/s
         dv = array( [ data[i+7], data[i+8], data[i+9] ] ) # ??
         # Add value
         self.__t.append(  t  )
         self.__r.append(  r  )
         self.__v.append(  v  )
         self.__dv.append( dv )
         # Calculate center
         center += r
      # Center
      self.__center = center / (len(data) / 10)

      # Calculate size
      dmax = 0.
      for r in self.__r:
         dist = linalg.norm( r - self.__center )
         if dist > dmax:
            dmax = dist
      self.__size = dmax

      # Generate VBO
      self.__genTraj()

   def center( self ):
      return self.__center

   def size( self ):
      return self.__size

   def __genTraj( self, subdivide = 50 ):
      """
      Generates the vertex trajectory from the data.
      """
      self.__vertex = []
      center = array( (0., 0., 0.) )

      # Create vertex
      for i in range( len( self.__t )-1 ):

         # Calculate how to chop up
         delta = self.__t[ i+1 ] - self.__t[ i+0 ]
         step  = delta / subdivide

         # Add first point
         r = self.__r[ i+0 ]
         self.__vertex.append( [ r[0], r[1], r[2] ] )
         center += r

         # Add interpolated points
         for j in frange( 0., delta, step ):
            r, v = keplerian_toolbox.propagate_kep( tuple(self.__r[ i+0 ]), tuple(self.__v[ i+0 ]), j, self.mu )
            self.__vertex.append( [ r[0], r[1], r[2] ] )
            center += r

      # Add final point
      r = self.__r[ -1 ]
      self.__vertex.append( [ r[0], r[1], r[2] ] )
      center += r

      # Convert to numpy
      self.__vertex = array( self.__vertex, dtype = float32 )

      # Create the VBO
      if self.__vbo != None:
         glDeleteBuffers( 1, GLuint( self.__vbo ) )
      self.__vbo = glGenBuffers( 1 )
      glBindBuffer( GL_ARRAY_BUFFER_ARB, self.__vbo )
      glBufferData( GL_ARRAY_BUFFER_ARB,
            ADT.arrayByteCount( self.__vertex ),
            ADT.voidDataPointer( self.__vertex ),
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


   def display( self ):
      glEnableClientState(GL_VERTEX_ARRAY)

      glColor3d( 1., 1., 1. )
      glBindBuffer( GL_ARRAY_BUFFER_ARB, self.__vbo )
      glVertexPointer( 3, GL_FLOAT, 0, None )
      glDrawArrays( GL_LINE_STRIP, 0, len( self.__vertex ) )

      glDisableClientState( GL_VERTEX_ARRAY )


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
      glBegin(GL_LINES)
      glColor3d( *self.xColor )
      glVertex3d( 0, 0, 0 )
      glVertex3d( self.expanse, 0, 0 )
      glEnd()

      glBegin(GL_LINES)
      glColor3d( *self.yColor )
      glVertex3d( 0, 0, 0 )
      glVertex3d( 0., self.expanse, 0 )
      glEnd()

      glBegin(GL_LINES)
      glColor3d( *self.zColor )
      glVertex3d( 0, 0, 0 )
      glVertex3d( 0., 0., self.expanse )
      glEnd()



###############################################################################
#
#        CAMERA
#
###############################################################################
class Camera:
   
   def __init__( self, center=(0., 0., 0.), zoom=1. ):
      self.center = array( center )
      self.zoom   = zoom

      # Euler angles
      self.yaw    = 0.
      self.pitch  = 0.
      self.roll   = 0.

      # Calculate
      self.__calc()

   def __calc( self ):
      cosy = math.cos( self.yaw )
      siny = math.sin( self.yaw )
      cosp = math.cos( self.pitch )
      sinp = math.sin( self.pitch )

      u = array( ( cosy, siny, 0. ) )
      v = array( ( siny*cosp, cosy*cosp, sinp ) )
      w = cross( u, v )

      self.right  = u
      self.at     = v
      self.up     = w

   def zoomIn( self, factor=math.sqrt(2.) ):
      self.zoom *= factor

   def zoomOut( self, factor=math.sqrt(2.) ):
      self.zoom /= factor

   def zoomSet( self, value ):
      self.zoom  = value

   def move( self, vec ):
      self.center += vec

   def rotate( self, yaw, pitch, roll ):
      self.yaw   += yaw
      self.pitch += pitch
      self.roll  += roll
      self.__calc()

   def absolute( self, yaw, pitch, roll ):
      self.yaw    = yaw
      self.pitch  = pitch
      self.roll   = roll
      self.__calc()

   def refresh( self ):
      self.__calc()

   def view( self ):
      # Reset matrix
      glMatrixMode( GL_PROJECTION )
      glLoadIdentity()
      # Rotate to have x forward, y right, z top
      glRotatef( -90., 0., 1., 0. )
      glRotatef( -90., 1., 0., 0. )
      u = self.right
      v = self.at
      w = self.up
      glRotatef( self.yaw * 180./math.pi,   0., 0., 1. )
      glRotatef( self.pitch * 180./math.pi, u[0], u[1], u[2] )
      glRotatef( self.roll * 180./math.pi,  v[0], v[1], v[2] )
      glTranslate( self.center[0], self.center[1], self.center[2] )
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
      self.keymap[ 'c' ]      = self.__key_autoZoom
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
   def __key_autoZoom( self, x, y ):
      self.autozoom()
      self.redisplay()


   def start( self ):
      """
      Starts the main loop.
      """
      if self.__camera == None:
         raise "No camera"
      self.autozoom()
      self.__time = time.time()
      glutMainLoop()


   def autozoom( self ):
      """
      Automatically focuses and zooms in on all the objects (fits entire scene).
      """

      # Calculate scene center
      """
      i = 0.
      center = array( (0., 0., 0.) )
      for objs in self.objects:
         c = objs.center()
         if c != None:
            i += 1.
            center += objs.center()
      center /= i
      """

      # Calculate max distance from center
      dmax = 0.
      for objs in self.objects:
         center = objs.center()
         size   = objs.size()
         dist   = 0.
         if center != None:
            dist += linalg.norm( center )
         if size != None:
            dist += size
         if dist > dmax:
            dmax = dist
      if dmax == 0.:
         dmax = 1

      # Set zoom and center
      self.__camera.zoom = 1. / dmax
      self.__camera.center = array( (0., 0., 0.) )
      #self.__camera.center = center * self.__camera.zoom

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
            # Need to calculate projection base
            base_x = cross( self.__camera.up, self.__camera.at )
            base_y = cross( base_x, self.__camera.at )
            # Need to normalize vectors
            base_x /= linalg.norm( base_x )
            base_y /= linalg.norm( base_y )
            # Move along the projection base
            move    = base_x * delta[0] / self.width + base_y * delta[1] / self.height
            self.__camera.move( move )
         else:
            yaw   = delta[0] * sensitivity
            pitch = delta[1] * sensitivity
            roll  = 0.
            #self.__camera.rotate( yaw, pitch, 0. )
            self.__camera.rotate( yaw, 0., pitch )
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



