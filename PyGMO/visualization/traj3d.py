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
import os
import sys
import time
import math

# General OpenGL imports
try:
   from OpenGL.GL   import *
   from OpenGL.GLU  import *
   from OpenGL.GLUT import *
   from OpenGL.arrays import ArrayDatatype as ADT
except ImportError:
   raise ImportError( "Warning: The python-opengl bindings are missing, you won't be able to use the visualization module." )

# FTGL
try:
   import FTGL
except ImportError:
   try:
      import _FTGL as FTGL
   except:
      raise ImportError( "Warning: The python-ftgl bindings are missing, you won't be able to use the visualization module." )

# NumPy
try:
   from numpy import *
except ImportError:
   raise ImportError( "Warning: The numpy bindings are missing, you won't be able to use the visualization module." )

# Local imports
from frange import *
from traj3d_object import *

# Misc PaGMO imports
from PyGMO import keplerian_toolbox


###############################################################################
#
#        CAMERA
#
###############################################################################
class Camera:
   
   def __init__( self, width, height, center=(0., 0., 0.), zoom=1. ):
      self.center = array( center )
      self.up     = array( (0., 0., 1.) )
      self.winSize( width, height )
      self.zoomSet( 1. )

      # Initial values
      self.rho    = 1.
      self.theta  = 0.
      self.phi    = 0.
      self.winSize( 1., 1. )
      self.yawvel = 0.
      self.pitchvel = 0.
      self.rollvel = 0.

      # Calculate
      self.__calc()

   def __calc( self ):
      r = self.rho*math.cos( self.phi )
      z = self.rho*math.sin( self.phi )
      self.eye    = array( (r*math.cos(self.theta), r*math.sin(self.theta), z) )
      self.at     = -self.eye

   def winSize( self, width, height ):
      d        = math.sqrt(width**2 + height**2)
      self.__w = width / d
      self.__h = height / d

   def zoomIn( self, factor=math.sqrt(2.) ):
      return self.zoomSet( self.zoom * factor )

   def zoomOut( self, factor=math.sqrt(2.) ):
      return self.zoomSet( self.zoom / factor )

   def zoomSet( self, value ):
      self.zoom  = value
      return self.zoomEffective()

   def zoomEffective( self ):
      glPushMatrix()
      glLoadIdentity()
      w = self.__w
      h = self.__h
      glOrtho( -w, w, -h, h, -100., 10. )
      gluLookAt(  0., 0., 0.,
                  1., 0., 0.,
                  0., 0., 1. )
      glScalef( self.zoom, self.zoom, self.zoom )
      off       = 10. ** floor( math.log( 1./self.zoom, 10. ) )
      proj_base = gluProject( 0., 0., 0. )[1]
      proj_off  = gluProject( 0., 0., off )[1]
      zoom      = (proj_off-proj_base) / off
      glPopMatrix()
      return zoom

   def move( self, vec ):
      self.center += vec

   def yawVel( self, vel ):
      self.yawvel = vel

   def pitchVel( self, vel ):
      self.pitchvel = vel

   def rotate( self, yaw, pitch, roll ):
      self.theta += yaw
      self.phi   += pitch
      if abs(self.phi) > math.pi/2.:
         self.phi = math.pi/2 * (self.phi/abs(self.phi))
      #self.roll  += roll
      self.__calc()

   def absolute( self, yaw, pitch, roll ):
      self.theta  = yaw
      self.phi    = pitch
      if abs(self.phi > math.pi/2.):
         self.phi    = math.fmod( self.phi, math.pi/2. )
      #self.roll   = roll
      self.__calc()

   def refresh( self ):
      self.__calc()

   def view( self ):
      # Reset matrix
      glLoadIdentity()
      w = self.__w
      h = self.__h
      glOrtho( w, -w, -h, h, -100., 10. )
      gluLookAt( 0., 0., 0.,
            #self.center[0], self.center[1], self.center[2],
            self.eye[0], self.eye[1], self.eye[2],
            self.up[0], self.up[1], self.up[2])
      glTranslatef( self.center[0], self.center[1], self.center[2] )
      glScalef( self.zoom, self.zoom, self.zoom )

   def update( self, dt ):
      self.rotate( dt*self.yawvel, dt*self.pitchvel, dt*self.rollvel )


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

      # Set variables
      self.objects = []  # No objects yet
      self.width  = width
      self.height = height
      self.title  = title

      # Initialize GLUT
      glutInit()

      # Create window
      glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE )
      glutInitWindowSize( self.width, self.height )
      self.window = glutCreateWindow( self.title )
      glutSetWindow( self.window )

      # Set up some stuff
      glShadeModel( GL_FLAT )
      glClearColor( 0., 0., 0., 0. )
      glEnable( GL_LINE_SMOOTH )
      #glEnable( GL_DEPTH_TEST )
      #glEnable( GL_COLOR_MATERIAL )
      #glEnable( GL_LIGHTING )
      #glEnable( GL_LIGHT0 )
      self.clear()

      # Misc variables
      self.__time = 0
      self.__quit = False  # Do not exit yet
      self.__mindt = 1./60.
      self.__buttons = []
      self.__camera = Camera( self.width, self.height )

      # Input
      self.__keyboardfunc = None

      # Resize window
      self.reshape( width, height )

      # Set the callbacks
      glutDisplayFunc(     self.__display )
      glutIdleFunc(        self.__idle )
      glutReshapeFunc(     self.__reshape )
      glutKeyboardFunc(    self.__keyboarddown )
      glutKeyboardUpFunc(  self.__keyboardup )
      glutSpecialFunc(     self.__specialdown )
      glutSpecialUpFunc(   self.__specialup )
      glutMouseFunc(       self.__mouse )
      #glutMouseWheelFunc(  self.__wheel )  # From FreeGLUT, not standard GLUT
      glutMotionFunc(      self.__motion )
      #glutPassiveMotionFunc( self.__passive )
      glutVisibilityFunc(  self.__visibility )
      #glutEntryFunc(       self.__entry )

      # Set keymap
      self.keymap = {}
      self.keymap[ 'q' ]      = self.__key_exit
      self.keymap[ '\033' ]   = self.__key_exit
      self.keymap[ 'c' ]      = self.__key_autoZoom
      self.keymap[ 'z' ]      = self.__key_zoomIn
      self.keymap[ 'Z' ]      = self.__key_zoomOut
      self.keymap[ '2' ]      = self.__key_face2
      self.keymap[ '4' ]      = self.__key_face4
      self.keymap[ '5' ]      = self.__key_face5
      self.keymap[ '6' ]      = self.__key_face6
      self.keymap[ '8' ]      = self.__key_face8

      # Special keymap
      self.specialkeymap = {}
      self.specialkeymap[ GLUT_KEY_LEFT ]    = self.__key_left
      self.specialkeymap[ GLUT_KEY_RIGHT ]   = self.__key_right
      self.specialkeymap[ GLUT_KEY_UP ]      = self.__key_up
      self.specialkeymap[ GLUT_KEY_DOWN ]    = self.__key_down

      # Clear the window
      self.clear()


   # Keybindings
   def __key_exit( self, p, x, y ):
      if p:
         self.terminate()
   def __key_zoomIn( self, p, x, y ):
      if p:
         self.__camera.zoomIn()
         self.__objScale()
         self.redisplay()
   def __key_zoomOut( self, p, x, y ):
      if p:
         self.__camera.zoomOut()
         self.__objScale()
         self.redisplay()
   def __key_autoZoom( self, p, x, y ):
      if p:
         self.autozoom()
         self.__objScale()
         self.redisplay()
   def __key_face2( self, p, x, y ):
      self.__key_face( p, 0., -math.pi/2. )
   def __key_face8( self, p, x, y ):
      self.__key_face( p, 0., +math.pi/2. )
   def __key_face4( self, p, x, y ):
      self.__key_face( p, -math.pi/2., 0. )
   def __key_face6( self, p, x, y ):
      self.__key_face( p, +math.pi/2., 0. )
   def __key_face5( self, p, x, y ):
      self.__key_face( p, 0., 0. )
   def __key_face( self, p, roll, pitch, yaw=0. ):
      if p:
         self.__camera.absolute( roll, pitch, yaw )
         self.__objScale()
         self.redisplay()

   # Special keybindings
   def __key_left( self, p, x, y ):
      vel = -math.pi/2. if p else 0.
      self.__camera.yawVel( vel )
   def __key_right( self, p, x, y ):
      vel = math.pi/2. if p else 0.
      self.__camera.yawVel( vel )
   def __key_up( self, p, x, y ):
      vel = -math.pi/2. if p else 0.
      self.__camera.pitchVel( vel )
   def __key_down( self, p, x, y ):
      vel = math.pi/2. if p else 0.
      self.__camera.pitchVel( vel )

   def start( self ):
      """
      Starts the main loop.
      """
      if self.__camera == None:
         raise "No camera"
      self.autozoom()
      self.__time = time.time()
      glutMainLoop()

   def zoom( self, z ):
      self.__camera.zoomSet( z )
      self.__objScale()

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
      self.zoom( 1. / dmax )
      self.__camera.center = array( (0., 0., 0.) )
      self.__camera.absolute( math.pi/4., math.pi/4., 0. )
      #self.__camera.center = center * self.__camera.zoom
      self.__objScale()

   def __objScale( self ):
      """
      Tells all the objects what the current scale level is.
      """
      z = self.__camera.zoomEffective()
      for obj in self.objects:
         obj.setScale( z )

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

      # Change to 2D
      glLoadIdentity()
      glOrtho( 0., self.width, 0., self.height, -1., 1. )

      # Go to ortho overlay
      for obj in self.objects:
         obj.displayOver( self.width, self.height )

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
         time.sleep( self.__mindt - dt )
         return
      self.__time = t

      # Update camera
      if self.__camera != None:
         self.__camera.update( dt )

      # Update objects
      for obj in self.objects:
         obj.update( dt )

      # Draw again
      self.redisplay()


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
      self.__camera.winSize( width, height )
      self.__camera.refresh()
      self.__objScale()
      # Redraw
      self.redisplay()


   def __keyboarddown( self, key, x, y ):
      "Wrapper for keyboard button presses."
      self.__keyboard( True, key, x, y )
   def __keyboardup( self, key, x, y ):
      "Wrapper for keyboard button releases."
      self.__keyboard( False, key, x, y )


   def __keyboard( self, pressed, key, x, y ):
      """
      Handles key presses.
      """
      if self.keymap.has_key( key ):
         self.keymap[key]( pressed, x, y )
      if self.__keyboardfunc != None:
         self.__keyboardfunc( pressed, key, x, y )

   def inputKeyboard( self, func ):
      self.__keyboardfunc = func


   def __specialdown( self, key, x, y ):
      "Wrapper for keyboard special button presses."
      self.__special( True, key, x, y )
   def __specialup( self, key, x, y ):
      "Wrapper for keyboard special button releases."
      self.__special( False, key, x, y )


   def __special( self, pressed, key, x, y ):
      """
      Handles special key presses.
      """
      if self.specialkeymap.has_key( key ):
         self.specialkeymap[key]( pressed, x, y )


   def __mouse( self, button, state, x, y ):
      """
      Handles mouse events.
      """
      if state == GLUT_DOWN:
         self.__buttons.append( button )
         self.__posx = x
         self.__posy = y
         for obj in self.objects:
            obj.mouseDown( button, x, self.height-y )
      elif state == GLUT_UP:
         self.__buttons.remove( button )
         # Hack because otherwise __wheel doesn't seem to run...
         if button == 3:
            self.__wheel( 0, -1, x, y )
         elif button == 4:
            self.__wheel( 0, +1, x, y )
         for obj in self.objects:
            obj.mouseUp( button, x, self.height-y )

   def __wheel( self, wheel, direction, x, y ):
      if direction > 0:
         self.__camera.zoomOut()
         self.__objScale()
         self.redisplay()
      elif direction < 0:
         self.__camera.zoomIn()
         self.__objScale()
         self.redisplay()

   def __motion( self, x, y ):
      """
      Handles mouse motion events.
      """
      mod = glutGetModifiers()
      if GLUT_MIDDLE_BUTTON in self.__buttons:
         sensitivity = 0.005
         delta    =  -(x - self.__posx), y - self.__posy
         if (mod & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL:
            self.__moveCam( delta[0], delta[1] )
         else:
            yaw   = delta[0] * sensitivity
            pitch = delta[1] * sensitivity
            roll  = 0.
            self.__camera.rotate( yaw, pitch, 0. )
         self.__posx = x
         self.__posy = y
         self.redisplay()
      for obj in self.objects:
         obj.mouseMove( x, self.height-y )

   def __moveCam( self, x, y ):
      # Need to calculate projection base
      base_x = cross( self.__camera.up, self.__camera.at )
      base_y = cross( base_x, self.__camera.at )
      # Need to normalize vectors
      base_x /= linalg.norm( base_x )
      base_y /= linalg.norm( base_y )
      # Move along the projection base
      move    = base_x * x / self.width + base_y * y / self.height
      self.__camera.move( move )

   def __visibility( self, vis ):
      self.__idle()

   def redisplay( self ):
      glutPostRedisplay()

   def reshape( self, width, height ):
      """
      Provokes a window resize.
      """
      self.__reshape( width, height )



