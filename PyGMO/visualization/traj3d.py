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

   def setScale( self, zoom ):
      if self.__class__ is Object:
         raise NotImplementedError

   def center( self ):
      if self.__class__ is Object:
         raise NotImplementedError

   def display( self ):
      if self.__class__ is Object:
         raise NotImplementedError

   def displayOver( self, width, height ):
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

   def __init__( self, data, conv_t=1., conv_r=1., conv_v=1., conv_dv=1., mu = 1.32712428e20 ):
      Object.__init__( self )
      self.data = data # Store data
      self.mu   = mu # Store MU, defaults to ASTRO_MU_SUN from astro_constants.h
      self.__vbo = None
      self.fontsize( 16 )

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
      self.__curt    = self.__t[ 0 ]
      self.__rad     = dmax / 100.
      self.__showvec = False
      self.__repeat  = False
      self.update( 0. )

      # Generate VBO
      self.__genTraj()

   def __del__( self ):
      "Cleans up after the trajectory."
      # Delete the VBO
      if self.__vbo != None:
         glDeleteBuffers( 1, GLuint( self.__vbo ) )

   def fontsize( self, size ):
      "Sets the font size."
      self.font = FTGL.PixmapFont( os.path.join( os.path.dirname(__file__), "Vera.ttf" ) )
      self.font.FaceSize( size )
      self.fontsize = size

   def showVectors( self, enable ):
      self.__showvec = enable

   def setScale( self, zoom ):
      "Gives an indication of the current scale size."
      self.__rad  = 0.01 / zoom

   def center( self ):
      "Gets the center of the object."
      return self.__center

   def size( self ):
      "Gets the size of the object."
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
            r, v = keplerian_toolbox.propagate_kep( self.__r[ i+0 ], self.__v[ i+0 ], j, self.mu )
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
            #ADT.arrayByteCount( self.__vertex ),
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
      r, v = keplerian_toolbox.propagate_kep( r, v, t, self.mu )

      return array(r), array(v), dv


   def display( self ):
      "Displays the trajectory."
      # Render the trajectory VBO
      glColor3d( 1., 1., 1. )
      glEnableClientState(GL_VERTEX_ARRAY)
      glBindBuffer( GL_ARRAY_BUFFER_ARB, self.__vbo )
      glVertexPointer( 3, GL_FLOAT, 0, None )
      glDrawArrays( GL_LINE_STRIP, 0, len( self.__vertex ) )
      glDisableClientState( GL_VERTEX_ARRAY )

      # Get data
      r = self.__curr
      v = self.__curv

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
      glColor3d( 1., 1., 1. )
      glPushMatrix()
      glTranslatef( r[0], r[1], r[2] )
      glutSolidSphere( self.__rad, 10, 10 )
      glPopMatrix()


   def faster( self, factor=1.1 ):
      "Speeds up the animation."
      self.playspeed *= factor
   
   def slower( self, factor=1.1 ):
      "Slows down the animation."
      self.playspeed /= factor

   def speed( self, speed ):
      "Sets the animation speed."
      self.playspeed = speed

   def duration( self, duration ):
      "Sets the animation total duration in seconds."
      self.playspeed = (self.__t[ -1 ] - self.__t[ 0 ]) / duration

   def restart( self ):
      "Restarts the playback."
      self.__curt = self.__t[ 0 ]

   def repeat( self, enable ):
      "Sets animation repetition."
      self.__repeat = enable

   def pause( self, enable ):
      "Pauses the playback."
      self.playing = not enable

   def ispaused( self ):
      "Checks to see if is paused."
      return not self.playing

   def update( self, dt ):
      "Updates the animation of the trajectory."
      # must be playing
      if self.playing:
         # Update
         self.__curt += self.playspeed * dt
         # Stop when finished
         if self.__curt > self.__t[ -1 ]:
            if self.__repeat:
               self.__curt = self.__t[ 0 ]
            else:
               self.__curt = self.__t[ -1 ]
               self.playing = False
      r, v, dv = self.position( self.__curt )
      self.__curr  = r
      self.__curv  = v
      self.__curdv = dv

   def displayOver( self, width, height):
      "Displays the trajectory overlay."

      glEnable(GL_BLEND)
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

      t  = self.__curt
      r  = self.__curr
      v  = self.__curv
      dv = self.__curdv

      bx = 10.
      y  = 10. + 2.*( 1.5 *  self.fontsize )

      # Text to render
      st = "Time: %.2E s" % linalg.norm( t )
      sr = "Dist: %.2E m" % linalg.norm( r )
      sv = "Vel: %.2E m" % linalg.norm( v )

      # Calculate width
      w  = max( self.font.Advance( sr ),
                self.font.Advance( sv ) )
      x = width - w  - bx

      # Time
      glColor3d( 1., 1., 1. )
      glRasterPos( x, y )
      self.font.Render( st )

      # Position
      y  -= self.fontsize * 1.5
      glColor3d( 1., 0., 0. )
      glRasterPos( x, y )
      self.font.Render( sr )

      # Velocity
      y  -= self.fontsize * 1.5
      glColor3d( 0., 1., 0. )
      glRasterPos( x, y )
      self.font.Render( sv )

      glDisable(GL_BLEND)



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
      self.up     = array( (0., 0., 1.) )
      self.zoom   = zoom

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
      d = math.sqrt(width**2 + height**2)
      self.__w = width / d
      self.__h = height / d

   def zoomIn( self, factor=math.sqrt(2.) ):
      self.zoom *= factor

   def zoomOut( self, factor=math.sqrt(2.) ):
      self.zoom /= factor

   def zoomSet( self, value ):
      self.zoom  = value

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
      self.phi    = math.fmod( self.phi, math.pi/2. )
      #self.roll   = roll
      self.__calc()

   def refresh( self ):
      self.__calc()

   def view( self ):
      # Reset matrix
      glMatrixMode( GL_PROJECTION )
      glLoadIdentity()
      w = self.__w
      h = self.__h
      glOrtho( -w, w, -h, h, -100., 10. )
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
      self.__camera = Camera()
      self.__camera.absolute( math.pi/4., math.pi/4., 0. )

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
      self.__camera.absolute( math.pi/4., math.pi/4., 0. )
      #self.__camera.center = center * self.__camera.zoom
      self.__objScale()

   def __objScale( self ):
      """
      Tells all the objects what the current scale level is.
      """
      z = self.__camera.zoom
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
      glMatrixMode( GL_PROJECTION )
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
         delta    =  x - self.__posx, y - self.__posy
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



