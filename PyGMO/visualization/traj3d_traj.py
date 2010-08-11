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
from traj3d_object import *
from traj3d_path import *
from traj3d_planet import *

# Misc PaGMO imports
from PyGMO import keplerian_toolbox, astro_toolbox


###############################################################################
class Trajectory(Object):
   """
   Represents a 3D trajectory.
   """

   def __init__( self, data, conv_t=1., conv_r=1., conv_v=1., conv_dv=1., mu = 1.32712428e20 ):
      Object.__init__( self )
      self.__vbo = None
      self.__zoom = 1.
      self.__axes = None
      self.__controls = True
      self.__drag = False
      self.__dragPlaying = True
      self.playing = True
      self.control_size = 20
      self.control_pos = array( (50., 50.) )
      self.__path = Path( data, conv_t, conv_r, conv_v, conv_dv, mu )
      self.__t = self.__path.interval()
      self.__curt = self.__t[0]
      self.__planets = []
      self.setUnits()
      self.duration( 30. )
      self.update( 0. )
      self.fontsize( 16 )


   def axes( self, enable ):
      if enable:
         self.__axes = Axes( self.font )
         self.__axes.setUnits( self.__pos_val, self.__pos_sym )
      else:
         self.__axes = None

   def fontsize( self, size ):
      "Sets the font size."
      self.font = FTGL.PixmapFont( os.path.join( os.path.dirname(__file__), "Vera.ttf" ) )
      self.font.FaceSize( size )
      self.fontsize = size
      if self.__axes != None:
         self.__axes.setFont( self.font )
      for pnt in self.__planets:
         pnt.setFont( self.font, size )

   def showVectors( self, enable ):
      self.__path.showVectors( enable )

   def controls( self, enable=None ):
      if enable is None:
         self.__controls = not self.__controls
      else:
         self.__controls = enable

   def setScale( self, zoom ):
      "Gives an indication of the current scale size."
      self.__zoom = zoom
      self.__path.setScale( zoom )
      for pnt in self.__planets:
         pnt.setScale( zoom )

   def center( self ):
      "Gets the center of the object."
      return self.__path.center()

   def size( self ):
      "Gets the size of the object."
      return self.__path.size()

   def addPlanets( self, mjd2000, planets ):
      "Adds some planets."
      self.__planetsStart = mjd2000
      self.__planetsColour = planets
      for key, value in planets.items():
         self.__planets.append( Planet( mjd2000, key, value ) )
      for pnt in self.__planets:
         pnt.setFont( self.font, self.fontsize )
         pnt.speed( self.playspeed )
      """
      self.setPosition( self.__t[0] )
      print("START")
      print( self.__path.position( self.__path.curt )[0] )
      for pnt in self.__planets:
         print( pnt.name(), pnt.position( self.__path.curt )[0] )
      self.setPosition( self.__t[-1] )
      print("STOP")
      print( self.__path.position( self.__path.curt )[0] )
      for pnt in self.__planets:
         print( pnt.name(), pnt.position( self.__path.curt )[0] )
      """

   def display( self ):
      "Displays the trajectory."
      for pnt in self.__planets:
         pnt.display()
      self.__path.display()

      # Get data
      r = self.__curr
      v = self.__curv
      dv = self.__curdv
      origin = gluProject( 0., 0., 0. )
      pos    = gluProject( r[0], r[1], r[2] ) # Save screen position
      if self.__axes:
         axes = [ { "colour" : (1.,0.,0.),"pos" : pos } ]
         self.__axes.refresh( origin, self.__zoom, axes )

   def faster( self, factor=1.1 ):
      "Speeds up the animation."
      self.speed( self.playespeed * factor )
   
   def slower( self, factor=1.1 ):
      "Slows down the animation."
      self.speed( self.playespeed / factor )

   def speed( self, speed ):
      "Sets the animation speed."
      self.playspeed = speed
      for pnt in self.__planets:
         pnt.speed( speed )

   def duration( self, duration ):
      "Sets the animation total duration in seconds."
      self.speed( (self.__t[ -1 ] - self.__t[ 0 ]) / duration )

   def setPosition( self, t ):
      "Sets the current position."
      self.__curt = t
      self.__path.setPosition( t )
      for pnt in self.__planets:
         pnt.setPosition( t - self.__t[0] + self.__planetsStart, self.__t[0] )

   def restart( self ):
      "Restarts the playback."
      self.setPosition( self.__t[0] )

   def repeat( self, enable=True ):
      "Sets animation repetition."
      self.__repeat = enable

   def pause( self, enable=True ):
      "Pauses the playback."
      self.playing = not enable

   def ispaused( self ):
      "Checks to see if is paused."
      return not self.playing

   def mouseDown( self, button, x, y ):
      if button != GLUT_LEFT_BUTTON:
         return False
      # Check position
      x = x - self.control_pos[0]
      y = y - self.control_pos[1]
      if y < 0 or y >= self.control_size:
         return False
      w = self.control_size
      if x >= 3*(w+10) and x < self.control_len-3*(w+10):
         self.__drag = True
         self.__dragPlaying = self.playing
         self.pause()
         x = x - 3*(w+10)
         w = self.control_len - 6*(w+10)
         p = x / w
         self.__curt = (self.__t[-1] - self.__t[0])*p + self.__t[0]
         self.setPosition( self.__curt )
         return True
      return False

   def mouseUp( self, button, x, y ):
      "Handle mouse clicks."
      # Check button
      if button != GLUT_LEFT_BUTTON:
         return False

      # Stop dragging
      if self.__drag:
         self.__drag = False
         self.pause( not self.__dragPlaying )

      # Check position
      x = x - self.control_pos[0]
      y = y - self.control_pos[1]
      if x < 0 or x >= self.control_len:
         return False
      if y < 0 or y >= self.control_size:
         return False

      # Check click
      w = self.control_size
      if x < w: # Rewind
         self.pause()
         self.restart()
         return True
      elif x >= w+10 and x < w+10+w: # Stop
         self.pause()
         self.__controls = False
         return True
      elif x >= 2*(w+10) and x < 2*(w+10)+w: # Slow down
         self.speed( self.playspeed / 1.1 )
         return True
      elif x >= self.control_len-w: # End
         self.pause()
         self.setPosition( self.__t[ -1 ] )
         return True
      elif x >= self.control_len-(w+10)-w and x < self.control_len-(10+w): # Play
         self.pause( not self.ispaused() )
         return True
      elif x >= self.control_len-2*(w+10)-w and x < self.control_len-2*(10+w): # Speed up
         self.speed( self.playspeed * 1.1 )
         return True
      return False

   def mouseMove( self, x, y ):
      if not self.__drag:
         return
      # Check position
      x = x - self.control_pos[0]
      w = self.control_size
      if x >= 3*(w+10) and x < self.control_len-3*(w+10):
         x = x - 3*(w+10)
         w = self.control_len - 6*(w+10)
         p = x / w
         self.__curt = (self.__t[-1] - self.__t[0])*p + self.__t[0]
         self.setPosition( self.__curt )

   def setUnits( self, pos="m", time="s", vel="m/s" ):
      if pos=="m":
         self.__pos_val = 1.
      elif pos=="km":
         self.__pos_val = 1000.
      else:
         raise ValueError
      self.__pos_sym = pos
      if time=="s":
         self.__tim_val = 1.
      elif time=="min":
         self.__tim_val = 60.
      elif time=="h":
         self.__tim_val = 60.*60.
      elif time=="d":
         self.__tim_val = 24.*60.*60.
      elif time=="y":
         self.__tim_val = 365*24.*60.*60.
      else:
         raise ValueError
      self.__tim_sym = time
      self.__vel_sym = "%s/%s" % (self.__pos_sym, self.__tim_sym)
      if vel=="m/s":
         self.__vel_val = 1.
      elif vel=="km/s":
         self.__vel_val = 1000.
      elif vel=="km/h":
         self.__vel_val = 1000./3600.
      self.__vel_sym = vel
      if self.__axes:
         self.__axes.setUnits( self.__pos_val, self.__pos_sym )


   def update( self, dt ):
      "Updates the animation of the trajectory."
      # must be playing
      if self.playing:
         # Update
         self.__curt += self.playspeed * dt
         # Stop when finished
         if self.__curt > self.__t[ -1 ]:
            if self.__repeat:
               self.__curt = self.__t[0]
            else:
               self.__curt = self.__t[-1]
               self.playing = False
         self.setPosition( self.__curt )
      r, v, dv = self.__path.position( self.__curt )
      self.__curr  = r
      self.__curv  = v
      self.__curdv = dv

   def displayOver( self, width, height):
      "Displays the trajectory overlay."

      for pnt in self.__planets:
         pnt.displayOver( width, height )
      self.__path.displayOver( width, height )

      glEnable(GL_BLEND)
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

      t  = self.__curt
      r  = self.__curr
      v  = self.__curv
      dv = self.__curdv

      bx = 10.
      y  = 20. + 3.*( 1.5 *  self.fontsize )

      # Text to render
      st = "Time: %.2E %s" % (linalg.norm( t ) / self.__tim_val, self.__tim_sym )
      sr = "Dist: %.2E %s" % (linalg.norm( r ) / self.__pos_val, self.__pos_sym )
      sv = "Vel: %.2E %s" % (linalg.norm( v ) / self.__vel_val, self.__vel_sym )

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

      # Render controls
      if self.__controls:
         glBegin( GL_TRIANGLES )

         glColor3d( 1., 1., 1. )

         # Backwards button
         x = self.control_pos[0]
         y = self.control_pos[1]
         h = self.control_size
         self.control_len = width - x - 20 - w
         glVertex3d( x,       y+h/2, 0 )
         glVertex3d( x+h*0.7, y, 0 )
         glVertex3d( x+h*0.7, y+h, 0 )

         glVertex3d( x+h*0.9, y, 0 )
         glVertex3d( x+h*0.9, y+h, 0 )
         glVertex3d( x+h,     y, 0 )
         glVertex3d( x+h*0.9, y+h, 0 )
         glVertex3d( x+h,     y, 0 )
         glVertex3d( x+h,     y+h, 0 )
         x = x + 30

         # Stop button
         glVertex3d( x,       y, 0 )
         glVertex3d( x,       y+h, 0 )
         glVertex3d( x+h,     y, 0 )
         glVertex3d( x,       y+h, 0 )
         glVertex3d( x+h,     y, 0 )
         glVertex3d( x+h,     y+h, 0 )
         x = x + 30

         # Slow up Button
         glVertex3d( x+h*0.0, y+h/2, 0 )
         glVertex3d( x+h*0.5, y, 0 )
         glVertex3d( x+h*0.5, y+h, 0 )
         glVertex3d( x+h*0.5, y+h/2, 0 )
         glVertex3d( x+h*1.0, y, 0 )
         glVertex3d( x+h*1.0, y+h, 0 )
         x = x + 30

         # Position Bar
         w = self.control_len - 30*6
         glColor3d( 0.8, 0.8, 0.8 )
         glVertex3d( x,       y+h*0.2, 0 )
         glVertex3d( x,       y+h*0.8, 0 )
         glVertex3d( x+w,     y+h*0.2, 0 )
         glVertex3d( x,       y+h*0.8, 0 )
         glVertex3d( x+w,     y+h*0.2, 0 )
         glVertex3d( x+w,     y+h*0.8, 0 )

         p = (self.__curt - self.__t[0]) / (self.__t[-1] - self.__t[0])
         glColor3d( 1.0, 1.0, 1.0 )
         glVertex3d( x+w*p-2, y+h*0.0, 0 )
         glVertex3d( x+w*p-2, y+h*1.0, 0 )
         glVertex3d( x+w*p+2, y+h*0.0, 0 )
         glVertex3d( x+w*p-2, y+h*1.0, 0 )
         glVertex3d( x+w*p+2, y+h*0.0, 0 )
         glVertex3d( x+w*p+2, y+h*1.0, 0 )
         x = x + w + 10

         # Speed up button
         glVertex3d( x+h*0.5, y+h/2, 0 )
         glVertex3d( x+h*0.0, y, 0 )
         glVertex3d( x+h*0.0, y+h, 0 )
         glVertex3d( x+h*1.0, y+h/2, 0 )
         glVertex3d( x+h*0.5, y, 0 )
         glVertex3d( x+h*0.5, y+h, 0 )
         x = x + 30

         # Pause/play button
         if self.playing:
            # pause button
            glVertex3d( x+h*0.0, y, 0 )
            glVertex3d( x+h*0.0, y+h, 0 )
            glVertex3d( x+h*0.4, y, 0 )
            glVertex3d( x+h*0.0, y+h, 0 )
            glVertex3d( x+h*0.4, y, 0 )
            glVertex3d( x+h*0.4, y+h, 0 )
            
            glVertex3d( x+h*1.0, y, 0 )
            glVertex3d( x+h*1.0, y+h, 0 )
            glVertex3d( x+h*0.6, y, 0 )
            glVertex3d( x+h*1.0, y+h, 0 )
            glVertex3d( x+h*0.6, y, 0 )
            glVertex3d( x+h*0.6, y+h, 0 )
         else:
            # Play button
            glVertex3d( x+h*1.0, y+h/2, 0 )
            glVertex3d( x+h*0.0, y, 0 )
            glVertex3d( x+h*0.0, y+h, 0 )
         x = x + 30

         # Forward button
         glVertex3d( x+h*0.0, y, 0 )
         glVertex3d( x+h*0.0, y+h, 0 )
         glVertex3d( x+h*0.1, y, 0 )
         glVertex3d( x+h*0.0, y+h, 0 )
         glVertex3d( x+h*0.1, y, 0 )
         glVertex3d( x+h*0.1, y+h, 0 )

         glVertex3d( x+h*0.3, y, 0 )
         glVertex3d( x+h*0.3, y+h, 0 )
         glVertex3d( x+h*1.0, y+h/2, 0 )

         glEnd() # GL_TRIANGLES

      # Display axes if set
      if self.__axes:
         self.__axes.displayOver( width, height )



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

      glColor3d( *self.yColor )
      glVertex3d( 0, 0, 0 )
      glVertex3d( 0., self.expanse, 0 )

      glColor3d( *self.zColor )
      glVertex3d( 0, 0, 0 )
      glVertex3d( 0., 0., self.expanse )
      glEnd()


###############################################################################
class Axes(Object):
   """
   Shows the X/Y axes.
   """

   def __init__( self, font = None, origin = (0., 0., 0.), scale=1., track=[] ):
      self.__font = font
      self.setUnits()
      self.refresh( origin, scale, track )

   def refresh( self, origin, scale, track ):
      self.__origin  = origin
      self.__scale   = scale
      self.__track   = track

   def setFont( self, font ):
      self.__font = font

   def setUnits( self, pos_val=1., pos_sym="m" ):
      self.__pos_val = pos_val
      self.__pos_sym = pos_sym

   def displayOver( self, width, height ):
      empty    = 20.
      margin   = 35.

      glColor3d( 1., 1., 1. )
      glBegin(GL_LINES)

      # Print Base
      glVertex3d( margin, margin, 0. )
      glVertex3d( margin, height-margin, 0. )
      glVertex3d( margin, margin, 0. )
      glVertex3d( width-margin, margin, 0. )

      # Print coordinates
      vstep = 10. ** floor( math.log( 250.*(1./self.__scale), 10. ) )
      step  = vstep * self.__scale
      x     = self.__origin[0] % step
      while x < margin:
         x += step
      hx = x - step/2.
      if step > 25 and hx > margin:
         glVertex3d( hx, margin, 0. )
         glVertex3d( hx, (margin+empty)/2., 0. )
      while x < width-margin:
         glVertex3d( x, margin, 0. )
         glVertex3d( x, empty, 0. )
         if step > 25:
            hx = x + step/2.
            if hx < width-margin:
               glVertex3d( hx, margin, 0. )
               glVertex3d( hx, (margin+empty)/2., 0. )
         x += step
      y     = self.__origin[1] % step
      while y < margin:
         y += step
      hy = y - step/2.
      if step > 25 and hy > margin:
         glVertex3d( margin, hy, 0. )
         glVertex3d( (margin+empty)/2., hy, 0. )
      while y < height-margin:
         glVertex3d( margin, y, 0. )
         glVertex3d( empty, y, 0. )
         if step > 25:
            hy = y + step/2.
            if hy < height-margin:
               glVertex3d( margin, hy, 0. )
               glVertex3d( (margin+empty)/2., hy, 0. )
         y += step

      # Print tracking markers
      for t in self.__track:
         glColor3d( *t["colour"] )
         x = t["pos"][0]
         if x > margin and x < width-margin:
            glVertex3d( x, margin, 0. )
            glVertex3d( x, empty, 0. )
         # Print vertical
         y = t["pos"][1]
         if y > margin and y < height-margin:
            glVertex3d( margin, y, 0. )
            glVertex3d( empty, y, 0. )
      glEnd()

      # Render text - needs to be done outside of glBeign()
      glColor3d( 1., 1., 1. )
      x     = self.__origin[0] % step
      if x < margin:
         x += step
      while x < width-margin:
         s = "%.0f" % ((x - self.__origin[0]) / step)
         glRasterPos( x-self.__font.Advance(s)/2., 2. )
         self.__font.Render( s )
         x += step
      y     = self.__origin[1] % step
      if y < margin:
         y += step
      while y < height-margin:
         s = "%.0f" % ((y - self.__origin[1]) / step)
         glRasterPos( (margin-self.__font.Advance(s))/2., y )
         self.__font.Render( s )
         y += step

      # Display scale
      glRasterPos( 5., height - margin*0.75 )
      self.__font.Render( "%.0E %s" % (vstep / self.__pos_val, self.__pos_sym) )

