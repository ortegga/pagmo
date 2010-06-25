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


from OpenGL.GL   import *
from OpenGL.GLU  import *
from OpenGL.GLUT import *


"""
"""
class traj3d:

   def __init__( self, title="3D Trajectory Visualizer", width=640, height=480 ):
      """
      Initalizes.
      """

      # Initialize GLUT
      glutInit()

      # Set variables
      self.width  = width
      self.height = height
      self.title  = title

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
      glutDisplayFunc(     self.display )
      #glutIdleFunc(        self.__idle )
      glutReshapeFunc(     self.reshape )
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
      glutMainLoop()


   def clear( self ):
      glClear( GL_COLOR_BUFFER_BIT )


   def flush( self ):
      glFlush()
      glutSwapBuffers()


   def display( self ):
      self.clear()
      self.flush()


   def reshape( self, width=640, height=480 ):
      """
      Resizes the window.
      """
      glViewport( 0, 0, width, height )
      glMatrixMode( GL_PROJECTION )
      glLoadIdentity()
      gluPerspective( 60.0, float(width)/height, .1, 1000. )
      glMatrixMode( GL_MODELVIEW )
      glLoadIdentity()




