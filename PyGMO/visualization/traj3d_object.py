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
      "Initializes the object."
      if self.__class__ is Object:
         raise NotImplementedError

   def size( self ):
      "Returns the size of the object."
      if self.__class__ is Object:
         raise NotImplementedError

   def setScale( self, zoom ):
      "Sets the global zoom in 'unites per pixel'"
      if self.__class__ is Object:
         raise NotImplementedError

   def center( self ):
      "Returns the center of the object."
      if self.__class__ is Object:
         raise NotImplementedError

   def mouseDown( self, button, x, y ):
      "Handles mouse down events."
      if self.__class__ is Object:
         raise NotImplementedError
      return False

   def mouseUp( self, button, x, y ):
      "Handles mouse up events."
      if self.__class__ is Object:
         raise NotImplementedError
      return False

   def mouseMove( self, x, y ):
      "Handles mouse movement events."
      if self.__class__ is Object:
         raise NotImplementedError

   def display( self ):
      "Draws the object on the window."
      if self.__class__ is Object:
         raise NotImplementedError

   def displayOver( self, width, height ):
      "Draws the object overlay."
      if self.__class__ is Object:
         raise NotImplementedError

   def update( self, dt ):
      "Updates the object."
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


