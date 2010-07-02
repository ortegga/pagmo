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


import math


class Vector:
  
   def __init__(self, data):
      self.data = data
      if type(data).__name__ != 'tuple':
         raise TypeError
      for k in data:
         if type(k).__name__ != 'float':
            raise TypeError
      if len(data) != 3:
         raise AssertionError

   def __repr__(self):
      return repr(self.data)  

   def __add__(self, other):
      return Vector( ( self[0]+other[0], self[1]+other[1], self[2]+other[2]) )

   def __getitem__(self, index):
      return self.data[index]

   def __len__(self):
      return len(self.data)

   def __mul__(self, other):
      return Vector( (self[0]*other[0], self[1]*other[1], self[2]*other[2]) )

   def mod(self):
      return math.sqrt( self.mod2() )

   def mod2(self):
      return sum( map(lambda x: x*x, self) )

   def dot(self, other):
      a = self
      b = other
      return Vector( (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]) )

