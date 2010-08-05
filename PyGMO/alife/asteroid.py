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
  
## @package asteroid
#  This module contains the Asteroid classes.
#
#  Each Asteroid creates a XODE string that is returned by the
#  get_xode() method.
#
#  The XODE string must contain a geometry node called 'asteroid'.
#  This is all that is retained by the environment, the rest is ignored.
#  If the XODE string contains more than one node called 'asteroid', only
#  the first one is used.
import xml.dom.minidom as md

## Asteroid class
#
#  Creates an ODE geometry object called 'asteroid' in an XODE string
#
#  @author John Glover
class Asteroid(object):
    ## Constructor
    #  @param x3d_file The path to a X3D file that contains a model called 'Asteroid' (optional) 
    def __init__(self, x3d_file=None):
        ## @var xode_string The XODE data for this Asteroid
        self.xode_string = ""
        ## @var x3d_file The path to a X3D file that contains a model called 'Asteroid'
        self.x3d_file = x3d_file
        if self.x3d_file:
            self.load_x3d(self.x3d_file)
            
    ## @return The XODE data for this Asteroid
    def get_xode(self):
        return self.xode_string
    
    ## Loads the asteroid X3D file (XML) and parses it. The resulting
    #  geometry is a XODE trimesh object. The geometry is encoded in a valid
    #  XODE file, and the geometry node is called 'asteroid'.
    #
    #  X3D file restraints:
    #  - The file must contain at least 1 object called 'Asteroid'. If not,
    #    an exception is raised
    #  - If more than 1 object called 'Asteroid' exists, only the first one
    #    is processed, the rest are ignored.
    #  - The file must consist of faces stored as triangles. If the file 
    #    cannot be read or it does not contain any triangles an exception
    #    is raised.
    #     
    #  @param file_name The file path to the .x3d file containing the asteroid model.
    def load_x3d(self, file_name):
        dom = md.parse(file_name)
        root = dom.createElement('xode')
        root.attributes['version'] = '1.0r23'
        root.attributes['name'] = 'alife'
        root.attributes['xmlns:xsi'] = 'http://www.w3.org/2001/XMLSchema-instance'
        root.attributes['xsi:noNamespaceSchemaLocation'] = 'http://tanksoftware.com/xode/1.0r23/xode.xsd'
        world = dom.createElement('world')
        root.appendChild(world)
        space = dom.createElement('space')
        world.appendChild(space)
        
        geom = dom.createElement('geom')
        geom.attributes['name'] = 'Asteroid'
        trimesh = dom.createElement('trimesh')
        geom.appendChild(trimesh)
        trimesh_triangles = dom.createElement('triangles')
        trimesh.appendChild(trimesh_triangles)
        trimesh_vertices = dom.createElement('vertices')
        trimesh.appendChild(trimesh_vertices)
        space.appendChild(geom)
        
        for node in dom.getElementsByTagName('Transform'):
            if 'DEF' in node.attributes.keys():
                # take the first transform node defined as 'Asteroid'
                if node.attributes['DEF'].value == 'Asteroid':
                    # get scale information from the model
                    if 'scale' in node.attributes.keys():
                        scale_string = node.attributes['scale'].value
                        scale = scale_string.split()
                        scale = [float(s) for s in scale]
                    else:
                        scale = (1, 1, 1)
                        
                    # todo: get translation information from the model
                    # todo: get rotation information from the model
                    
                    if node.getElementsByTagName('IndexedFaceSet'):
                        ifs = node.getElementsByTagName('IndexedFaceSet')[0]
                        # form triangles from the coordIndex
                        coord_index = ifs.attributes['coordIndex'].value
                        for face in coord_index.split(','):
                            # make sure that the given faces are triangles
                            # there should 4 indicies, the last one equal to -1
                            indicies = face.split()
                            if len(indicies) == 4 and int(indicies[3]) == -1:
                                # x3d indices count from zero but xode indices
                                # count form one, so add one to each index value
                                t = dom.createElement('t')
                                t.attributes['ia'] = str(int(indicies[0])+1)
                                t.attributes['ib'] = str(int(indicies[1])+1)
                                t.attributes['ic'] = str(int(indicies[2])+1)
                                trimesh_triangles.appendChild(t)
                        # form vertices from the Coordinate point attribute
                        coordinate = ifs.getElementsByTagName('Coordinate')[0]
                        coord_points = coordinate.attributes['point'].value
                        for points in coord_points.split(','):
                            # each vertex should have 3 points
                            points = points.split()
                            if len(points) == 3:
                                v = dom.createElement('v')
                                v.attributes['x'] = str(float(points[0]) * scale[0])
                                v.attributes['y'] = str(float(points[1]) * scale[1])
                                v.attributes['z'] = str(float(points[2]) * scale[2])
                                trimesh_vertices.appendChild(v)
                        break
        self.xode_string = root.toxml()