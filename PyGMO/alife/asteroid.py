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
import os.path
import ode

## Asteroid class
#
#  Creates an ODE geometry object called 'asteroid' in an XODE string
#
#  @author John Glover
class Asteroid(object):
    ## Constructor
    #  @param x3d_file The path to a X3D file that contains a model called 'Asteroid' (optional) 
    def __init__(self, space, x3d_file=None):
        ## @var space The ODE space that this asteroid will be added to
        self.space = space
        ## @var geom The asteroid ODE geometry object
        self.geom = None
        ## @var name The name of this object
        self.name = "Asteroid"
        ## @var mass The mass of the asteroid. Used to calculate the force of gravity between
        #  the asteroid and the robot
        self.mass = 10000.0
        ## @var _texture_coords Texture Coordinates. For each triangle in the mesh, there should
        #  be a 3-tuple of texture coordinates (s, t), so there is one (s, t) tuple for each vertex.
        self._texture_coords = []
        ## @var texture_file The path to the texture file, relative to the X3D file.
        #  This is automatically extracted from the X3D file if load_x3d is called.
        self.texture_file = ""
        ## @var x3d_file The path to a X3D file that contains a model called 'Asteroid'
        self.x3d_file = x3d_file
        if self.x3d_file:
            self.load_x3d(self.x3d_file)
    
    ## @return The texture coordinates for the given face (triangle) number
    def get_texture_coords(self, face_number):
        return self._texture_coords[face_number]
    
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
    #  - For textures, there must be one TextureCoordinate point for each vertex
    #    of each face, and they must be in the same order as the faces (given in
    #    the IndexedFaceSet coordIndex. The texCoordIndex field is currently ignored.
    #     
    #  @param file_name The file path to the .x3d file containing the asteroid model.
    def load_x3d(self, file_name):
        dom = md.parse(file_name)
        vertices = []
        triangles = []
        
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
                    
                    # get the path to the texture file
                    try:
                        shape_node = node.getElementsByTagName('Shape')[0]
                        app_node = shape_node.getElementsByTagName('Appearance')[0]
                        tex_node = app_node.getElementsByTagName('ImageTexture')[0]
                        tex_path = tex_node.attributes['url'].value
                        self.texture_file = os.path.join(os.path.dirname(file_name), tex_path)
                    except:
                        # no texture data found
                        pass
                                
                    # load trimesh data
                    if node.getElementsByTagName('IndexedFaceSet'):
                        ifs = node.getElementsByTagName('IndexedFaceSet')[0]
                        # form triangles from the coordIndex
                        coord_index = ifs.attributes['coordIndex'].value
                        for face in coord_index.split(','):
                            # make sure that the given faces are triangles
                            # there should 4 indicies, the last one equal to -1
                            indicies = face.split()
                            if len(indicies) == 4 and int(indicies[3]) == -1:
                                triangles.append((int(indicies[0]),
                                                  int(indicies[1]),
                                                  int(indicies[2])))                                
                        # form vertices from the Coordinate point attribute
                        coordinate = ifs.getElementsByTagName('Coordinate')[0]
                        coord_points = coordinate.attributes['point'].value
                        for points in coord_points.split(','):
                            # each vertex should have 3 points
                            points = points.split()
                            if len(points) == 3:
                                vertices.append((float(points[0]) * scale[0],
                                                 float(points[1]) * scale[1],
                                                 float(points[2]) * scale[2]))
                                
                        # get texture coordinate points
                        if ifs.getElementsByTagName('TextureCoordinate'):
                            tex_coordinate = ifs.getElementsByTagName('TextureCoordinate')[0]
                            tex_coord_points = tex_coordinate.attributes['point'].value
                            points = tex_coord_points.split(',')
                            self._texture_coords = []
                            for i in range(len(points)/3):
                                # there should be 1 coordinate for each vertex of each face
                                # each coordinate should have 2 values (s,t)
                                point1 = points[i*3].split()
                                point2 = points[(i*3)+1].split()
                                point3 = points[(i*3)+2].split()
                                if len(point1) == len(point2) == len(point3) == 2:
                                    tex_coords = ((float(point1[0]), float(point1[1])), 
                                                  (float(point2[0]), float(point2[1])),
                                                  (float(point3[0]), float(point3[1])))
                                    self._texture_coords.append(tex_coords)
                        break
                    
        # make ODE TriMesh object
        data = ode.TriMeshData()
        data.build(vertices, triangles)
        self.geom = ode.GeomTriMesh(data=data, space=self.space)
        self.geom.name = self.name
