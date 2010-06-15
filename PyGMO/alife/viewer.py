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
  
## @package viewer
#  This module contains the ALifeViewer class. 
#  It renders the scene created by its ALifeEnvironment object using OpenGL.
#  Code is based on the ode.viewer module in PyBrain
from pybrain.rl.environments.ode.tools.mathhelpers import crossproduct, norm, dotproduct
import ode
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import acos, pi, sqrt
import time

## ALifeViewer class
#  
#  It renders the scene created by its ALifeEnvironment object using OpenGL.
#  Code is based on the ode.viewer module in PyBrain
#
#  @author John Glover
class ALifeViewer(object):
    def __init__(self):
        ## @var env ALife Environment object
        self.env = None
        ## @var width viewport width
        self.width = 800
        ## @var height viewport height
        self.height = 600  
        
        # initialize object which the camera follows
        self.centerObj = None
        self.mouseView = True
        self.viewDistance = 300
        self.lastx = -0.5
        self.lasty = 1
        self.lastz = -1

        ## @var fps the number of frames to render per second 
        self.fps = 25
        self.dt = 1.0 / self.fps
        self.lasttime = time.time()
        self.starttime = self.lasttime
        
        ## @var zoom_increment when zooming the viewing distance changes by 
        #  this amount at every step
        self.zoom_increment = 10.0

        # init OpenGL
        self._init()  

        # set callback functions
        glutMotionFunc(self._motion)
        glutPassiveMotionFunc(self._passive_motion)
        glutDisplayFunc(self._draw) 
        glutIdleFunc(self._idle)
        glutKeyboardFunc(self._key_pressed)
        glutSpecialFunc(self._special_func)
        
    ## Initialise OpenGL. This function has to be called only once before drawing.
    #  @param width The width of the GLUT window.
    #  @param height The height of the GLUT window.  
    def _init(self, width=800, height=600):
        glutInit([])

        # Open a window
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
        self.width = width
        self.height = height
        glutInitWindowPosition(500, 0)
        glutInitWindowSize(self.width, self.height)
        self._window = glutCreateWindow("ALife Viewer")

        # Initialize Viewport and Shading
        glViewport(0, 0, self.width, self.height)
        glShadeModel(GL_SMOOTH)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glClearColor(1.0, 1.0, 1.0, 0.0)

        # Initialize Depth Buffer
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Initialize Lighting
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 5.0, 5.0, 1.0])
        glEnable(GL_LIGHT1)

        # enable material coloring
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        
    ## Prepare drawing. This function is called in every step. 
    #  It clears the screen and sets the new camera position
    def _prepare(self):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Projection mode
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1.3333, 0.2, 500)

        # Initialize ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # View transformation (if "centerOn(...)" is set, keep camera to specific object)
        if self.centerObj is not None:
            (centerX, centerY, centerZ) = self.centerObj.getPosition()
        else:
            centerX = centerY = centerZ = 0
        # use the mouse to shift eye sensor on a hemisphere
        eyeX = self.viewDistance * self.lastx
        eyeY = self.viewDistance * self.lasty + centerY
        eyeZ = self.viewDistance * self.lastz
        gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, 0, 1, 0)
        
    ## Draw an ODE object.
    #  @param body The body object.
    #  @param geom The body geometry object.
    def _draw_object(self, body, geom):
        glDisable(GL_TEXTURE_2D)
        glPushMatrix()
        
        if body != None:
            # set the colour
            if hasattr(body, 'color'):
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(*(body.color))
            else: 
                glColor3f(0.1, 0.1, 0.1)

            # transform (rotate, translate) body accordingly
            (x, y, z) = body.getPosition()
            R = body.getRotation() 
            rot = [R[0], R[3], R[6], 0.0,
                   R[1], R[4], R[7], 0.0,
                   R[2], R[5], R[8], 0.0,
                      x, y, z, 1.0]
            glMultMatrixd(rot)
            
            # switch different geom objects
            if type(geom) == ode.GeomBox:
                (sx, sy, sz) = geom.getLengths()
                glScaled(sx, sy, sz)
                glutSolidCube(1)
                
            elif type(geom) == ode.GeomSphere:
                glutSolidSphere(geom.getRadius(), 20, 20)

            elif type(geom) == ode.GeomCCylinder:
                # capped cylinder
                radius = geom.getParams()[0]
                length = geom.getParams()[1] - 2 * radius
                quad = gluNewQuadric()
                # draw cylinder and two spheres, one at each end
                glTranslate(0.0, 0.0, -length / 2)
                gluCylinder(quad, radius, radius, length, 32, 32)
                glutSolidSphere(radius, 20, 20)
                glTranslate(0.0, 0.0, length)
                glutSolidSphere(radius, 20, 20)

            elif type(geom) == ode.GeomCylinder:
                # solid cylinder
                radius = geom.getParams()[0]
                length = geom.getParams()[1]
                glTranslate(0.0, 0.0, -length / 2)
                quad = gluNewQuadric()
                gluDisk(quad, 0, radius, 32, 1)
                quad = gluNewQuadric()
                gluCylinder(quad, radius, radius, length, 32, 32)
                glTranslate(0.0, 0.0, length)
                quad = gluNewQuadric()
                gluDisk(quad, 0, radius, 32, 1)    
                
        else:
            # no body found, then it must be a plane or triangular mesh
            if type(geom) == ode.GeomPlane:
                # set color of plane (currently green)
                glColor3f(0.2, 0.6, 0.3)
    
                # for planes, we need a Quadric object
                quad = gluNewQuadric()
                gluQuadricTexture(quad, GL_TRUE)
    
                p = geom.getParams()[0] # the normal vector to the plane
                d = geom.getParams()[1] # the distance to the origin
                q = (0.0, 0.0, 1.0)     # the normal vector of default gluDisks (z=0 plane)
    
                # calculate the cross product to get the rotation axis
                c = crossproduct(p, q)
                # calculate the angle between default normal q and plane normal p
                theta = acos(dotproduct(p, q) / (norm(p) * norm(q))) / pi * 180
    
                # rotate the plane
                glPushMatrix()
                glTranslate(d * p[0], d * p[1], d * p[2])
                glRotate(-theta, c[0], c[1], c[2])
                gluDisk(quad, 0, 20, 20, 1)
                glPopMatrix()
                
            elif type(geom) == ode.GeomTriMesh:
                # note: geom.getTriangleCount seems to be undocumented, can't find
                #       it in the API anywhere, just stumbled across it on the 
                #       pyode-user mailing list
                glPushMatrix()
                glColor3f(0.53, 0.44, 0.35)
                glBegin(GL_TRIANGLES)
                for i in range(geom.getTriangleCount()):
                    triangle = geom.getTriangle(i)
                    glVertex3fv(triangle[0])
                    glVertex3fv(triangle[1])
                    glVertex3fv(triangle[2])
                glEnd()
                glPopMatrix()
        glPopMatrix()
        
    ## The drawing callback function.
    #  Prepares the screen for drawing, then goes through every ODE object
    #  and renders it.
    def _draw(self):
        """ draw callback function """
        # Draw the scene
        self._prepare()
        
        if self.env:
            self.env.step(self.dt)
            for (body, geom) in self.env.get_objects():
                self._draw_object(body, geom)
        
        glutSwapBuffers()
#        if self.captureScreen:
#            self._screenshot()  
    
    ## The idle callback function.
    #  Calculates how long to sleep to achieve the target frame rate, then tells GLUT
    #  to draw a new frame.
    def _idle(self):
        t = self.dt - (time.time() - self.lasttime)
        if (t > 0):
            time.sleep(t)
        self.lasttime = time.time()
        glutPostRedisplay() 

    ## The keyboard callback function.
    #  @param key The key that was pressed
    def _key_pressed(self, key, x, y):
        if key == 's':
            self.setCaptureScreen(not self.getCaptureScreen())
            print "Screen Capture: " + (self.getCaptureScreen() and "on" or "off")
        if key in ['x', 'q']:
            sys.exit()
        if key == 'v':
            self.mouseView = not self.mouseView
            
    ## Callback function for 'special' keys
    #  Up and down arrow keys are used for zooming in and out respectively
    #  @param key The key that was pressed     
    def _special_func(self, key, x, y):
        if key == GLUT_KEY_UP:
            self.viewDistance -= self.zoom_increment
        elif key == GLUT_KEY_DOWN:
            self.viewDistance += self.zoom_increment
    
    ## Control the zoom factor
    def _motion(self, x, z):
        if not self.mouseView: return
        zn = 2.75 * float(z) / self.height + 0.25   # [0.25,3]
        self.viewDistance = 3.0 * zn * zn
        self._passive_motion(x, z)
    
    ## Store the mouse coordinates (relative to centre and normalised)
    #  the eye does not exactly move on a unit hemisphere; we fudge the projection
    #  a little by shifting the hemisphere into the ground by 0.1 units,
    #  such that approaching the perimeter does not cause a huge change in the
    #  viewing direction. The limit for l is thus cos(arcsin(0.1)).
    def _passive_motion(self, x, z):
        if not self.mouseView: return
        x1 = 3 * float(x) / self.width - 1.5
        z1 = -3 * float(z) / self.height + 1.5
        lsq = x1 * x1 + z1 * z1
        l = sqrt(lsq)
        if l > 0.994987:
            # for mouse outside window, project onto the unit circle
            x1 = x1 / l
            z1 = z1 / l
            y1 = 0
        else:
            y1 = max(0.0, sqrt(1.0 - x1 * x1 - z1 * z1) - 0.1)
        self.lasty = y1
        self.lastx = x1
        self.lastz = z1

    ## Sets the ODE environment to be rendered
    #  @param env The ALifeEnvironment object
    def set_environment(self, env):
        self.env = env
        self.centerObj = env.get_robot_body()
        
    ## Start the main GLUT rendering loop
    def start(self):
        glutMainLoop()

if __name__ == "__main__":  
    from environment import ALifeEnvironment, Robot
    import random
    random.seed()
    print "Starting ALife"
    print "Press q to exit"
    print "Press the up arrow to zoom in"
    print "Press the down arrow to zoom out"
    print "Move the mouse to move the camera around the robot"
    e = ALifeEnvironment()
    robot_position = [random.randint(-100, 100), 150, 0]
    r = Robot("Robot", robot_position)
    e.load_robot(r.get_xode())
    e.load_asteroid("models/asteroid.x3d")
    v = ALifeViewer()
    v.set_environment(e)
    v.start()