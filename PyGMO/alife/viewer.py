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
import Image
from math import acos, pi, sqrt
import time
import os


## ALifeViewer class
#  
#  It renders the scene created by its ALifeEnvironment object using OpenGL.
#  Code is based on the ode.viewer module in PyBrain
#
#  @author John Glover
class ALifeViewer(object):
    def __init__(self, env, exp=None):
        ## @var env ALifeEnvironment object
        self.env = env
        ## @var exp ALifeExperiment object
        self.exp = exp
        ## @var width viewport width
        self.width = 800
        ## @var height viewport height
        self.height = 600  
        ## @var _center_obj The ode object to center the camera view on
        self._center_obj = env.robot.body
        ## @var _center_on_obj Whether or not to center the camera view on self._center_obj
        self._center_on_obj = True
        ## @var _center_x x coordinate of the center of the current view point
        self._center_x = 0
        ## @var _center_y y coordinate of the center of the current view point
        self._center_y = 0
        ## @var _center_z z coordinate of the center of the current view point
        self._center_z = 0
        ## @var _mouse_view Whether or not the view point can be changed by the moving the mouse
        self._mouse_view = True
        ## @var _view_distance The initial distance between the camera and the robot
        self._view_distance = 100
        ## @var _last_x The last x coordinate of the camera
        self._last_x = -0.5
        ## @var _last_y The last y coordinate of the camera
        self._last_y = 1
        ## @var _last_z The last z coordinate of the camera
        self._last_z = -1
        ## @var asteroid_tex_id Texture ID for the asteroid
        self.asteroid_tex_id = None
        ## @var fps The number of frames to render per second 
        self._fps = 25
        ## @var _steps_per_frame The number of Environment/Experiment steps per frame rendered
        self._steps_per_frame = 2
        ## @var _dt The increment by which to progress (step) the Environment
        #  Only used for this if there is no self.experiment set, 
        #  otherwise experiment.update is called.
        #  This value is also used to control the FPS rate by calling sleep() in the
        #  GLUT idle callback
        self._dt = 1.0 / self._fps
        ## @var _last_time The value of time.time at the last check
        self._last_time = time.time()
        ## @var zoom_increment when zooming the viewing distance changes by 
        #  this amount at every step
        self.zoom_increment = 10.0
        ## @var _capture_screen Whether or not each frame should be saved (as a PNG image)
        self._capture_screen = False
        ## @var _screenshot_dir Directory to write image files to when taking screen shots
        self._screenshot_dir = "screenshots/"
        # initialise OpenGL
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
        
        # load asteroid textures
        if self.env.asteroid:
            if self.env.asteroid.texture_file:
                img = Image.open(self.env.asteroid.texture_file)
                raw_img = img.tostring()
                self.asteroid_tex_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self.asteroid_tex_id)
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, img.size[0], img.size[1], 0, 
                             GL_LUMINANCE, GL_UNSIGNED_BYTE, raw_img)
        
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

        # keep camera pointed at a specific object)
        if self._center_on_obj and self._center_obj:
            (self._center_x, self._center_y, self._center_z) = self._center_obj.getPosition()

        # use the mouse to shift eye sensor on a hemisphere
        eyeX = self._view_distance * self._last_x
        eyeY = self._view_distance * self._last_y + self._center_y
        eyeZ = self._view_distance * self._last_z
        gluLookAt(eyeX, eyeY, eyeZ, self._center_x, self._center_y, self._center_z, 0, 1, 0)
        
    ## Draw an ODE object.
    #  @param body The body object.
    #  @param geom The body geometry object.
    def _draw_object(self, body, geom):
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
            if type(geom) == ode.GeomPlane:
                # set color of plane
                glColor3f(0.53, 0.44, 0.35)
    
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
                gluDisk(quad, 0, 200, 200, 1)
                glPopMatrix()
                
            elif type(geom) == ode.GeomSphere:
                glColor3f(0.53, 0.44, 0.35)
                quad = gluNewQuadric()
                gluQuadricTexture(quad, GL_TRUE)
                glPushMatrix()
                gluSphere(quad, geom.getRadius(), 32, 32);
                glPopMatrix()
        glPopMatrix()
        
    ## Draw the Asteroid object.
    #  The asteroid is drawn separately for now so that it can be textured
    #  @param asteroid An object of type Asteroid
    def _draw_asteroid(self, asteroid):
        # assume asteroid.geom is of type ode.GeomTriMesh
        # note: geom.getTriangleCount seems to be undocumented, can't find
        #       it in the API anywhere, just stumbled across it on the 
        #       pyode-user mailing list
        if self.asteroid_tex_id:
            # setup the texture if one exists
            glEnable(GL_BLEND)
            glColor(1, 1, 1)
            #glColor3f(0.53, 0.44, 0.35)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.asteroid_tex_id)
            # draw with texture coordinates
            glPushMatrix()
            glBegin(GL_TRIANGLES)
            for i in range(asteroid.geom.getTriangleCount()):
                triangle = asteroid.geom.getTriangle(i)
                texture = asteroid.get_texture_coords(i)
                glTexCoord2fv(texture[0])
                glVertex3fv(triangle[0])
                glTexCoord2fv(texture[1])
                glVertex3fv(triangle[1])
                glTexCoord2fv(texture[2])
                glVertex3fv(triangle[2])
            glEnd()
            glPopMatrix()
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            
        # if no texture, just draw the mesh
        glPushMatrix()
        glColor3f(0.53, 0.44, 0.35)
        glBegin(GL_TRIANGLES)
        for i in range(asteroid.geom.getTriangleCount()):
            triangle = asteroid.geom.getTriangle(i)
            glVertex3fv(triangle[0])
            glVertex3fv(triangle[1])
            glVertex3fv(triangle[2])
        glEnd()
        glPopMatrix()
        
    ## The drawing callback function.
    #  Prepares the screen for drawing, then goes through every ODE object
    #  and renders it.
    def _draw(self):
        self._prepare()
        # Call step on the experiment first if one exists. This will perform
        # its own calculations then step the environment for us.
        if self.exp:
            for i in range(self._steps_per_frame):
                self.exp.step()
        # If no experiment exists, step the environment directly
        else:
            for i in range(self._steps_per_frame):
                self.env.step(self._dt)
        # Draw all objects in the environment
        for (body, geom) in self.env.get_objects():
            self._draw_object(body, geom)
        if self.env.asteroid:
            self._draw_asteroid(self.env.asteroid)
        glutSwapBuffers()
        if self._capture_screen:
            self.screenshot()
    
    ## The idle callback function.
    #  Calculates how long to sleep to achieve the target frame rate, then tells GLUT
    #  to draw a new frame.
    def _idle(self):
        t = self._dt - (time.time() - self._last_time)
        if (t > 0):
            time.sleep(t)
        self._last_time = time.time()
        glutPostRedisplay() 

    ## The keyboard callback function.
    #  @param key The key that was pressed
    def _key_pressed(self, key, x, y):
        if key == 's':
            self._capture_screen = not self._capture_screen
            print "Screen Capture: " + (self._capture_screen and "on" or "off")
        if key == 'c':        
            self._center_on_obj = not self._center_on_obj
            print "Centering camera: " + (self._center_on_obj and "on" or "off")
        if key in ['x', 'q']:
            sys.exit()
        if key == 'v':
            self._mouse_view = not self._mouse_view
            print "Mouse view:" + (self._mouse_view and "on" or "off")
            
    ## Callback function for 'special' keys
    #  Up and down arrow keys are used for zooming in and out respectively
    #  @param key The key that was pressed     
    def _special_func(self, key, x, y):
        if key == GLUT_KEY_UP:
            self._view_distance -= self.zoom_increment
        elif key == GLUT_KEY_DOWN:
            self._view_distance += self.zoom_increment
    
    ## Control the zoom factor
    def _motion(self, x, z):
        if not self._mouse_view: return
        zn = 2.75 * float(z) / self.height + 0.25   # [0.25,3]
        self._view_distance = 3.0 * zn * zn
        self._passive_motion(x, z)
    
    ## Store the mouse coordinates (relative to centre and normalised)
    #  the eye does not exactly move on a unit hemisphere; we fudge the projection
    #  a little by shifting the hemisphere into the ground by 0.1 units,
    #  such that approaching the perimeter does not cause a huge change in the
    #  viewing direction. The limit for l is thus cos(arcsin(0.1)).
    def _passive_motion(self, x, z):
        if not self._mouse_view: return
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
        self._last_y = y1
        self._last_x = x1
        self._last_z = z1
        
    ## Start the main GLUT rendering loop
    def start(self):
        glutMainLoop()
        
    ## Save the current frame to a png image
    #  The path is <self._screenshot_dir><img_number>.png
    #  self._screenshot_dir is automatically created if it does not exist.
    #  Shots are automatically numerated based on how many files are already 
    #  in the directory.
    def screenshot(self):
        if not os.path.exists(self._screenshot_dir):
            os.makedirs(self._screenshot_dir)
        
        num_present = len(os.listdir(self._screenshot_dir))
        num_digits = len(str(num_present))
        index = '0' * (5 - num_digits) + str(num_present) 
        
        path = os.path.join(self._screenshot_dir, index + '.png')
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.fromstring("RGB", (self.width, self.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(path, 'png')

    ## Prints the key/mouse controls for this viewer object
    def print_controls(self):
        print "Press 'q' to exit"
        print "Press 'c' to toggle camera centering on the robot on/off"
        print "Press 's' to toggle screen capturing on/off"
        print "Press 'v' to toggle mouse view on/off"
        print "Press the up arrow to zoom in"
        print "Press the down arrow to zoom out"
        print "Move the mouse to move the camera around the robot"
        
        
if __name__ == "__main__":  
    from environment import ALifeEnvironment
    from robot import Robot
    from asteroid import Asteroid
    
    env = ALifeEnvironment()
    robot_position = [0, 150, 0]
    robot = Robot(env.world, env.space, robot_position)
    env.set_robot(robot)
    asteroid = Asteroid(env.space, "models/asteroid_textured.x3d")
    env.set_asteroid(asteroid)
    viewer = ALifeViewer(env)
    viewer.print_controls()
    viewer.start()
