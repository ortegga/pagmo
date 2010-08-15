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
  
## @package alifegui
#  This module contains a simple GUI for the alife problem 
#
#  @author John Glover

# todo:
# - robot body sections/legs
# - load/save data
# - update robot when parameters change
# - launch evolve in separate thread?


from math import acos, pi, sqrt
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt4 import QtGui, QtCore
from PyQt4.QtOpenGL import QGLWidget
import Image
from ui.alifeui import Ui_ALife
from viewer import ALifeViewer
from environment import ALifeEnvironment
from task import ALifeExperiment
from robot import Robot
from asteroid import Asteroid
from task import ALifeExperiment, ALifeAgent, ALifeTask

##
class ALifeViewerWidget(QGLWidget, ALifeViewer):
    ## Constructor
    def __init__(self, parent, geometry, size_policy):
        QGLWidget.__init__(self, parent)
        self.setGeometry(geometry)
        self.setSizePolicy(size_policy)
        ##
        self._timer = QtCore.QTimer()
        ##
        self._timer.setSingleShot(False)
        ##
        self._timer.timeout.connect(self.step)
        ## @var environment ALifeEnvironment object
        self.environment = ALifeEnvironment()
        ## @var robot The robot that will interact with the environment
        self.robot = Robot(self.environment.world, self.environment.space, [0, 150, 0])
        self.environment.set_robot(self.robot)
        ## @var asteroid The asteroid 
        self.asteroid = Asteroid(self.environment.space, "models/asteroid_textured.x3d")
        self.environment.set_asteroid(self.asteroid)
        ##
        self.task = ALifeTask(self.environment)
        ##
        self.agent = ALifeAgent(len(self.task.getObservation()))
        ## @var experiment ALifeExperiment object
        self.experiment = ALifeExperiment(self.task, self.agent, self.environment) 
        ## @var _center_obj The ode object to center the camera view on
        self._center_obj = self.robot.body
        ## @var _center_on_obj Whether or not to center the camera view on self._center_obj
        self._center_on_obj = True
        ## @var _center_x x coordinate of the center of the current view point
        self._center_x = 0
        ## @var _center_y y coordinate of the center of the current view point
        self._center_y = 0
        ## @var _center_z z coordinate of the center of the current view point
        self._center_z = 0
        ## @var _mouse_view Whether or not the view point can be changed by the moving the mouse
        self._mouse_view = False
        ## @var _view_distance The initial distance between the camera and the robot
        self._view_distance = 20
        ## @var _last_x The last x coordinate of the camera
        self._last_x = -0.5
        ## @var _last_y The last y coordinate of the camera
        self._last_y = 1
        ## @var _last_z The last z coordinate of the camera
        self._last_z = -1
        ## @var asteroid_tex_id Texture ID for the asteroid
        self.asteroid_tex_id = None
        ## @var fps The number of frames to render per second 
        self._fps = 15
        ## @var _steps_per_frame The number of Environment/Experiment steps per frame rendered
        self._steps_per_frame = 3
        ## @var _dt The increment by which to progress (step) the Environment
        #  Only used for this if there is no self.experiment set, 
        #  otherwise experiment.update is called.
        self._dt = 1.0 / self._fps
        ## @var zoom_increment when zooming the viewing distance changes by 
        #  this amount at every step
        self.zoom_increment = 10.0
        ## @var _capture_screen Whether or not each frame should be saved (as a PNG image)
        self._capture_screen = False
        ## @var _screenshot_dir Directory to write image files to when taking screen shots
        self._screenshot_dir = "screenshots/"
        self.print_controls()
        
    def start(self):
        self._timer.start(self._fps)
        
    def pause(self):
        self._timer.stop()
        
    def step(self):
        for i in range(self._steps_per_frame):
            self.experiment.step()
        self.update()

    def paintGL(self):
        self._prepare()
        for (body, geom) in self.environment.get_objects():
            self._draw_object(body, geom)
        if self.asteroid:
            self._draw_asteroid(self.asteroid)
        
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        gluPerspective(40.0, float(w)/h, 1.0, 30.0)
    
    def initializeGL(self):
        # Initialize Viewport and Shading
        glViewport(0, 0, self.width(), self.height())
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

        # Enable material coloring
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        
        # Load asteroid textures
        if self.asteroid.texture_file:
            img = Image.open(self.asteroid.texture_file)
            raw_img = img.tostring()
            self.asteroid_tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.asteroid_tex_id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, img.size[0], img.size[1], 0, 
                         GL_LUMINANCE, GL_UNSIGNED_BYTE, raw_img)
       
    ## Store the mouse coordinates (relative to centre and normalised)
    #  the eye does not exactly move on a unit hemisphere; we fudge the projection
    #  a little by shifting the hemisphere into the ground by 0.1 units,
    #  such that approaching the perimeter does not cause a huge change in the
    #  viewing direction. The limit for l is thus cos(arcsin(0.1)).     
    def mouseMoveEvent(self, event):
        if not self._mouse_view:
            return 
        x1 = 3 * float(event.x()) / self.width() - 1.5
        z1 = -3 * float(event.y()) / self.height() + 1.5
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
        self.update()
            
    def mousePressEvent(self, event):
        self._mouse_view = True

    def mouseReleaseEvent(self, event):
        self._mouse_view = False
        
    def zoom_in(self):
        self._view_distance -= self.zoom_increment
        self.update()
        
    def zoom_out(self):
        self._view_distance += self.zoom_increment
        self.update()
        
    ## Prints the key/mouse controls for this viewer object
    def print_controls(self):
        print "Press the 'z' key to zoom in"
        print "Hold shift and press the 'z' key zoom out"
        print "Click and drag with the mouse to move the camera around the robot"
        
        
##
class ALifeGUI(QtGui.QMainWindow):
    ##
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        ##
        self.ui = Ui_ALife()
        ##
        self.ui.setupUi(self)
        size_policy = self.ui.main_view.sizePolicy()
        geometry = self.ui.main_view.frameGeometry()
        ##
        self.ui.main_view = ALifeViewerWidget(self.ui.centralwidget,
                                              geometry,
                                              size_policy)
        self.ui.main_view.setObjectName("main_view")
        self.connect(self.ui.start_pause, 
                     QtCore.SIGNAL("clicked()"), 
                     self.start_pause)
        self.connect(self.ui.restart, QtCore.SIGNAL("clicked()"), self.restart)
        
    ##
    def keyPressEvent(self, event):
        if event.text() == 'z':
            self.ui.main_view.zoom_in()
        elif event.text() == 'Z':
            self.ui.main_view.zoom_out()
            
    def start_pause(self):
        if self.ui.start_pause.text() == "Start Simulation":
            self.ui.start_pause.setText("Pause Simulation")
            self.ui.main_view.start()
        else:
            self.ui.start_pause.setText("Start Simulation")
            self.ui.main_view.pause()
            
    def restart(self):
        self.ui.main_view.experiment.reset()
        self.ui.main_view.update()
        

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    gui = ALifeGUI()
    gui.show()
    sys.exit(app.exec_())
