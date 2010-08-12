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

from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt4 import QtGui, QtCore
from PyQt4.QtOpenGL import QGLWidget
from ui.alifeui import Ui_ALife


class ALifeGUI(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_ALife()
        self.ui.setupUi(self)
        #size_policy = self.ui.plot_window.sizePolicy()
        #geometry = self.ui.plot_window.frameGeometry()
        #self.ui.main_view = ViewerWidget(self.ui.centralwidget,
        #                                 window_geometry,
        #                                 window_size_policy)
        #self.ui.main_view.setObjectName("main_view")

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    gui = ALifeGUI()
    gui.show()
    sys.exit(app.exec_())
