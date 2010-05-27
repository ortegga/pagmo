from pybrain.rl.environments.ode.viewer import ODEViewer
from OpenGL.GL import (glPushMatrix, glPopMatrix, glBegin, glEnd,
                       GL_TRIANGLES, glVertex3fv, glColor3f)

class ALifeViewer(ODEViewer):
    def __init__(self):
        super(ALifeViewer, self).__init__()
        
    def draw_item(self, item):
        if item['type'] in ['GeomBox', 'GeomSphere', 'GeomCylinder', 'GeomCCylinder', 'GeomPlane']:
            # call parent object's draw_item
            ODEViewer.draw_item(self, item)
        elif item['type'] == 'GeomTriMesh':
            # todo: drawing a triangular mesh can be done more efficiently using
            #       vertex arrays
            glPushMatrix()
            glColor3f(0.53, 0.44, 0.35)
            glBegin(GL_TRIANGLES)
            for triangle in item['triangles']:
                glVertex3fv(triangle[0])
                glVertex3fv(triangle[1])
                glVertex3fv(triangle[2])
            glEnd()
            glPopMatrix()

if __name__ == "__main__":
    # press 'x' or 'q' to exit
    print "ALife viewer started"
    print "Press q or x to exit"
    # press 's' to take a screenshot
    viewer = ALifeViewer()
    viewer.start()