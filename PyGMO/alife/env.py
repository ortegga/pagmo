from pybrain.rl.environments.ode import ODEEnvironment, sensors, actuators
from pybrain.utilities import threaded
from scipy import array
import ode
import xode.parser
import time
import xml.dom.minidom as md

class Triangle(object):
    def __init__(self):
        ia = "0"
        ib = "0"
        ic = "0"
        
class Vertex(object):
    def __init__(self):
        x = "0"
        y = "0"
        z = "0"
        
class ALifeEnvironment(ODEEnvironment):
    def __init__(self, renderer=True, realtime=False, ip="127.0.0.1", 
                 port="21590", buf='16384'):
        ODEEnvironment.__init__(self, renderer, realtime, ip, port, buf)
        # load model file
        self.loadXODE("alife.xode")
        self.loadAsteroid("asteroid.x3d")

        # standard sensors and actuators    
#        self.addSensor(sensors.JointSensor())
#        self.addSensor(sensors.JointVelocitySensor()) 
#        self.addActuator(actuators.JointActuator())
            
        #set act- and obsLength, the min/max angles and the relative max touques of the joints  
        self.actLen = self.indim
        self.obsLen = len(self.getSensors())
        
#        self.tourqueList = array([0.2],)
#        self.cHighList = array([1.0],)
#        self.cLowList = array([-0.5],)        

        self.stepsPerAction = 1
        
    def loadAsteroid(self, asteroid_file):
        triangles = []
        vertices = []

        # parse asteroid x3d file, building the list of triangles and vertices
        dom = md.parse(asteroid_file)
        for node in dom.getElementsByTagName('Transform'):
            if 'DEF' in node.attributes.keys():
                if node.attributes['DEF'].value == 'Asteroid':
                    # take the first transform node defined as 'Asteroid'
                    if node.getElementsByTagName('IndexedFaceSet'):
                        ifs = node.getElementsByTagName('IndexedFaceSet')[0]
                        # form triangles from the coordIndex
                        coord_index = ifs.attributes['coordIndex'].value
                        for face in coord_index.split(','):
                            # make sure that the given faces are triangles
                            # there should 4 indicies, the last one equal to -1
                            indicies = face.split()
                            if len(indicies) == 4 and int(indicies[3]) == -1:
                                t = Triangle()
                                # x3d indices count from zero but xode indices
                                # count form one, so add one to each index value
                                t.ia = str(int(indicies[0])+1)
                                t.ib = str(int(indicies[1])+1)
                                t.ic = str(int(indicies[2])+1)
                                triangles.append(t)
                        # form vertices from the Coordinate point attribute
                        coordinate = ifs.getElementsByTagName('Coordinate')[0]
                        coord_points = coordinate.attributes['point'].value
                        for points in coord_points.split(','):
                            # each vertex should have 3 points
                            points = points.split()
                            if len(points) == 3:
                                v = Vertex()
                                v.x = points[0]
                                v.y = points[1]
                                v.z = points[2]
                                vertices.append(v)

        # form XODE geometry objects from the lists of triangles and vertices
        if triangles and vertices:
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
            # triangles
            trimesh_triangles = dom.createElement('triangles')
            for triangle in triangles:
                t = dom.createElement('t')
                t.attributes['ia'] = triangle.ia
                t.attributes['ib'] = triangle.ib
                t.attributes['ic'] = triangle.ic
                trimesh_triangles.appendChild(t)
            trimesh.appendChild(trimesh_triangles)
            # vertices
            trimesh_vertices = dom.createElement('vertices')
            for vertex in vertices:
                v = dom.createElement('v')
                v.attributes['x'] = vertex.x
                v.attributes['y'] = vertex.y
                v.attributes['z'] = vertex.z
                trimesh_vertices.appendChild(v)
            trimesh.appendChild(trimesh_vertices)
            space.appendChild(geom)
        
        # parse, adding to environment
        parser = xode.parser.Parser()
        #print root.toprettyxml()
        self._parseBodies(parser.parseString(root.toxml()))
        
    @threaded()  
    def updateClients(self):
        self.updateDone = False      
        if not self.updateLock.acquire(False): 
            return
        
        # build message to send
        message = []
        for (body, geom) in self.body_geom:
            item = {}
            # real bodies (boxes, spheres, ...)
            if body != None:
                # transform (rotate, translate) body accordingly
                item['position'] = body.getPosition()
                item['rotation'] = body.getRotation()
                if hasattr(body, 'color'): item['color'] = body.color
                
                # switch different geom objects
                if type(geom) == ode.GeomBox:
                    # cube
                    item['type'] = 'GeomBox'
                    item['scale'] = geom.getLengths()
                elif type(geom) == ode.GeomSphere:
                    # sphere
                    item['type'] = 'GeomSphere'
                    item['radius'] = geom.getRadius()

                elif type(geom) == ode.GeomCCylinder:
                    # capped cylinder
                    item['type'] = 'GeomCCylinder'
                    item['radius'] = geom.getParams()[0]
                    item['length'] = geom.getParams()[1] - 2 * item['radius']

                elif type(geom) == ode.GeomCylinder:
                    # solid cylinder
                    item['type'] = 'GeomCylinder'
                    item['radius'] = geom.getParams()[0]
                    item['length'] = geom.getParams()[1]   
                else:
                    # TODO: add other geoms here
                    pass

            else:
                # no body found, then it must be a plane or triangular mesh
                if type(geom) == ode.GeomPlane:
                    # plane
                    item['type'] = 'GeomPlane'
                    item['normal'] = geom.getParams()[0] # the normal vector to the plane
                    item['distance'] = geom.getParams()[1] # the distance to the origin
                    
                elif type(geom) == ode.GeomTriMesh:
                    # this code block is the only difference from the parent
                    # object's updateClients method
                    item['type'] = 'GeomTriMesh'
                    # list of triangles
                    # note: geom.getTriangleCount seems to be undocumented, can't find
                    #       it in the API anywhere, just stumbled across it on the 
                    #       pyode-user mailing list
                    item['triangles'] = [geom.getTriangle(i) 
                                         for i in range(geom.getTriangleCount())]
            
            message.append(item)
        
        # Listen for clients
        self.server.listen()
        if self.server.clients > 0: 
            # If there are clients send them the new data
            self.server.send(message)
        time.sleep(0.02)
        self.updateLock.release()
        self.updateDone = True
                
if __name__ == '__main__' :
    from threading import Thread, Event
    import sys
    import select
    
    class EnvironmentThread(Thread):
        def __init__(self):
            Thread.__init__(self)
            self.running = Event()
            self.analysis_runs = []
            self.env = ALifeEnvironment()
            
        def run(self):
            while not self.running.isSet():
                self.env.step()
    
    e = EnvironmentThread()
    e.start()
    print 'Press Return to exit'
    while not select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        pass
    print "Stopping ALife environment... "
    e.running.set()
    e.join()