from pybrain.rl.environments.ode.tools.xodetools import XODEfile

class XODEALife(XODEfile):
    def __init__(self, name, **kwargs):
        XODEfile.__init__(self, name, **kwargs)
        self.insertBody('robot_body', 'box', [4.12, 3.0, 2], 30, 
                        pos=[0, 150, 0], passSet=['total'], mass=3.356)
#        self.insertBody('left_leg', 'cappedCylinder', [0.25,7.5], 5, 
#                        pos=[2.06,-2.89,0], euler=[90,0,0], 
#                        passSet=['total'], mass=2.473)
#        self.insertBody('right_leg', 'cappedCylinder', [0.25, 7.5], 5, 
#                        pos=[-2.06, -2.89, 0], euler=[90, 0, 0], 
#                        passSet=['total'], mass=2.473)
        self.centerOn('robot_body')
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None
      
if __name__ == "__main__":
    alife = XODEALife('alife')
    alife.writeXODE()