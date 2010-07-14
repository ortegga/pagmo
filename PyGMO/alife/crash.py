import ode
import xode.parser
        
data = '''<?xml version="1.0" encoding="UTF-8"?>
<xode version="1.0r23" name="Robot" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://tanksoftware.com/xode/1.0r23/xode.xsd">
<world>
    <space>
        <body name="robot_body">
            <transform>
                <position y="150" x="0" z="0"/>
            </transform>
            <mass>
                <mass_shape density="0.0208333333333">
                    <box sizex="4.0" sizez="4.0" sizey="3.0"/>
                </mass_shape>
            </mass>
            <geom>
                <box sizex="4.0" sizez="4.0" sizey="3.0"/>
            </geom>
        </body>
        <body name="robot_leg1">
            <transform>
                <position y="147.1" x="1.2" z="-1.2"/>
                <rotation>
                    <euler y="0" x="90" aformat="degrees" z="0"/>
                </rotation>
            </transform>
            <mass>
                <mass_shape density="0.376326959173">
                    <cappedCylinder length="3.8" radius="0.25"/>
                </mass_shape>
            </mass>
            <geom>
                <cappedCylinder length="3.8" radius="0.25"/>
            </geom>
        </body>
        <body name="robot_leg2">
            <transform>
                <position y="147.1" x="-1.2" z="-1.2"/>
                <rotation>
                    <euler y="0" x="90" aformat="degrees" z="0"/>
                </rotation>
            </transform>
            <mass>
                <mass_shape density="0.376326959173">
                    <cappedCylinder length="3.8" radius="0.25"/>
                </mass_shape>
            </mass>
            <geom>
                <cappedCylinder length="3.8" radius="0.25"/>
            </geom>
        </body>
        <body name="robot_leg3">
            <transform>
                <position y="147.1" x="1.2" z="1.2"/>
                <rotation>
                    <euler y="0" x="90" aformat="degrees" z="0"/>
                </rotation>
            </transform>
            <mass>
                <mass_shape density="0.376326959173">
                    <cappedCylinder length="3.8" radius="0.25"/>
                </mass_shape>
            </mass>
            <geom>
                <cappedCylinder length="3.8" radius="0.25"/>
            </geom>
        </body>
        <body name="robot_leg4">
            <transform>
                <position y="147.1" x="-1.2" z="1.2"/>
                <rotation>
                    <euler y="0" x="90" aformat="degrees" z="0"/>
                </rotation>
            </transform>
            <mass>
                <mass_shape density="0.376326959173">
                    <cappedCylinder length="3.8" radius="0.25"/>
                </mass_shape>
            </mass>
            <geom>
                <cappedCylinder length="3.8" radius="0.25"/>
            </geom>
        </body>
        <joint name="robot_body_leg1">
            <link1 body="robot_body"/>
            <link2 body="robot_leg1"/>
            <hinge>
                <axis LowStop="-1.2" y="0" x="-1" z="0" HiStop="1.2"/>
                <anchor y="148.5" x="1.2" z="-1.2" absolute="true"/>
            </hinge>
        </joint>
        <joint name="robot_body_leg2">
            <link1 body="robot_body"/>
            <link2 body="robot_leg2"/>
            <hinge>
                <axis LowStop="-1.2" y="0" x="-1" z="0" HiStop="1.2"/>
                <anchor y="148.5" x="-1.2" z="-1.2" absolute="true"/>
            </hinge>
        </joint>
        <joint name="robot_body_leg3">
            <link1 body="robot_body"/>
            <link2 body="robot_leg3"/>
            <hinge>
                <axis LowStop="-1.2" y="0" x="-1" z="0" HiStop="1.2"/>
                <anchor y="148.5" x="1.2" z="1.2" absolute="true"/>
            </hinge>
        </joint>
        <joint name="robot_body_leg4">
            <link1 body="robot_body"/>
            <link2 body="robot_leg4"/>
            <hinge>
                <axis LowStop="-1.2" y="0" x="-1" z="0" HiStop="1.2"/>
                <anchor y="148.5" x="-1.2" z="1.2" absolute="true"/>
            </hinge>
        </joint>
    </space>
</world>
</xode>
'''
        
class Environment(object):
    def __init__(self):
        self.test = []
        p = xode.parser.Parser()
        self.root = p.parseString(data)
        
if __name__ == "__main__":
    for i in range(500):
        print i,
        e = Environment()
        print e
        