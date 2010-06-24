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
  
## @package task
#  This module contains the ALife tasks
from pybrain.rl.experiments import Experiment
from pybrain.rl.environments import EpisodicTask
import ode
import numpy as np
import random


class ALifeExperiment(Experiment):
    def __init__(self, task, agent):
        Experiment.__init__(self, task, agent)
        
    def update(self):
        if not self.task.isFinished():
            self._oneInteraction()
       
       
class ALifeAgent(object):
    def __init__(self):
        pass
    
    def integrateObservation(self, observation):
        pass
    
    def getAction(self):
        pass
    
    def giveReward(self, reward):
        pass
     

class ALifeTask(EpisodicTask):
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        self._steps_per_action = 50 # 2 seconds @ 25 fps
        self._step = 0
        self._robot = env.get_robot_body()
        self._prev_robot_position = self._robot.getPosition()
        # set joint max forces
        for j in env.get_robot_joints():
            j.setParam(ode.ParamFMax, 20.0)

    def performAction(self, action):
        if self._step == self._steps_per_action:
            self._step = 0
            # calculate distance travelled since last check
            p = self._robot.getPosition()
            # subtract previous position
            p = (p[0] - self._prev_robot_position[0], 
                 p[1] - self._prev_robot_position[1], 
                 p[2] - self._prev_robot_position[2])
            # get the length of the vector
            distance = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            print "Distance moved:", distance
            self._prev_robot_position = self._robot.getPosition()
            # set new random motion of robot legs
            for j in self.env.get_robot_joints():
                j.setParam(ode.ParamVel, (random.random() * 10) - 5)
        else:
            self._step += 1
        
    def getReward(self):
        return 1.0
    
    def getObservation(self):
        pass
    
    def isFinished(self):
        return False
    
    def reset(self):
        pass
  
        
if __name__ == "__main__":  
    from environment import ALifeEnvironment, Robot
    from viewer import ALifeViewer
    import random
    random.seed()
    # environment
    e = ALifeEnvironment()
    robot_position = [random.randint(-100, 100), 150, 0]
    r = Robot("Robot", robot_position)
    e.load_robot(r.get_xode())
    e.load_asteroid("models/asteroid.x3d")
    # task
    a = ALifeAgent()
    t = ALifeTask(e)
    exp = ALifeExperiment(t, a)
    e.add_experiment(exp)
    # viewer
    v = ALifeViewer()
    v.set_environment(e)
    v.print_controls()
    v.start()