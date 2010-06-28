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
#  This module contains the ALife Experiment, Agent and Task classes,
#  all derived from their PyBrain equivalents.
#
#  Experiments manage the interaction between Agent and Task objects,
#  and evaluate Tasks at a regular interval.
#
#  Agents are objects that are capable of producing actions based on
#  previous observations.
#  ALifeAgents supply action values that control Robots.
#
#  Tasks associate a purpose with an environment.
#  In this case, it uses the Agent control data to move a Robot, trying
#  to maximise the distance moved between fitness evaluations.
from pybrain.rl.experiments import Experiment
from pybrain.rl.environments import EpisodicTask
from pybrain.rl.environments.ode import sensors, actuators
import ode
import numpy as np
import random

## ALifeExperiment class
#
#  Experiments manage the interaction between Agent and Task objects,
#  and evaluate Tasks at a regular interval.
#
#  @author John Glover
class ALifeExperiment(Experiment):
    # @param task an ALifeTask object
    # @param agent an ALifeAgent object  
    def __init__(self, task, agent):
        Experiment.__init__(self, task, agent)
        ## @var steps_per_evaluation Number of update calls between each
        #  Task evaluation
        self._steps_per_evaluation = 50
        ## @var step The current step number
        self._step = 0
    
    ## Update the experiment, getting action data from the Agent
    #  and passing it to the Task. 
    #  Evaluates the task at regular intervals.
    #  This is called at each step of the ODE environment
    def update(self):
        # perform task action
        if not self.task.isFinished():
            self._oneInteraction()
            
        # evaluate task at regular intervals
        if self._step == self._steps_per_evaluation:
            self._step = 0
            self.task.evaluate()
        else:
            self._step += 1


## ALifeAgent class
#
#  Agents are objects that are capable of producing actions based on
#  previous observations.
#  ALifeAgents supply action values that control Robots.
#
#  @author John Glover
class ALifeAgent(object):
    ## Constructor
    def __init__(self):
        self._force_inc = 1
        self._force_max = 50
        self._force = 0
        self._direction = 1
    
    def integrateObservation(self, observation):
        pass
    
    def getAction(self):
        if self._direction:
            if self._force < self._force_max - self._force_inc:
                self._force += self._force_inc
            else:
                self._direction = 0
        else:
            if self._force > (self._force_max - self._force_inc) * -1:
                self._force -= self._force_inc
            else:
                self._direction = 1
        return [self._force, self._force, self._force, self._force]
    
    def giveReward(self, reward):
        pass


## ALifeTask class
#
#  Tasks associate a purpose with an environment.
#  In this case, it uses the Agent control data to move a Robot, trying
#  to maximise the distance moved between fitness evaluations.
#
#  @author John Glover
class ALifeTask(EpisodicTask):
    ## Constructor
    #  @param env ALifeEnvironment object 
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        ## @var _robot_stable Whether the robot has stabilised on the asteroid or not.
        #  A robot is stable if it moves less than 1m between consecutive calls to evaluate()
        #  The task will not really start until this has happened. 
        self._robot_stable = False
        ## @var _robot The Robot being controlled in this task
        self._robot = env.get_robot_body()
        ## @var _prev_robot_position The position of the robot at the previous  Task evaluation 
        self._prev_robot_position = self._robot.getPosition()
        ## @var _distance_moved The distance moved by the robot between consecutive evaluations
        self._distance_moved = 0.0
        ## @var _sensors The Robot's joint sensors
        self._sensors = None
        # todo: add sensors
        ## @var _actuator The Robot's actuator, moves the legs 
        self._actuator = actuators.JointActuator()
        for j in env.get_robot_joints():
            self._actuator._joints.append(j)
        self._actuator._numValues = self._actuator._countValues()

    ## Perform an action with the robot using the given action data
    #  @param action Action data from an ALifeAgent 
    def performAction(self, action):
        # statements from EpisodicTask and Task performAction methods
        # Task.performAction calls environment.performAction which we want to avoid
        self.samples += 1
        # check that robot has stabilised on the asteroid surface
        if not self._robot_stable:
            return
        # update actuator with values from action
        self._actuator._update(action)
        # also from EpisodicTask's performAction method
        self.addReward()
        
    ## Evaluate the recent actions of the robot. Calculates how far it has
    #  moved since the last evaluation.
    #  @return The distance moved since the last evaluation
    def evaluate(self):         
        # calculate distance travelled since last check
        p = self._robot.getPosition()
        # subtract previous position
        p = (p[0] - self._prev_robot_position[0], 
             p[1] - self._prev_robot_position[1], 
             p[2] - self._prev_robot_position[2])
        # get the length of the vector
        self._distance_moved = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
        self._prev_robot_position = self._robot.getPosition()
        if not self._robot_stable and self._distance_moved < 1.0:
            self._robot_stable = True
        return self._distance_moved
        
    def getReward(self):
        return 1.0
    
    def getObservation(self):
        pass
    
    def isFinished(self):
        # for now, just run until the user quits the task
        # this task could be limited to a maximum number of steps (self.samples), 
        # and/or set to stop once a given self._distance_moved value is obtained
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