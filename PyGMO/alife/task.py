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
from pybrain.rl.agents.agent import Agent
from pybrain.structure import RecurrentNetwork, LinearLayer, FullConnection
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
            self.agent.giveReward(self.task.getReward())
        else:
            self._step += 1
            
    ## Change the default behaviour of _oneInteraction so that the reward
    #  is only given to the agent after each task evaluation. This is done
    #  in the update function.
    def _oneInteraction(self):
        self.stepid += 1
        self.agent.integrateObservation(self.task.getObservation())
        self.task.performAction(self.agent.getAction())


## ALifeAgent class
#
#  Agents are objects that are capable of producing actions based on
#  previous observations.
#  ALifeAgents supply action values that control Robots.
#
#  @author John Glover
class ALifeAgent(Agent):
    ## Constructor
    #  @param num_observations The number of observations that will be returned
    #  from the Task sensors, and the number of action parameters that the Agent
    #  will produce. Determines the number of inputs and outputs to the Agent's
    #  neural network. 
    def __init__(self, num_observations=4):
        ## @var _last_action The last action that the Agent produced 
        self._last_action = None
        ## @var _last_observation The last observation that the Agent received
        self._last_observation = None
        ## @var _last_reward The last reward that the Agent received
        self._last_reward = None
        ## @var _num_observations The number of observations that will be returned
        #  from the Task sensors, and the number of action parameters that the Agent
        #  will produce. Determines the number of inputs and outputs to the Agent's
        #  neural network.
        self._num_observations = num_observations
        ## @var _network The Agent's neural network
        self._network = None
        self._create_network()
        
    ## Create the Agent's neural network.
    def _create_network(self):
        self._network = RecurrentNetwork()
        # create and add the input layer
        input_layer = LinearLayer(self._num_observations)
        self._network.addInputModule(input_layer)
        # create and add the output layer
        output_layer = LinearLayer(self._num_observations)
        self._network.addOutputModule(output_layer)
        # hidden layer has a random number of neurons
        hidden_layer = LinearLayer(random.randint(3, 5))
        self._network.addModule(hidden_layer)
        # add connections
        input_connection = FullConnection(input_layer, hidden_layer)
        self._network.addConnection(input_connection)
        output_connection = FullConnection(hidden_layer, output_layer)
        self._network.addConnection(output_connection)
        # initialise modules
        self._network.sortModules()
    
    ## Update the last observation received by the Agent
    #  @param observation The latest observation 
    def integrateObservation(self, observation):
        self._last_observation = observation
    
    ## @return The latest action that the Agent has produced
    def getAction(self):
        self._last_action = self._network.activate(self._last_observation)
        return self._last_action
    
    ## Update the reward received by the Agent
    #  @var reward The last reward received by the Agent
    def giveReward(self, reward):
        self._last_reward = reward


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
        self._sensors = sensors.JointSensor()
        for j in env.get_robot_joints():
            self._sensors._joints.append(j)
        self._sensors._update()
        self._sensors._numValues = len(self._sensors.getValues())
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
        
    ## Evaluate the recent actions of the robot. Calculates how far it has
    #  moved since the last evaluation.
    #  @return The distance moved since the last call to getReward()
    def getReward(self):         
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
    
    ## Get the observation for the task, which is the current value of 
    #  the Robot's sensors
    #  @return Robot sensor values (numpy array)
    def getObservation(self):
        self._sensors._update()
        return np.asarray(self._sensors.getValues())
    
    ## @return True if the task has finished, False otherwise
    def isFinished(self):
        # for now, just run until the user quits the task
        # this task could be limited to a maximum number of steps (self.samples), 
        # and/or set to stop once a given self._distance_moved value is obtained
        return False
    
    ## Reset the task
    def reset(self):
        pass
  
        
if __name__ == "__main__":  
    from environment import ALifeEnvironment, Robot
    from viewer import ALifeViewer
    random.seed()
    # environment
    e = ALifeEnvironment()
    robot_position = [random.randint(-100, 100), 150, 0]
    r = Robot("Robot", robot_position)
    e.load_robot(r.get_xode())
    e.load_asteroid("models/asteroid.x3d")
    # task, agent, experiment
    t = ALifeTask(e)
    a = ALifeAgent(len(t.getObservation()))
    exp = ALifeExperiment(t, a)
    e.add_experiment(exp)
    # viewer
    v = ALifeViewer()
    v.set_environment(e)
    v.print_controls()
    v.start()