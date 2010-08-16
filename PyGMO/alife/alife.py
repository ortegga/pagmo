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
  
## @package alife
#  This package defines a PaGMO problem for the ALife project.
#  The goal is to evolve a walking behaviour for a robot in a low gravity
#  environment (an asteroid in this case).
from PyGMO.problem import base
from PyGMO import problem, algorithm, topology, archipelago
from environment import ALifeEnvironment, ALifePlane
from robot import Robot
from asteroid import Asteroid
from task import ALifeExperiment, ALifeAgent, ALifeTask
from viewer import ALifeViewer
import random


## ALifeProblem class
#
#  This class defines a PaGMO problem for the ALife project.
#  The goal is to evolve a walking behaviour for a robot in a low gravity
#  environment (an asteroid in this case).
#
#  The objective function measures the distance moved by the robot during
#  the one run of an ALifeExperiment, the robot being controlled by the 
#  given input weights
#
#  @author John Glover
class ALifeProblem(base):
    def __init__(self, env, robot, asteroid=None):
        if not env:
            self.environment = ALifeEnvironment()
        else:
            self.environment = env
        if not robot:
            self.robot = Robot(self.environment.world, self.environment.space, 
                               [random.randint(-100, 100), 150, 0])
            self.environment.set_robot(self.robot)
        else:
            self.robot = robot
        if not asteroid:
            self.asteroid = Asteroid(self.environment.space, "models/asteroid_textured.x3d")
            self.environment.set_asteroid(self.asteroid)
        else:
            self.asteroid = asteroid
        self.task = ALifeTask(self.environment)
        self.agent = ALifeAgent(len(self.task.getObservation()))
        self.experiment = ALifeExperiment(self.task, self.agent, self.environment)
        super(ALifeProblem, self).__init__(self.agent.num_weights())    
        self.lb = [-10 for i in range(self.agent.num_weights())]
        self.ub = [10 for i in range(self.agent.num_weights())]

    ## @return The name of this problem
    def get_name(self):
        return "ALife"
    
    ## @return A copy of this problem, controlled by an agent with the same weights
    def __copy__(self):
        environment = ALifeEnvironment()
        robot = Robot(environment.world, environment.space, 
                      [random.randint(-100, 100), 150, 0])
        robot.set_num_legs(self.robot.get_num_legs())
        robot.set_body_density(self.robot.get_body_density())
        robot.set_leg_density(self.robot.get_leg_density())
        environment.set_robot(robot)
        asteroid = Asteroid(environment.space, "models/asteroid_textured.x3d")
        asteroid.mass = self.asteroid.mass
        environment.set_asteroid(asteroid)
        p = ALifeProblem(environment, robot, asteroid)
        p.agent.set_weights(self.agent.get_weights())
        return p
    
    ## The objective function for this problem. Performs a run of an ALifeExperiment
    #  with the given control weights.
    #  @param x The new set of weights to evaluate
    #  @return The distance moved by the robot during the experiment, under
    #  the control of the given weights.
    def _objfun_impl(self, x):
        # update agent weights
        self.agent.set_weights(x)
        # perform another run of the experiment
        result = self.experiment.perform()
        # return the distance moved by the robot
        return (-result,)
    
    
## ALifeOnAPlane class
#
#  This class defines a PaGMO problem for the ALife project.
#  The goal is to evolve a walking behaviour for a robot in a low gravity
#  environment (an infinitely flat plane in this case).
#
#  The objective function measures the distance moved by the robot during
#  the one run of an ALifeExperiment, the robot being controlled by the 
#  given input weights
#
#  @author John Glover
class ALifeOnAPlane(base):
    ## Constructor. Initialises the problem, declaring an ALife Environment,
    #  Task, Agent and Experiment. Also sets the upper and lower bounds for 
    #  possible controller data.
    def __init__(self):
        self.environment = ALifePlane()
        self.robot = Robot(self.environment.world, self.environment.space, [10, 20, 0])
        self.environment.set_robot(self.robot)
        self.task = ALifeTask(self.environment)
        self.agent = ALifeAgent(len(self.task.getObservation()))
        self.experiment = ALifeExperiment(self.task, self.agent, self.environment)
        super(ALifeOnAPlane, self).__init__(self.agent.num_weights())    
        self.lb = [-10 for i in range(self.agent.num_weights())]
        self.ub = [10 for i in range(self.agent.num_weights())]

    ## @return The name of this problem
    def get_name(self):
        return "ALifeOnAPlane"
    
    ## @return A copy of this problem, controlled by an agent with the same weights
    def __copy__(self):
        p = ALifeOnAPlane()
        p.agent.set_weights(self.agent.get_weights())
        return p
    
    ## The objective function for this problem. Performs a run of an ALifeExperiment
    #  with the given control weights.
    #  @param x The new set of weights to evaluate
    #  @return The distance moved by the robot during the experiment, under
    #  the control of the given weights.
    def _objfun_impl(self, x):
        # update agent weights
        self.agent.set_weights(x)
        # perform another run of the experiment
        result = self.experiment.perform()
        # return the distance moved by the robot
        return (-result,)
      
        
if __name__ == "__main__":
    # ALifeProblem defines a PaGMO problem intended to evolve a walking
    # behaviour in our robot
    prob = ALifeProblem()
    # define the algorithm used to optimise the problem's objective function
    algo = algorithm.de(75)
    # define the archipelago
    topo = topology.ring()
    num_islands = 5
    num_individuals = 10
    a = archipelago(prob, algo, num_islands, num_individuals, topo)
    # allow to evolve
    a.evolve(3)
    a.join()
    # get the weights from the winning robot
    max_distance_moved = 0
    best_weights = []
    count = 0
    for i in a:
        if -i.population.champion.f[0] > max_distance_moved:
            max_distance_moved = -i.population.champion.f[0]
            best_weights = i.population.champion.x
    print "Distance moved during the experiment by the winning robot:", max_distance_moved
    if not best_weights:
        print "No control data available for this run."
    else:
        print "Winning control data:"
        print best_weights

    # setup an experiment with a robot controlled by the weights calculated above
    environment = ALifeEnvironment()
    robot = Robot(environment.world, environment.space, [20, 150, 0])
    environment.set_robot(robot)
    asteroid = Asteroid(environment.space, "models/asteroid_textured.x3d")
    environment.set_asteroid(asteroid)
    task = ALifeTask(environment)
    agent = ALifeAgent(len(task.getObservation()))
    agent.set_weights(best_weights)
    experiment = ALifeExperiment(task, agent, environment)
    viewer = ALifeViewer(environment, experiment)
    viewer.print_controls()
    viewer.start()
