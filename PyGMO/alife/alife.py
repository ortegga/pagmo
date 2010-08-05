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
  
## @package problem
from PyGMO.problem import base
from environment import ALifeEnvironment
from robot import Robot
from task import ALifeExperiment, ALifeAgent, ALifeTask
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
    def __init__(self):
        self.environment = ALifeEnvironment()
        r = Robot("Robot", [random.randint(-100, 100), 150, 0])
        self.environment.load_robot(r.get_xode())
        self.environment.load_asteroid("models/asteroid.x3d")
        self.task = ALifeTask(self.environment)
        self.agent = ALifeAgent(len(self.task.getObservation()))
        self.experiment = ALifeExperiment(self.task, self.agent, self.environment)
        super(ALifeProblem, self).__init__(self.agent.num_weights())    
        self.lb = [-10 for i in range(self.agent.num_weights())]
        self.ub = [10 for i in range(self.agent.num_weights())]

    def get_name(self):
        return "ALife"
    
    def __copy__(self):
        p = ALifeProblem()
        p.agent.set_weights(self.agent.get_weights())
        return p
    
    def _objfun_impl(self, x):
        # update agent weights
        self.agent.set_weights(x)
        # perform another run of the experiment
        result = self.experiment.perform()
        # return the distance moved by the robot
        return (-result,)
      
        
if __name__ == "__main__":
    from PyGMO import problem, algorithm, topology, archipelago
    from viewer import ALifeViewer
        
    # ALifeProblem defines a PaGMO problem intended to evolve a walking
    # behaviour in our robot
    prob = ALifeProblem()
    # define the algorithm used to optimise the problem's objective function
    algo = algorithm.de(50)
    #algo = algorithm.ihs(500)
    # define the archipelago
    topo = topology.ring()
    num_islands = 8
    num_individuals = 20
    a = archipelago(prob, algo, num_islands, num_individuals, topo)
    # allow to evolve
    a.evolve(5)
    a.join()
    max_distance_moved = 0
    best_weights = []
    for i in a:
        if -i.population.champion.f[0] > max_distance_moved:
            max_distance_moved = -i.population.champion.f[0]
            best_weights = i.population.champion.x
    print "Distance moved during the experiment by the winning robot:", max_distance_moved
    
    # setup an experiment with a robot controlled by the weights calculated above
    environment = ALifeEnvironment()
    robot_position = [0, 110, 0]
    r = Robot("Robot", robot_position)
    environment.load_robot(r.get_xode())
    environment.load_asteroid("models/asteroid.x3d")
    task = ALifeTask(environment)
    agent = ALifeAgent(len(task.getObservation()))
    agent.set_weights(best_weights)
    experiment = ALifeExperiment(task, agent, environment)
    viewer = ALifeViewer()
    viewer.set_experiment(experiment)
    viewer.print_controls()
    viewer.start()
