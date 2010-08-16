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
from environment import ALifeEnvironment, ALifePlane
from robot import Robot
from asteroid import Asteroid
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
        robot = Robot(self.environment.world, self.environment.space, 
                      [random.randint(-100, 100), 150, 0])
        self.environment.set_robot(robot)
        asteroid = Asteroid(self.environment.space, "models/asteroid_textured.x3d")
        self.environment.set_asteroid(asteroid)
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
        p = ALifeProblem()
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
#    from PyGMO import problem, algorithm, topology, archipelago
#    from viewer import ALifeViewer
#    from asteroid import Asteroid
#        
#    # ALifeProblem defines a PaGMO problem intended to evolve a walking
#    # behaviour in our robot
#    prob = ALifeProblem()
#    # define the algorithm used to optimise the problem's objective function
#    algo = algorithm.de(75)
#    #algo = algorithm.ihs(50)
#    # define the archipelago
#    topo = topology.ring()
#    num_islands = 3
#    num_individuals = 10
#    a = archipelago(prob, algo, num_islands, num_individuals, topo)
#    # allow to evolve
#    a.evolve(1)
#    a.join()
#    # get the weights from the winning robot
#    max_distance_moved = 0
#    best_weights = []
#    count = 0
#    for i in a:
#        if -i.population.champion.f[0] > max_distance_moved:
#            max_distance_moved = -i.population.champion.f[0]
#            best_weights = i.population.champion.x
#    print "Distance moved during the experiment by the winning robot:", max_distance_moved
#    if not best_weights:
#        print "No control data available for this run."
#    else:
#        print "Winning control data:"
#        print best_weights
#    #best_weights = [-3.3279257205575021, 7.5526754162764895, -2.1841029656902933, -8.2184490079232031, -7.4723335890773068, -7.5104948855102949, -8.8874037367174452, 9.405544800130226, -9.9431732855207944, 5.0869028503884568, -5.7191022613263671, -4.9524283380735916, -1.6838917431271572, -1.1684353101704588, 2.1079858970902698, 8.9190341615295345, 2.2071612740002209, -4.2402503678950021, -5.8878217785505784, 3.3796487686966081, 1.7569072933466234, -8.3638226151129373, -6.7675502046640474, -1.5077758967017463, -1.1881536722072905, 2.322883548515744, 4.9954941449644927, -0.72970989228199556, -4.2340756154005188, -2.1087566503405162, -1.8727709929205201, 4.338171420933838, -8.3527275345779284, -4.2915300226691455, -7.7449791578063554, -9.8645663818945817, 6.2329749541664263, -2.389759321128726, -4.6848132598072407, 2.3438138629717891, -7.5201885937901114, 8.1904412789181258, -3.2532545488823716, 2.5510571734523939, -4.3818709667603217, -6.1052257903900253, 9.4702669507546986, 6.7768019966565092, -2.2886604557639556, -5.5160273383520799, -7.8711977301828551, 8.779888673480496, 3.5891012508320621, 2.0249294229083965, -7.0109702432630305, 6.623254601492782, -2.9597297710922819, -4.1780048088897059, 0.27344447586075127, -4.2186181007126509, -3.3829760676091318, 6.0663280320040229, -3.2247353282056146, 1.8734423461122862, 6.94557470916358, -5.7258391899297401, 8.6456849841826493, 4.0282597411687071, -7.6432222664758775, 0.36350540850637891, 9.4356327033002252, 4.6790052993958859, 0.042265113482500283, -9.6501597758849691, 9.7648004812791953, -2.9981527877954326, -9.9536462313995209, 9.0679450231305765, 4.9521675658898126, -8.9195240744571418]
#    # setup an experiment with a robot controlled by the weights calculated above
#    environment = ALifeEnvironment()
#    robot = Robot(environment.world, environment.space, [0, 150, 0])
#    environment.set_robot(robot)
#    asteroid = Asteroid(environment.space, "models/asteroid_textured.x3d")
#    environment.set_asteroid(asteroid)
#    task = ALifeTask(environment)
#    agent = ALifeAgent(len(task.getObservation()))
#    agent.set_weights(best_weights)
#    experiment = ALifeExperiment(task, agent, environment)
#    viewer = ALifeViewer(environment, experiment)
#    viewer.print_controls()
#    viewer.start()

    from PyGMO import problem, algorithm, topology, archipelago
    from viewer import ALifeViewer
        
    # define a PaGMO problem intended to evolve a walking behaviour in our robot
    prob = ALifeOnAPlane()
    # define the algorithm used to optimise the problem's objective function
    algo = algorithm.ihs(5)
    # define the archipelago
    topo = topology.ring()
    num_islands = 1
    num_individuals = 1
    a = archipelago(prob, algo, num_islands, num_individuals, topo)
    # allow to evolve
    a.evolve(1)
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
    environment = ALifePlane()
    robot = Robot(environment.world, environment.space, [10, 20, 0])
    environment.set_robot(robot)
    task = ALifeTask(environment)
    agent = ALifeAgent(len(task.getObservation()))
    agent.set_weights(best_weights)
    experiment = ALifeExperiment(task, agent, environment)
    viewer = ALifeViewer(environment, experiment)
    viewer.print_controls()
    viewer.start()
