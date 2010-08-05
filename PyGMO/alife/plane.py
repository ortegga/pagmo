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
  
## @package plane
#  This package defines a PaGMO problem for the ALife project.
#  The goal is to evolve a walking behaviour for a robot in a low gravity
#  environment (an infinitely flat plane in this case).
from PyGMO.problem import base
from environment import ALifePlane
from robot import Robot
from task import ALifeExperiment, ALifeAgent, ALifeTask
import random


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
        robot = Robot("Robot", [0, 20, 0])
        self.environment = ALifePlane(robot)
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
    from PyGMO import problem, algorithm, topology, archipelago
    from viewer import ALifeViewer
        
    # define a PaGMO problem intended to evolve a walking behaviour in our robot
    prob = ALifeOnAPlane()
    # define the algorithm used to optimise the problem's objective function
    algo = algorithm.de(20)
    # define the archipelago
    topo = topology.ring()
    num_islands = 3
    num_individuals = 10
    a = archipelago(prob, algo, num_islands, num_individuals, topo)
    # allow to evolve
    a.evolve(2)
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
    #best_weights = [-3.5646916549324099, 4.2297155952357812, 4.8319012567698154, 1.7653394395162678, -5.9385043139928193, -2.6072028117321455, 0.74244187282625518, -1.7792144434332968, -3.2569020737492504, 1.2469502456923465, -2.5071521592393076, 2.2939140650387024, -6.1860920732711353, -9.6491922807406549, 6.0256478805333842, -6.9357277331275213, -8.7757216103148838, 6.4470423724940424, -4.5586611652361775, 2.5819098042155666, -5.8417059814945826, -1.5521504837573978, -6.9929335890272029, 9.7654587307623224, 8.065046265781902, 9.2329171851583922, -2.3607065178087971, -6.3263277302741088, 5.7337236923479473, 1.0481509774105362, 2.5542006851678742, 8.451130398037968, -1.8510750139607808, 8.9824896203190931, 8.7033697019362535, -2.4650963698188733, 9.1255665282956961, -6.6031955802842912, 0.092168627392652525, 9.0781981116889021, -2.7240165907181968, -2.5465840963149322, -0.43880858499445807, 4.9443502118292759, -0.30579370474982603, 0.5254414403116705, -8.9054820117591724, 5.5639763135127485, -6.2499075534286996, -2.1961233133414026, 9.7557246310224031, 4.4641885482372601, 9.6171300575296641, -4.894807063876474, -7.0220494801383904, 0.41452052110295767, 6.8466297042059026, -8.6942236858366293, -3.8173415640530539, 8.265765436241173, -5.275916994806316, -0.33289925226280026, 4.8773942270105337, 5.1557105175738283, 2.2578760630058961, 8.7655248760855997, 9.6798924074320958, 9.9520072526782855, 3.3862091818695959, -7.9820720460098915, 5.3154199619921112, -3.7365056329746782, 0.55024714260385466, 8.0039562080092352, 2.4923166655863795, 8.3665703035579746, -3.0525822779653433, -4.7500870979857535, 5.2864160996735254, 8.795737138759101]
    # setup an experiment with a robot controlled by the weights calculated above
    robot = Robot("Robot", [0, 20, 0])
    environment = ALifePlane(robot)
    task = ALifeTask(environment)
    agent = ALifeAgent(len(task.getObservation()))
    agent.set_weights(best_weights)
    experiment = ALifeExperiment(task, agent, environment)
    viewer = ALifeViewer()
    viewer.set_experiment(experiment)
    viewer.print_controls()
    viewer.start()
