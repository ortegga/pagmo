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
from environment import ALifeEnvironment
from robot import Robot
from asteroid import SphericalAsteroid
from task import ALifeExperiment, ALifeAgent, ALifeTask
import random


## ALifeOnASphere class
#
#  This class defines a PaGMO problem for the ALife project.
#  The goal is to evolve a walking behaviour for a robot in a low gravity
#  environment (a sphere in this case).
#
#  The objective function measures the distance moved by the robot during
#  the one run of an ALifeExperiment, the robot being controlled by the 
#  given input weights
#
#  @author John Glover
class ALifeOnASphere(base):
    def __init__(self):
        robot = Robot("Robot", [random.randint(-100, 100), 150, 0])
        asteroid = SphericalAsteroid()
        self.environment = ALifeEnvironment(robot, asteroid)
        self.task = ALifeTask(self.environment)
        self.agent = ALifeAgent(len(self.task.getObservation()))
        self.experiment = ALifeExperiment(self.task, self.agent, self.environment)
        super(ALifeOnASphere, self).__init__(self.agent.num_weights())    
        self.lb = [-10 for i in range(self.agent.num_weights())]
        self.ub = [10 for i in range(self.agent.num_weights())]

    ## @return The name of this problem
    def get_name(self):
        return "ALifeOnASphere"
    
    ## @return A copy of this problem, controlled by an agent with the same weights
    def __copy__(self):
        p = ALifeOnASphere()
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
    from asteroid import Asteroid
        
    # define a PaGMO problem intended to evolve a walking behaviour in our robot
    prob = ALifeOnASphere()
    # define the algorithm used to optimise the problem's objective function
    #algo = algorithm.de(50)
    algo = algorithm.ihs(50)
    # define the archipelago
    topo = topology.ring()
    num_islands = 3
    num_individuals = 5
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
    #best_weights = [6.4148824487215705, 5.2338637188628248, 2.8107037342170513, -9.3369358224788339, -4.3672738947888927, -3.7626172791121633, -9.4782832007099387, -0.91293668439681319, 3.7781155796329848, -0.54767323127380707, 1.9436653117418292, 2.5096242751894913, -6.541182607346574, 2.4825944293302498, -6.9113544525838551, 6.2276806474414323, -6.3788904549765011, -1.0213529681999824, 4.6083065000454368, -8.8062039427611847, 5.2301545364728472, -1.5964390211120314, 9.8463742346231662, 4.1383746154867689, -4.3501654023188001, -3.4900260582852263, -0.62956515233971722, -2.0223759156893628, -7.6242786509368088, -0.56280919731925327, 8.2288985085616968, -2.4655894751718099, 3.9024350559097556, -3.2711545663817425, -9.6225658488861168, 2.1515758934435061, 9.2293096440994162, -8.8452033745880954, -3.2070805946498333, -1.9585180196651297, -3.1219763823949562, 9.8384319565467937, -9.946618583628867, -1.0266504472925972, 3.5894372939223023, 4.8951654270258249, 4.4214723392667992, -3.8296161015584627, 4.3337986120955296, -5.239306416437457, -6.8452149768538044, -2.2191397194529254, 4.3505177493332399, 0.29514267866005994, -5.958331568902846, -4.1056402427748635, 3.3491870504951762, -8.1873977733273229, 2.4817854859744974, -0.011506703662912822, 7.7566862443793507, -8.3266925070175617, -5.6833433023772884, 7.3694964450086449, 5.843945030060036, -1.721281424528371, -6.3680703565634396, -9.4880903874130542, -9.9985923727133468, 0.74042393354146152, -9.2024622302524115, 7.0338139046906711, 7.3497940552073384, 1.3365191378567653, 8.0811852523115615, -3.6132515010519453, 4.983718978132508, 5.099581913745701, -5.317218567970329, 7.947436909863665]
    # setup an experiment with a robot controlled by the weights calculated above
    robot = Robot("Robot", [0, 150, 0])
    asteroid = SphericalAsteroid()
    environment = ALifeEnvironment(robot, asteroid)
    task = ALifeTask(environment)
    agent = ALifeAgent(len(task.getObservation()))
    agent.set_weights(best_weights)
    experiment = ALifeExperiment(task, agent, environment)
    viewer = ALifeViewer()
    viewer.set_experiment(experiment)
    viewer.print_controls()
    viewer.start()
