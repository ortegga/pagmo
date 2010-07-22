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
#  todo: description
from PyGMO.problem import base
from environment import ALifeEnvironment, Robot
from task import ALifeExperiment, ALifeAgent, ALifeTask
import random


class ALifeProblem(base):
    def __init__(self):
        self.environment = ALifeEnvironment()
#        r = Robot("Robot", [random.randint(-100, 100), 150, 0])
        r = Robot("Robot", [0, 110, 0])
        self.environment.load_robot(r.get_xode())
        self.environment.load_asteroid("models/asteroid.x3d")
        self.task = ALifeTask(self.environment)
        self.agent = ALifeAgent(len(self.task.getObservation()))
        self.experiment = ALifeExperiment(self.task, self.agent, self.environment)
        
        super(ALifeProblem, self).__init__(self.agent.num_weights())    
        self.lb = [-10 for i in range(self.agent.num_weights())]
        self.ub = [10 for i in range(self.agent.num_weights())]

    def get_name(self):
        return "ALifeProblem"
    
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
#        if result:
#            print result
        return (result,)
      
        
if __name__ == "__main__":
    from PyGMO import problem, algorithm, topology, archipelago
    from viewer import ALifeViewer
        
    # evolve control for the robot
    prob = ALifeProblem()
    algo = algorithm.scipy_slsqp()
    topo = topology.ring()
    a = archipelago(prob, algo, 8, 20, topo)
    a.evolve(1)
    a.join()
    print min([i.population.champion.f[0] for i in a])
    
    # view the robot with winning control data
#    v = ALifeViewer()
#    v.set_environment(e)
#    v.print_controls()
#    v.start()