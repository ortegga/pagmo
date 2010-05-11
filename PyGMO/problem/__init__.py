# -*- coding: iso-8859-1 -*-
from _problem import *

class base(_problem._base):
	def __init__(self,*args):
		if len(args) == 0:
			raise ValueError("Cannot initialise base problem without parameters for the constructor.")
		_problem._base.__init__(self,*args)
	def _get_typename(self):
		return str(type(self))
	def get_name(self):
		return self._get_typename()

class schubert(base):
    def __init__(self):
        super(schubert, self).__init__(2, 0, 1, 2, 2)    
        self.lb = [0, -10]
        self.ub = [10, 10]
    def __copy__(self):
        retval = schubert()
        retval.lb = self.lb
        retval.ub = self.ub
        return retval
    def _objfun_impl(self, x):
    	import numpy as np
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            sum1 += i*np.cos((i+1)*x[0] + i)
            sum2 += i*np.cos((i+1)*x[1] + i)
        return (sum1*sum2,)
    def get_name(self):
        return "Schubert"