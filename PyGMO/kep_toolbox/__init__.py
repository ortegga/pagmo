from _kep_toolbox import *

def propagate_kep( r0, v0, t, mu ):
	"""
	Keplerian propagation.
	"""
	#from PyGMO import vector
	from _kep_toolbox import __propagate_kep
	return __propagate_kep( r0, v0, t, mu )
