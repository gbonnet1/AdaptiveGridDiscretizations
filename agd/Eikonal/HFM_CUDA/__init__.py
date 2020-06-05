# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

from . import _Interface

def RunGPU(hfmIn,*args,cache=None,**kwargs):
	"""
	Runs the GPU eikonal solver.

	Main input:
	- hfmIn : a dictionary like-structure

	The solver embed some help information, which can be accessed as follows.
	"""
	if cache is not None: print(f"Warning : gpu eikonal solver does not support caching")
	return _Interface.Interface(hfmIn).Run(*args,**kwargs)

class EikonalGPU_NotImplementedError(Exception):
	def __init__(self,message):
		super(EikonalGPU_NotImplementedError,self).__init__(message)