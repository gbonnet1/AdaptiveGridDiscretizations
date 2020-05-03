# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import importlib
import functools

#from .Grid import GetCorners,Rect,GetAxes,GetGrid,PointFromIndex,IndexFromPoint,CenteredLinspace,GridNeighbors
from .LibraryCall import GetBinaryDir
from .run_detail import Cache
from .DictIn import dictIn,dictOut,CenteredLinspace

# def Run(hfmIn,smart=False,**kwargs):
# 	"""
# 	Calls to the HFM library, returns output and prints log.

# 	Parameters
# 	----------
# 	smart : bool  
# 		Choose between a smart and raw run
# 	**kwargs
# 		Passed to RunRaw or RunSmart
# 	"""
# 	return RunSmart(hfmIn,**kwargs) if smart else RunRaw(hfmIn,**kwargs)

# def RunGPU(*args,**kwargs):
# 	from . import HFM_CUDA
# 	return HFM_CUDA.RunGPU(*args,**kwargs)

def VoronoiDecomposition(arr):
	"""
	Calls the FileVDQ library to decompose the provided quadratic form(s),
	as based on Voronoi's first reduction of quadratic forms.
	"""
	from ..Metrics import misc
	from . import FileIO
	bin_dir = GetBinaryDir("FileVDQ",None)
	vdqIn ={'tensors':np.moveaxis(misc.flatten_symmetric_matrix(arr),0,-1)}
	vdqOut = FileIO.WriteCallRead(vdqIn, "FileVDQ", bin_dir)
	return np.moveaxis(vdqOut['weights'],-1,0),np.moveaxis(vdqOut['offsets'],[-1,-2],[0,1])


# ----- Basic utilities for HFM input and output -----

# def GetGeodesics(output,suffix=''): 
# 	if suffix != '' and not suffix.startswith('_'): suffix='_'+suffix
# 	return np.vsplit(output['geodesicPoints'+suffix],
# 					 output['geodesicLengths'+suffix].cumsum()[:-1].astype(int))

# ----------- Helper class ----------

# class dictIn(dict):
# 	"""
# 	A very shallow subclass of a python dictionnary, intended for storing the inputs to the HFM library.
# 	Usage: a number of the free functions of HFMUtils are provided as methods, for convenience.
# 	"""

# 	def __init__(self,*args,**kwargs):
# 		super(dictIn,self).__init__(*args,**kwargs)
# 		if 'arrayOrdering' not in self: 
# 			self['arrayOrdering']='RowMajor'

# 	# Coordinates related methods
# 	@property
# 	def Corners(self):
# 		return GetCorners(self)
# 	@functools.wraps(Rect)
# 	def SetRect(self,*args,**kwargs):
# 		self.update(Rect(*args,**kwargs))
# 	@property
# 	def vdim(self):
# 		"""Vector dimension of the ambient space."""
# 		return len(self['dims'])
# 	@property
# 	def shape(self): return tuple(self['dims'].astype(int))

# 	def copy(self):
# 		return dictIn(dict.copy(self))

# 	Axes=GetAxes
# 	Grid=GetGrid
# 	PointFromIndex=PointFromIndex
# 	IndexFromPoint=IndexFromPoint
# 	GridNeighbors=GridNeighbors

# 	# Running
# 	Run = Run
# 	RunRaw = RunRaw
# 	RunSmart = RunSmart
# 	RunGPU = RunGPU






