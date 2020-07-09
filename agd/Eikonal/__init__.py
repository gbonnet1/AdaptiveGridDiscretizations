# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
The Eikonal package embeds CPU and GPU numerical solvers of (generalized) eikonal 
equations. These are variants of the fast marching and fast sweeping method, based on 
suitable discretizations of the PDE, and written and C++.

Please see the illustrative notebooks for detailed usage instructions and examples:
https://github.com/Mirebeau/AdaptiveGridDiscretizations

Main object : 
- dictIn : a dictionary-like structure, used to gathers the arguments of the 
eikonal solver, and eventually call it.
"""

import numpy as np
import importlib
import functools

from .LibraryCall import GetBinaryDir
from .run_detail import Cache
from .DictIn import dictIn,dictOut,CenteredLinspace


def VoronoiDecomposition(arr,mode=None,steps='Both',*args,**kwargs):
	"""
	Calls the FileVDQ library to decompose the provided quadratic form(s),
	as based on Voronoi's first reduction of quadratic forms.
	- mode : 'cpu' or 'gpu' or 'gpu_transfer'. 
	 Defaults to VoronoiDecomposition.default_mode if specified, or to gpu/cpu adequately.
	- args,kwargs : passed to gpu decomposition method
	"""
	from ..AutomaticDifferentiation.cupy_generic import cupy_set,cupy_get,from_cupy
	if mode is None: mode = VoronoiDecomposition.default_mode
	if mode is None: mode = 'gpu' if from_cupy(arr) else 'cpu'
	if mode in ('gpu','gpu_transfer'):
		from .HFM_CUDA.VoronoiDecomposition import VoronoiDecomposition as VD
		if mode=='gpu_transfer': arr = cupy_set(arr)
		out = VD(arr,*args,steps=steps,**kwargs)
		if mode=='gpu_transfer': out = cupy_get(out,iterables=(tuple,))
		return out
	elif mode=='cpu':
		from ..Metrics import misc
		from . import FileIO
		bin_dir = GetBinaryDir("FileVDQ",None)
		dim = arr.shape[0]; shape = arr.shape[2:]
		arr = arr.reshape( (dim,dim,np.prod(shape,dtype=int)) )
		arr = np.moveaxis(misc.flatten_symmetric_matrix(arr),0,-1)
		vdqIn ={'tensors':arr,'steps':steps}
		vdqOut = FileIO.WriteCallRead(vdqIn, "FileVDQ", bin_dir)
		weights = np.moveaxis(vdqOut['weights'],-1,0)
		offsets = np.moveaxis(vdqOut['offsets'],(-1,-2),(0,1)).astype(int)
		weights,offsets = (e.reshape(e.shape[:depth]+shape) 
			for (e,depth) in zip((weights,offsets),(1,2)))
		if steps=='Both': return weights,offsets
		objective = vdqOut['objective'].reshape(shape)
		vertex = vdqOut['vertex'].reshape(shape).astype(int)
		chg = np.moveaxis(vdqOut['chg'],(-1,-2),(0,1)) 
		chg=chg.reshape((dim,dim)+shape)
		return chg,vertex,objective,weights,offsets
	else: raise ValueError(f"VoronoiDecomposition unsupported mode {mode}")

VoronoiDecomposition.default_mode = None