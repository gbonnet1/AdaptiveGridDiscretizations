# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant

def graph_reverse(fwd,fwd_weight,
	invalid=None,nrev=None,blockDim=1024):
	"""
	Inverses a weighted graph. 
	- invalid : special value to be ignored in fwd (defaults to Int_Max)
	- nrev : expected max number of neighbors in reversed graph
	- irev_t
	Output : 
	 - rev, rev_weight. !! Warning : likely not contiguous arrays. !!
	"""
	fwd = cp.ascontiguousarray(fwd)
	fwd_weight = cp.ascontiguousarray(fwd_weight)

	nfwd = len(fwd)
	shape = fwd.shape[1:]
	size = np.prod(shape)

	int_t = fwd.dtype.type
	weight_t = fwd_weight.dtype.type

	if invalid is None: invalid = np.iinfo(int_t).max
	if nrev is None: nrev = 2*nfwd # Default guess, will be increased if needed
	for dtype in (np.int8,np.int16,np.int32,np.int64):
		if np.iinfo(dtype).max>=nrev:
			irev_t=dtype
			break
	else: raise ValueError("Impossible nrev")

	rev = cp.full( (nrev,)+shape, invalid, dtype=int_t)
	rev_weight = cp.zeros( (nrev,)+shape, dtype=weight_t)
	irev = cp.zeros( (nfwd,)+shape, dtype=irev_t)

	traits = {
		'Int':int_t,
		'weightT':weight_t,
		'irevT':irev_t,
		'invalid':(invalid,int_t)
	}

	source = cupy_module_helper.traits_header(traits,integral_max=True)
	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)

	source += [
	'#include "GraphReverse.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 

	source="\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)
	SetModuleConstant(module,'size_tot',size,int_t)
	SetModuleConstant(module,'nfwd',nfwd,int_t)
	SetModuleConstant(module,'nrev',nrev,int_t)

	cupy_kernel = module.get_function("GraphReverse")
	gridDim = int(np.ceil(size/blockDim))

	irev_mmax = 0
	while True:
		irev_max = np.max(irev)
		irev_mmax = max(irev_max,irev_mmax)

		if irev_max==-1: return rev[:irev_mmax],rev_weight[:irev_mmax]
		if irev_max==nrev: return graph_reverse(fwd,fwd_weight,invalid,2*nrev)

		cupy_kernel((gridDim,),(blockDim,),(fwd,rev,irev,fwd_weight,rev_weight))








