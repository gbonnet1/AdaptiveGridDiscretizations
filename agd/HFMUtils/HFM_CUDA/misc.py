# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from ...AutomaticDifferentiation.cupy_generic import get_array_module

def packbits(arr,bitorder='big'):
	"""Implements bitorder option in cupy"""
	xp = get_array_module(arr)
	if bitorder=='little':
		shape = arr.shape
		arr = arr.reshape(-1,8)
		arr = arr[:,::-1]
		arr = arr.reshape(shape)
	return xp.packbits(arr)


def round_up(num,den):
	"""
	Returns the least multiple of den after num.
	num and den must be integers. 
	"""
	num,den = np.asarray(num),np.asarray(den)
	return (num+den-1)//den

def block_expand(arr,shape_i,**kwargs):
	"""
	Reshape an array so as to factor  shape_i (the inner shape),
	and move its axes last.
	Inputs : 
	 - **kwargs : passed to np.pad
	Output : 
	 - padded and reshaped array
	 - original shape
	"""
	ndim = len(shape_i)
	shape_pre = arr.shape[:-ndim]
	ndim_pre = len(shape_pre)
	shape_tot=np.array(arr.shape[-ndim:])
	shape_i = np.array(shape_i)

	# Extend data
	shape_o = round_up(shape_tot,shape_i)
	shape_pad = (0,)*ndim_pre + tuple(shape_o*shape_i - shape_tot)
	arr = np.pad(arr, tuple( (0,s) for s in shape_pad), **kwargs) 

	# Reshape
	shape_interleaved = tuple(np.stack( (shape_o,shape_i), axis=1).flatten())
	arr = arr.reshape(shape_pre + shape_interleaved)

	# Move axes
	rg = np.arange(ndim)
	axes_interleaved = ndim_pre + 1+2*rg
	axes_split = ndim_pre + ndim+rg
	arr = np.moveaxis(arr,axes_interleaved,axes_split)

	xp = get_array_module(arr)
	return xp.ascontiguousarray(arr)

def block_squeeze(arr,shape):
	"""
	Inverse operation to block_expand.
	"""
	ndim = len(shape)
	shape_pre = arr.shape[:-2*ndim]
	ndim_pre = len(shape_pre)
	shape_o = arr.shape[(-2*ndim):-ndim]
	shape_i = arr.shape[-ndim:]

	# Move axes
	rg = np.arange(ndim)
	axes_interleaved = ndim_pre + 1+2*rg
	axes_split = ndim_pre + ndim+rg
	arr = np.moveaxis(arr,axes_split,axes_interleaved)

	# Reshape
	arr = arr.reshape(shape_pre
		+tuple(s_o*s_i for (s_o,s_i) in zip(shape_o,shape_i)) )

	# Extract subdomain
	region = tuple(slice(0,s) for s in (shape_pre+shape))
	arr = arr.__getitem__(region)
	return arr