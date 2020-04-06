# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .ad_generic import is_ad,array,asarray,stack
"""
This file implements functions which have similarly named numpy counterparts, when
the latter behave badly in cunjunction with AD types.
(A typical issue with the numpy variants is downcasting to the numpy.ndarray type).
"""


def _full_like(arr,*args,**kwargs):
	try: 
		return np.full_like(arr,*args,**kwargs)
	except TypeError: # Some old versions of cupy lack the shape argument
		arr = np.broadcast_to(arr.flatten()[0], kwargs.pop('shape'))
		return np.full_like(arr,*args,**kwargs)

def full_like(a,*args,**kwargs):
	if is_ad(a): return type(a)(_full_like(a.value,*args,**kwargs))
	else: return _full_like(a,*args,**kwargs)

def zeros_like(a,*args,**kwargs): return full_like(a,0.,*args,**kwargs)
def ones_like(a,*args,**kwargs):  return full_like(a,1.,*args,**kwargs)

def broadcast_to(array,shape):
	if is_ad(array): return array.broadcast_to(shape)
	else: return np.broadcast_to(array,shape)

def where(mask,a,b): 
	if is_ad(a) or is_ad(b):
		A,B,Mask = (a,b,mask) if is_ad(b) else (b,a,np.logical_not(mask))
		result = B.copy()
		result[Mask] = A[Mask] if isinstance(A,np.ndarray) else A
		return result
	else: 
		return np.where(mask,a,b)

def sort(array,axis=-1,*varargs,**kwargs):
	if is_ad(array):
		ai = np.argsort(array.value,axis=axis,*varargs,**kwargs)
		return np.take_along_axis(array,ai,axis=axis)
	else:
		return np.sort(array,axis=axis,*varargs,**kwargs)


def concatenate(elems,axis=0):
	for e in elems:
		if is_ad(e): return type(e).concatenate(elems,axis)
	return np.concatenate(elems,axis)

# ------- Compatibility with cupy (old version?) --------

def max(a,*args,**kwargs):
	"""Reimplemeted to support cupy"""
	try: return np.max(a,*args,**kwargs)
	except TypeError: # cupy (old version ?) does not accept initial argument
		initial = kwargs.pop('initial')
		return np.maximum(initial,np.max(a,*args,**kwargs))

def flat(a):
	"""Reimplemented to support cupy"""
	try: return a.flat
	except AttributeError: return a.reshape(-1) # cupy (old version ?) does not have flat

def expand_dims(a,axis):
	"""Reimplemented to support cupy"""
	try: return np.expand_dims(a)
	except TypeError: 
		if axis<0: axis=a.ndim+axis
		newshape = a.shape[:axis]+(1,)+a.shape[axis:]
		return a.reshape(newshape)
