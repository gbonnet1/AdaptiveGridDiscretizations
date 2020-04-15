# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .ad_generic import is_ad,array,asarray
from . import ad_generic
"""
This file implements functions which have similarly named numpy counterparts, when
the latter behave badly in cunjunction with AD types.
(A typical issue with the numpy variants is downcasting to the numpy.ndarray type).
Thanks to the __array_function__ numpy behavior, the numpy variants can be used transparently.
"""

#https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
numpy_overloads = {}
cupy_only_overloads = {} # Take precedence over numpy overloads if necessary
numpy_implementation = {# Use original numpy implementation
	np.moveaxis,np.expand_dims,np.ndim,np.squeeze,
	np.amin,np.amax,np.argmin,np.argmax,
	np.sum,np.prod,
	} 

def implements(numpy_function,cupy_only=False):
	"""Register an __array_function__ implementation for MyArray objects."""
	def decorator(func):
		if cupy_only: cupy_only_overloads[numpy_function] = func
		else: numpy_overloads[numpy_function] = func
		return func
	return decorator

def _array_function_overload(self,func,types,args,kwargs):
	if self.cupy_based() and func in cupy_only_overloads:
		return cupy_only_overloads[func](*args,**kwargs)
	elif func in numpy_overloads:
		return numpy_overloads[func](*args,**kwargs)
	elif func in numpy_implementation: 
		return func._implementation(*args,**kwargs)
	else: return NotImplemented

# ---- overloads ----

stack = implements(np.stack)(ad_generic.stack)

@implements(np.full_like)
def np_full_like(a,*args,**kwargs):
	return type(a)(np.full_like(a.value,*args,**kwargs))
	
@implements(np.zeros_like)
def zeros_like(a,*args,**kwargs): return full_like(a,0.,*args,**kwargs)
@implements(np.ones_like)
def ones_like(a,*args,**kwargs):  return full_like(a,1.,*args,**kwargs)

@implements(np.broadcast_to)
def broadcast_to(array,shape):
	return array.broadcast_to(shape)
	
@implements(np.where)
def where(mask,a,b): 
	A,B,Mask = (a,b,mask) if is_ad(b) else (b,a,np.logical_not(mask))
	result = B.copy()
	result[Mask] = A[Mask] if isinstance(A,np.ndarray) else A
	return result

@implements(np.sort)
def sort(array,axis=-1,*varargs,**kwargs):
	ai = np.argsort(array.value,axis=axis,*varargs,**kwargs)
	return np.take_along_axis(array,ai,axis=axis)

@implements(np.concatenate)
def concatenate(elems,axis=0):
	for e in elems:
		if is_ad(e): return type(e).concatenate(elems,axis)
	return np.concatenate(elems,axis)	

# ------- Compatibility with cupy (old version?) --------

@implements(np.max,cupy_only=True)
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

@implements(np.expand_dims,cupy_only=True)
def expand_dims(a,axis):
	"""Reimplemented to support cupy"""
	try: return np.expand_dims(a,axis)
	except TypeError: 
		if axis<0: axis=1+a.ndim+axis
		newshape = a.shape[:axis]+(1,)+a.shape[axis:]
		return a.reshape(newshape)

def _full_like(a,*args,**kwargs): # needed for cupy variant below
	if is_ad(a): return type(a)(np.full_like(a.value,*args,**kwargs))
	else: return np.full_like(a,*args,**kwargs) 

@implements(np.full_like,cupy_only=True)
def full_like(arr,*args,**kwargs):
	"""Reimplemented to support cupy"""
	try: return _full_like(arr,*args,**kwargs)
	except TypeError: # Some old versions of cupy lack the shape argument
		arr = np.broadcast_to(arr.flatten()[0], kwargs.pop('shape'))
		return _full_like(arr,*args,**kwargs)



