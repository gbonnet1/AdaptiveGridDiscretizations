# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .ad_generic import is_ad
from . import ad_generic
"""
This file takes advantage of the __array_function__ mechanisme of numpy to reimplement 
a number of numpy functions in a way that is compatible with AD information.
"""

#https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
numpy_overloads = {}
cupy_only_overloads = {} # Take precedence over numpy overloads if necessary
numpy_implementation = {# Use original numpy implementation
	np.moveaxis,np.expand_dims,np.ndim,np.squeeze,
	np.amin,np.amax,np.argmin,np.argmax,
	np.sum,np.prod,
	np.full_like,np.ones_like,np.zeros_like
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

@implements(np.empty_like)
def empty_like(a,*args,**kwargs):
	return type(a)(np.empty_like(a.value,*args,**kwargs))

@implements(np.copyto)
def copy_to(dst,src,*args,**kwargs):
	if is_ad(src): raise ValueError("copyto is not supported with an AD source")
	np.copyto._implementation(dst.value,src,*args,**kwargs)
"""
@implements(np.full_like)
def full_like(a,*args,**kwargs):
	return type(a)(np.full_like(a.value,*args,**kwargs))

# Purposedly implemented for both numpy and cupy AD variables
@implements(np.zeros_like)
def zeros_like(a,*args,**kwargs): return full_like(a,0.,*args,**kwargs)
@implements(np.ones_like)
def ones_like(a,*args,**kwargs):  return full_like(a,1.,*args,**kwargs)
"""

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