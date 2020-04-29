# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import numbers
import functools
from .ad_generic import is_ad
from . import ad_generic
from .cupy_generic import isndarray
"""
This file takes advantage of the __array_function__ mechanisme of numpy to reimplement 
a number of numpy functions in a way that is compatible with AD information.
"""

#https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
numpy_overloads = {}
cupy_alt_overloads = {} # Used for numpy function unsupported by cupy
numpy_implementation = {# Use original numpy implementation
	np.moveaxis,np.ndim,np.squeeze,
	np.amin,np.amax,np.argmin,np.argmax,
	np.sum,np.prod,
	np.full_like,np.ones_like,np.zeros_like,np.reshape,np.take_along_axis,
	} 

def implements(numpy_function):
	"""Register an __array_function__ implementation for MyArray objects."""
	def decorator(func):
		numpy_overloads[numpy_function] = func
		return func
	return decorator

def implements_cupy_alt(numpy_function,exception):
	"""Register an alternative to a numpy function only partially supported by cupy"""
	def decorator(func):
		cupy_alt_overloads[numpy_function] = (func,exception)
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			try: return numpy_function(*args,**kwargs)
			except exception: return func(*args,**kwargs)
		return wrapper
	return decorator

def _array_function_overload(self,func,types,args,kwargs,cupy_alt=True):

	if cupy_alt and self.cupy_based() and func in cupy_alt_overloads:
		func_alt,exception = cupy_alt_overloads[func]
		try: return _array_function_overload(self,func,types,args,kwargs,cupy_alt=False)
		except exception: return func_alt(*args,**kwargs)

	if func in numpy_overloads:
		return numpy_overloads[func](*args,**kwargs)
	elif func in numpy_implementation: 
		return func._implementation(*args,**kwargs)
	else: return NotImplemented

# ---- overloads ----

stack = implements(np.stack)(ad_generic.stack)

@implements(np.expand_dims)
def expand_dims(a,axis):
	if axis<0: axis=axis+a.ndim+1
	return np.reshape(a,a.shape[:axis]+(1,)+a.shape[axis:])

@implements(np.empty_like)
def empty_like(a,*args,**kwargs):
	return type(a)(np.empty_like(a.value,*args,**kwargs))

@implements(np.copyto)
def copy_to(dst,src,*args,**kwargs):
	if is_ad(src): raise ValueError("copyto is not supported with an AD source")
	np.copyto._implementation(dst.value,src,*args,**kwargs)

@implements(np.broadcast_to)
def broadcast_to(array,shape):
	return array.broadcast_to(shape)
	
@implements(np.where)
def where(mask,a,b): 
	A,B,Mask = (a,b,mask) if is_ad(b) else (b,a,np.logical_not(mask))
	result = B.copy()
	result[Mask] = A[Mask] if isndarray(A) else A
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

@implements(np.pad)
def pad(array, pad_width, *args,**kwargs):
	if isinstance(pad_width,numbers.Integral):
		pad_width = (pad_width,)
	if isinstance(pad_width[0],numbers.Integral) and len(pad_width)==1:
		pad_width = ((pad_width[0],pad_width[0]),)
	if len(pad_width)==1:
		pad_width = pad_width*array.ndim
	return array.pad(pad_width,*args,**kwargs)









