# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements functionalities needed to make the agd library generic to cupy/numpy usage.
It does not import cupy, unless absolutely required.
"""

import itertools
import numpy as np
import sys
import functools
from . import functional

# import the cupy module only if available on the system
try: 
	import cupy as cp
	_cp_ndarray = cp.ndarray
except ModuleNotFoundError: 
	cp=None
	class _cp_ndarray: pass

# ----- Getting the cupy module, if needed -----
def cupy_module():
	import cupy
	return cupy

# -------- Identifying data source -------

def from_cupy(x): 
	return functional.from_module(x,'cupy')

def get_array_module(arg,iterables=(tuple,)):
	"""Returns the module (numpy or cupy) of an array"""
	for x in functional.rec_iter(arg,iterables):
		if functional.is_ad(x): x = x.value
		if from_cupy(x): return sys.modules['cupy']
	return sys.modules['numpy']

def isndarray(x):
	return functional.is_ad(x) or isinstance(x,get_array_module(x).ndarray)

def samesize_int_t(float_t):
	"""
	Returns an integer type of the same size (32 or 64 bits) as a given float type
	"""
	float_t = np.dtype(float_t).type
	float_name = str(float_t)
	if   float_t==np.float32: return np.int32
	elif float_t==np.float64: return np.int64
	else: raise ValueError(
		f"Type {float_t} is not a float type, or has no default matching int type")

# ----------- Retrieving data from a cupy array ------------

def cupy_get(x,dtype64=False,iterables=tuple()):
	"""
	If argument is a cupy ndarray, returns output of 'get' member function, 
	which is a numpy ndarray. Returns unchanged argument otherwise.
	- dtype64 : convert 32 bit floats and ints to their 64 bit counterparts
	"""
	def caster(x):
		if from_cupy(x) and isndarray(x):
			x = x.get()
			if dtype64 and x.dtype.type in (np.int32,np.float32):
				dtype = np.int64 if x.dtype.type==np.int32 else np.float64
				return x.astype(dtype)
		return x
	return functional.map_iterables(caster,x,iterables)

def cupy_get_args(f,*args,**kwargs):
	"""
	Decorator applying cupy_get to all arguments of the given function.
	 - *args, **kwargs : passed to cupy_get
	"""
	@functools.wraps(f)
	def wrapper(*fargs,**fkwargs):
		fargs = tuple(cupy_get(arg,*args,**kwargs) for arg in fargs)
		fkwargs = {key:cupy_get(value,*args,**kwargs) for key,value in fkwargs.items()}
		return f(*fargs,**fkwargs)
	return wrapper

# ----- Casting data to appropriate floating point and integer types ------

def has_dtype(arg,dtype="dtype",iterables=(tuple)):
	"""
	Wether one member of args is an ndarray with the provided dtype.
	"""
	dtype = np.dtype(dtype)
	has_dtype_ = False
	def find_dtype(x):
		nonlocal has_dtype_
		has_dtype_ = has_dtype_ or (isndarray(x) and x.dtype==dtype)
	for x in functional.rec_iter(arg,iterables=iterables): find_dtype(x)
	return has_dtype_
			
def get_float_t(arg,**kwargs):
	"""
	Returns float32 if found in any argument, else float64.
	- kwargs : passed to has_dtype
	"""
	return np.float32 if has_dtype(arg,dtype=np.float32,**kwargs) else np.float64

def array_float_caster(arg,**kwargs):
	"""
	returns lambda arr : xp.asarray(arr,dtype=float_t) 
	where xp and float_t are in consistency with the arguments.
	"""
	xp = get_array_module(arg,**kwargs)
	float_t = get_float_t(arg,**kwargs)
	return lambda arr:xp.asarray(arr,dtype=float_t)

@functional.decorator_with_arguments
def set_output_dtype32(f,silent=False,iterables=(tuple,)):
	"""
	If the output of the given funtion contains ndarrays with 64bit dtype,
	int or float, they are converted to 32 bit dtype.
	"""
	def caster(a):
		if isndarray(a) and a.dtype in (np.float64,np.int64):
			xp = get_array_module(a)
			dtype = np.float32 if a.dtype==np.float64 else np.int32
			if not silent: print(
				f"Casting output of function {f.__name__} " 
				f"from {a.dtype} to {np.dtype(dtype)}")
			return xp.asarray(a,dtype=dtype)
		return a

	@functools.wraps(f)
	def wrapper(*args,**kwargs):
		output = f(*args,**kwargs)
		return functional.map_iterables(caster,output,iterables=iterables)

	return wrapper

def cupy_init_kwargs(x):
	"""
	Returns the parameters necessary to generate a copy of x.
	"""
	x = get_array_module(x).ascontiguousarray(x)
	return {'shape':x.shape,'dtype':x.dtype,
		'memptr':x.data,'strides':x.strides,'order':'C'}