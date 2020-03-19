"""
This file implements functionalities needed to make the agd library generic to cupy/numpy usage.
It does not import cupy, unless absolutely required.
"""

import itertools
import numpy as np
import sys

def from_module(x,module_name):
	_module_name = type(x).__module__
	return module_name == _module_name or _module_name.startswith(module_name+'.')

def from_cupy(x): 
	return from_module(x,'cupy')

def get_array_module(*args,**kwargs):
	"""Returns the module (numpy or cupy) of an array"""
	for arg in itertools.chain(args,kwargs.values()):
		if from_module(arg,'cupy'): 
			return sys.modules['cupy']
	return sys.modules['numpy']

#	import cupy # Alternative implementation requiring cupy import
#	return cupy.get_array_module(*args)

def isndarray(x):
	return isinstance(x,get_array_module(x).ndarray)

def samesize_int_t(float_t):
	"""
	Returns an integer type of the same size (32 or 64 bits) as a given float type
	"""
	float_name = str(float_t)
	if   'float32' in arg: return np.dtype('int32').type
	elif 'float64' in arg: return np.dtype('int64').type
	else: raise ValueError(
		f"Type {float_t} is not a float type, or has no default matching int type")

def cupy_get(x):
	"""
	If argument is a cupy ndarray, returns output of 'get' member function, 
	which is a numpy ndarray. Returns unchanged argument otherwise.
	"""
	return x.get() if from_cupy(x) and isndarray(x) else x

def cupy_get_args(f):
	"""
	Decorator which apply cupy_get to all arguments of given function.
	"""
	@functools.wraps(f)
	def wrapper(*args,**kwargs):
		args = tuple(cupy_get(arg) for arg in args)
		kwargs = {key:cupy_get(value) for key,value in kwargs.items()}
		return f(*args,**kwargs)
	return wrapper

def has_dtype(args,dtype):
	"""
	Wether one member of args is an ndarray with the provided dtype.
	"""
	dtype = np.dtype(dtype)
	return any(arg.dtype==dtype for arg in args if isndarray(arg))

def get_float_t(*args,**kwargs):
	"""
	Returns float32 if found in any argument, else float64.
	"""
	args = itertools.chain(args,kwargs.values())
	target_t = np.dtype('float32').type
	default_t = np.dtype('float64').type
	return target_t if has_dtype(args,target_t) else default_t

def array_float_caster(*args,**kwargs):
	"""
	returns lambda arr : xp.array(arr,dtype=float_t) 
	where xp and float_t are in consistency with the arguments.
	"""
	xp = get_array_module(*args,**kwargs)
	float_t = get_float_t(*args,**kwargs)
	return lambda arr:xp.array(arr,dtype=float_t)
