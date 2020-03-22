"""
This file implements functionalities needed to make the agd library generic to cupy/numpy usage.
It does not import cupy, unless absolutely required.
"""

import itertools
import numpy as np
import sys
import functools
import types
from . import misc

# ----- General methods -----

def decorator_with_arguments(decorator):
	"""
	Decorator intended to simplify writing decorators with arguments. 
	(In addition to the decorated function itself.)
	"""
	@functools.wraps(decorator)
	def wrapper(f=None,*args,**kwargs):
		if f is None: return lambda f: decorator(f,*args,**kwargs)
		else: return decorator(f,*args,**kwargs)
	return wrapper


def decorate_module_functions(module,decorator,
	copy_module=True,fct_names=None,ret_decorated=False):
	"""
	Decorate the functions of a module.
	Inputs : 
	 - module : whose functions must be decorated
	 - decorator : to be applied
	 - copy_module : create a shallow copy of the module
	 - fct_names (optional) : list of functions to be decorated.
	  If unspecified, all functions, builtin functions, and builtin methods, are decorated.
	"""
	if copy_module: #Create a shallow module copy
		new_module = type(module)(module.__name__, module.__doc__)
		new_module.__dict__.update(module.__dict__)
		module = new_module

	decorated = []

	for key,value in module.__dict__.items():
		if fct_names is None: 
			if not isinstance(value,(types.FunctionType,types.BuiltinFunctionType,
				types.BuiltinMethodType)):
				continue
		elif key not in fct_names:
			continue
		decorated.append(key)
		module.__dict__[key] = decorator(value)
	return (module,decorated) if ret_decorated else module


def from_module(x,module_name):
	_module_name = type(x).__module__
	return module_name == _module_name or _module_name.startswith(module_name+'.')

# -------- Identifying data source -------

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
	float_t = np.dtype(float_t).type
	float_name = str(float_t)
	if   float_t==np.float32: return np.int32
	elif float_t==np.float64: return np.int64
	else: raise ValueError(
		f"Type {float_t} is not a float type, or has no default matching int type")

# ----------- Retrieving data from a cupy array ------------

def cupy_get(x):
	"""
	If argument is a cupy ndarray, returns output of 'get' member function, 
	which is a numpy ndarray. Returns unchanged argument otherwise.
	"""
	return x.get() if from_cupy(x) and isndarray(x) else x

def cupy_get_args(f):
	"""
	Decorator applying cupy_get to all arguments of the given function.
	"""
	@functools.wraps(f)
	def wrapper(*args,**kwargs):
		args = tuple(cupy_get(arg) for arg in args)
		kwargs = {key:cupy_get(value) for key,value in kwargs.items()}
		return f(*args,**kwargs)
	return wrapper

# ----- Casting data to appropriate floating point and integer types ------

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
	return np.float32 if has_dtype(args,np.float32) else float64

def array_float_caster(*args,**kwargs):
	"""
	returns lambda arr : xp.array(arr,dtype=float_t) 
	where xp and float_t are in consistency with the arguments.
	"""
	xp = get_array_module(*args,**kwargs)
	float_t = get_float_t(*args,**kwargs)
	return lambda arr:xp.array(arr,dtype=float_t)

@decorator_with_arguments
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
			return xp.array(a,dtype=dtype)
		return a

	@functools.wraps(f)
	def wrapper(*args,**kwargs):
		output = f(*args,**kwargs)
		return misc.map_iterables(caster,output,iterables=iterables)

	return wrapper






