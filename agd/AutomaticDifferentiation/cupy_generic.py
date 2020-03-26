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

def get_array_module(arg,iterables=(tuple,)):
	"""Returns the module (numpy or cupy) of an array"""
	module = sys.modules['numpy']
	def gam(x): 
		if from_cupy(x): 
			nonlocal module
			module=sys.modules['cupy']
	misc.map_iterables(gam,arg,iterables=iterables)
	return module

#	import cupy # Alternative implementation requiring cupy import
#	return cupy.get_array_module(*arg)

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
				x = np.array(x,dtype=dtype)
		return x
	return misc.map_iterables(caster,x,iterables)

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
	for x in misc.rec_iter(arg,iterables=iterables): find_dtype(x)
	return has_dtype_
			
def get_float_t(arg,**kwargs):
	"""
	Returns float32 if found in any argument, else float64.
	- kwargs : passed to has_dtype
	"""
	return np.float32 if has_dtype(arg,dtype=np.float32,**kwargs) else np.float64

def array_float_caster(arg,**kwargs):
	"""
	returns lambda arr : xp.array(arr,dtype=float_t) 
	where xp and float_t are in consistency with the arguments.
	"""
	xp = get_array_module(arg,**kwargs)
	float_t = get_float_t(arg,**kwargs)
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






