from . import functional
from . import cupy_generic
import itertools
import numpy as np

"""
This file implements functions which apply indifferently to several AD types.
"""


def is_adtype(t):
	return (t.__module__.startswith('agd.AutomaticDifferentiation.') 
		and t.__name__ in ('denseAD','denseAD2','spAD','spAD2'))
#	return t in (Sparse.spAD, Dense.denseAD, Sparse2.spAD2, Dense2.denseAD2)

def is_ad(data,iterables=tuple()):
	"""
	Returns None if no ad variable found, or the adtype if one is found.
	Also checks consistency of the ad types.
	"""
	adtype=None
	def check(t):
		nonlocal adtype
		if is_adtype(t):
			if adtype is None: adtype = t
			elif adtype!=t: raise ValueError("Incompatible adtypes found")

	for value in functional.rec_iter(data,iterables): check(type(value))
	return adtype

def stack(elems,axis=0):
	for e in elems:
		if is_ad(e): return type(e).stack(elems,axis)
	return np.stack(elems,axis)

def array(a,copy=True):
	"""
	Similar to np.array, but does not cast AD subclasses of np.ndarray to the base class.
	Turns a list or tuple of arrays with the same dimensions. 
	Turns a scalar into an array scalar.
	"""
	if isinstance(a,(list,tuple)): return stack([asarray(e) for e in a],axis=0)
	elif cupy_generic.isndarray(a): return a.copy() if copy else a
	else: return np.array(a,copy=copy)

def asarray(a): return array(a,copy=False)

def remove_ad(data,iterables=tuple()):
	def f(a): return a.value if is_ad(a) else a
	return functional.map_iterables(f,data,iterables)

def common_cast(*args):
	"""
	If any of the arguments is an AD type, casts all other arguments to that type.
	Casts to ndarray if no argument is an AD type. 
	Usage : recommended when using array scalars with AD information.
	"""
	args = tuple(array(x) for x in args)
	common_type = None
	for x in args: 
		if is_ad(x):
			if common_type is None:
				common_type = type(x)
			if not isinstance(x,common_type):
				raise ValueError("Error : several distinct AD types")
	return args if common_type is None else tuple(common_type(x) for x in args)


def left_operand(data,iterables=tuple()):
	"""
	Turns numpy scalars into array scalars (zero-dimensional arrays).
	This operation is REQUIRED when the right operand is an array scalar with AD content 
	(due to a bug/feature in numpy operator precedence and casting rules). Harmless otherwise.
	"""
	if np.isscalar(data) and not isinstance(data,np.ndarray): 
		return np.array(data)
	return data

def min_argmin(array,axis=None):
	if axis is None: return min_argmin(array.flatten(),axis=0)
	ai = np.argmin(array,axis=axis)
	return np.squeeze(np.take_along_axis(array,np.expand_dims(ai,
		axis=axis),axis=axis),axis=axis),ai

def max_argmax(array,axis=None):
	if axis is None: return max_argmax(array.flatten(),axis=0)
	ai = np.argmax(array,axis=axis)
	return np.squeeze(np.take_along_axis(array,np.expand_dims(ai,
		axis=axis),axis=axis),axis=axis),ai

# ------- Linear operators ------


def apply_linear_mapping(matrix,rhs,niter=1):
	"""
	Applies the provided linear operator, to a dense AD variable of first or second order.
	"""
	def step(x):
		nonlocal matrix
		return np.dot(matrix,x) if isinstance(matrix,np.ndarray) else (matrix*x)
	operator = functional.recurse(step,niter)
	return rhs.apply_linear_operator(operator) if is_ad(rhs) else operator(rhs)

def apply_linear_inverse(solver,matrix,rhs,niter=1):
	"""
	Applies the provided linear inverse to a dense AD variable of first or second order.
	"""
	def step(x):
		nonlocal solver,matrix
		return solver(matrix,x)
	operator = functional.recurse(step,niter)
	return rhs.apply_linear_operator(operator) if is_ad(rhs) else operator(rhs)

# ------- Shape manipulation -------

def squeeze_shape(shape,axis):
	if axis is None:
		return shape
	assert shape[axis]==1
	if axis==-1:
		return shape[:-1]
	else:
		return shape[:axis]+shape[(axis+1):]

def expand_shape(shape,axis):
	if axis is None:
		return shape
	if axis==-1:
		return shape+(1,)
	if axis<0:
		axis+=1
	return shape[:axis]+(1,)+shape[axis:]

def _set_shape_free_bound(shape,shape_free,shape_bound):
	if shape_free is not None:
		assert shape_free==shape[0:len(shape_free)]
		if shape_bound is None: 
			shape_bound=shape[len(shape_free):]
		else: 
			assert shape_bound==shape[len(shape_free):]
	if shape_bound is None: 
		shape_bound = tuple()
	assert len(shape_bound)==0 or shape_bound==shape[-len(shape_bound):]
	if shape_free is None:
		if len(shape_bound)==0:
			shape_free = shape
		else:
			shape_free = shape[:len(shape)-len(shape_bound)]
	return shape_free,shape_bound

def disassociate(array,shape_free=None,shape_bound=None,
	expand_free_dims=-1,expand_bound_dims=-1):
	"""
	Turns an array of shape shape_free + shape_bound 
	into an array of shape shape_free whose elements 
	are arrays of shape shape_bound.
	Typical usage : recursive automatic differentiation.
	Caveat : by defaut, singleton dimensions are introduced 
	to avoid numpy's "clever" treatment of scalar arrays.

	Arguments: 
	- array : reshaped array
	- (optional) shape_free, shape_bound : outer and inner array shapes. One is deduced from the other.
	- (optional) expand_free_dims, expand_bound_dims. 
	"""
	shape_free,shape_bound = _set_shape_free_bound(array.shape,shape_free,shape_bound)
	shape_free  = expand_shape(shape_free, expand_free_dims)
	shape_bound = expand_shape(shape_bound,expand_bound_dims)
	
	size_free = np.prod(shape_free)
	array = array.reshape((size_free,)+shape_bound)
	result = np.zeros(size_free,object)
	for i in range(size_free): result[i] = array[i]
	return result.reshape(shape_free)

def associate(array,squeeze_free_dims=-1,squeeze_bound_dims=-1):
	"""
	Turns an array of shape shape_free, whose elements 
	are arrays of shape shape_bound, into an array 
	of shape shape_free+shape_bound.
	Inverse opeation to disassociate.
	"""
	if is_ad(array): 
		return array.associate(squeeze_free_dims,squeeze_bound_dims)
	result = stack(array.flatten(),axis=0)
	shape_free  = squeeze_shape(array.shape,squeeze_free_dims)
	shape_bound = squeeze_shape(result.shape[1:],squeeze_bound_dims) 
	return result.reshape(shape_free+shape_bound)