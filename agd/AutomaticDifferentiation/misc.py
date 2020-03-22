import numpy as np
import numbers
import functools
import operator

# We cannot directly import is_ad due to interdependency of packages
def is_ad(a):
	from . import is_ad as _is_ad
	return _is_ad(a)

# ------- Ugly utilities -------
def _tuple_first(a): 	return a[0] if isinstance(a,tuple) else a
def _getitem(a,where):
	return a if (where is True and not isinstance(a,np.ndarray)) else a[where]
def _add_dim(a):		return np.expand_dims(a,axis=-1)	
def _add_dim2(a):		return _add_dim(_add_dim(a))

def _to_tuple(a): return tuple(a) if hasattr(a,"__iter__") else (a,)

def key_expand(key,depth=1): 
	"""Modifies a key to access an array with more dimensions. Needed if ellipsis is used."""
	if isinstance(key,tuple):
		if any(a is ... for a in key):
			return key + (slice(None),)*depth
	return key

def _pad_last(a,pad_total): # Always makes a deep copy
		return np.pad(a, pad_width=((0,0),)*(a.ndim-1)+((0,pad_total-a.shape[-1]),), mode='constant', constant_values=0)
def _add_coef(a,b):
	if a.shape[-1]==0: return b
	elif b.shape[-1]==0: return a
	else: return a+b
def _prep_nl(s): return "\n"+s if "\n" in s else s

def _concatenate(a,b,shape=None):
	if shape is not None:
		if a.shape[:-1]!=shape: a = np.broadcast_to(a,shape+a.shape[-1:])
		if b.shape[:-1]!=shape: b = np.broadcast_to(b,shape+b.shape[-1:])
	return np.concatenate((a,b),axis=-1)

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

def _set_shape_constant(shape=None,constant=None):
	if isinstance(shape,np.ndarray): shape=tuple(shape)
	if constant is None:
		if shape is None:
			raise ValueError("Error : unspecified shape or constant")
		constant = np.full(shape,0.)
	else:
		if not isinstance(constant,np.ndarray):
			constant = np.array(constant)
		if shape is not None and shape!=constant.shape: 
			raise ValueError("Error : incompatible shape and constant")
		else:
			shape=constant.shape
	return shape,constant

def _test_or_broadcast_ad(array,shape,broadcast,ad_depth=1):
	if broadcast:
		if array.shape[:-ad_depth]==shape:
			return array
		else:
			return np.broadcast_to(array,shape+array.shape[-ad_depth:])
	else:
		assert array.shape[:-ad_depth]==shape
		return array

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

# -------- For Dense and Dense2 -----

def apply_linear_operator(op,rhs,flatten_ndim=0):
	"""Applies a linear operator to an array with more than two dimensions,
	by flattening the last dimensions"""
	assert (rhs.ndim-flatten_ndim) in [1,2]
	shape_tail = rhs.shape[1:]
	op_input = rhs.reshape((rhs.shape[0],np.prod(shape_tail,dtype=int)))
	op_output = op(op_input)
	return op_output.reshape((op_output.shape[0],)+shape_tail)


# -------- For Reverse and Reverse2 -------

class pair(object):
	"""
	A two element iterable. 
	Introduced as an alternative of tuple, to avoid confusion in map_iterables
	"""
	def __init__(self,first,second):
		self.first=first
		self.second=second
	def __iter__(self):
		yield self.first
		yield self.second
	def __str__(self):
		return "pair("+str(self.first)+","+str(self.second)+")"
	def __repr__(self):
		return "pair("+repr(self.first)+","+repr(self.second)+")"

def from_generator(iterable_type):
	"""
	Returns the method for constructing an object from a generator.
	"""
	return getattr(iterable_type,'from_generator',iterable_type)

def map_iterables(f,a,iterables,split=False): 
	"""
	Apply f to variable 'a' exploring recursively certain iterables
	"""
	if isinstance(a,iterables):
		type_a = type(a)
		if issubclass(type(a),dict):
			result = type_a({key:map_iterables(f,val,iterables,split=split) for key,val in a.items()})
			if split: return type_a({key:a for key,(a,_) in a.items()}), type_a({key:a for key,(_,a) in a.items()})
			else: return result
		else: 
			ctor_a = from_generator(type_a)
			result = ctor_a(map_iterables(f,val,iterables,split=split) for val in a)
			if split: return ctor_a(a for a,_ in result), ctor_a(a for _,a in result)
			else: return result 
	return f(a)

def map_iterables2(f,a,b,iterables):
	"""
	Apply f to variable 'a' and 'b' zipped, exploring recursively certain iterables
	"""
	for type_iterable in iterables:
		if isinstance(a,type_iterable):
			if issubclass(type_iterable,dict):
				return type_iterable({key:map_iterables2(f,a[key],b[key],iterables) for key in a})
			else: 
				return from_generator(type_iterable)(map_iterables2(f,ai,bi,iterables) for ai,bi in zip(a,b))
	return f(a,b)

def ready_ad(a):
	"""
	Readies a variable for adding ad information, if possible.
	Returns : readied variable, boolean (wether AD extension is possible)
	"""
	if is_ad(a):
		raise ValueError("Variable a already contains AD information")
	elif isinstance(a,numbers.Real) and not isinstance(a,numbers.Integral):
		return np.array(a),True
	elif isinstance(a,np.ndarray) and not issubclass(a.dtype.type,numbers.Integral):
		return a,True
	else:
		return a,False

# Applying a function
def _apply_output_helper(rev,val,iterables):
	"""
	Adds 'virtual' AD information to an output (with negative indices), 
	in selected places.
	"""
	def f(a):
		a,to_ad = ready_ad(a)
		if to_ad:
			shape = pair(rev.size_rev,a.shape)
			return rev._identity_rev(constant=a),shape		
		else:
			return a,None
	return map_iterables(f,val,iterables,split=True)


def register(identity,data,iterables):
	def reg(a):
		a,to_ad = ready_ad(a)
		if to_ad: return identity(constant=a)
		else: return a 
	return map_iterables(reg,data,iterables)


def _to_shapes(coef,shapes,iterables):
	"""
	Reshapes a one dimensional array into the given shapes, 
	given as a tuple of pair(start,shape) 
	"""
	def f(s):
		if s is None:
			return None
		else:
			start,shape = s
			return coef[start : start+np.prod(shape,dtype=int)].reshape(shape)
	return map_iterables(f,shapes,iterables)

def _apply_input_helper(args,kwargs,cls,iterables):
	"""
	Removes the AD information from some function input, and provides the correspondance.
	"""
	corresp = []
	def _make_arg(a):
		nonlocal corresp
		if is_ad(a):
			assert isinstance(a,cls)
			a_value = np.array(a)
			corresp.append((a,a_value))
			return a_value
		else:
			return a
	_args = tuple(map_iterables(_make_arg,val,iterables) for val in args)
	_kwargs = {key:map_iterables(_make_arg,val,iterables) for key,val in kwargs.items()}
	return _args,_kwargs,corresp


def sumprod(u,v,iterables,to_first=False):
	acc=0.
	def f(u,v):
		nonlocal acc
		if u is not None: 
			U = u.to_first() if to_first else u
			acc=acc+(U*v).sum()
	map_iterables2(f,u,v,iterables)
	return acc
	"""
	if u is None: return 0.
	elif isinstance(u,iterables): 
		if isinstance(u,dict):
			return sum(sumprod(u[key],v[key],iterables) for key in u)
		else:
			return sum(sumprod(x,y,iterables) for (x,y) in zip(u,v))
	else: 
		return (u*v).sum()
	"""

def reverse_mode(co_output):
	if co_output is None: 
		return "Forward"
	else:
		assert isinstance(co_output,pair)
		c,_ = co_output
		if isinstance(c,pair):
			return "Reverse2"
		else: 
			return "Reverse"

# ----- Functionnal -----

def recurse(step,niter=1):
	def operator(rhs):
		nonlocal step,niter
		for i in range(niter):
			rhs=step(rhs)
		return rhs
	return operator

# ------- Common functions -------

def flatten(a):
	return a.flatten() if isinstance(a,np.ndarray) else np.array([a])

def spsolve(mat,rhs):
	"""
	Solves a sparse linear system where the matrix is given as triplets.
	"""
	import scipy.sparse; import scipy.sparse.linalg
	return scipy.sparse.linalg.spsolve(
	scipy.sparse.coo_matrix(mat).tocsr(),rhs)

def spapply(mat,rhs,crop_rhs=False):
	"""
	Applies a sparse matrix, given as triplets, to an rhs.
	"""
	import scipy.sparse; import scipy.sparse.linalg
	if crop_rhs: 
		cols = mat[1][1]
		if len(cols)==0: 
			return np.zeros(0)
		size = 1+np.max(cols)
		if rhs.shape[0]>size:
			rhs = rhs[:size]
	return scipy.sparse.coo_matrix(mat).tocsr()*rhs


def min(array,axis=None,keepdims=False,out=None):
	if axis is None: return array.flatten().min(axis=0,out=out)
	ai = np.expand_dims(np.argmin(array.value, axis=axis), axis=axis)
	out = np.take_along_axis(array,ai,axis=axis)
	if not keepdims: out = out.reshape(array.shape[:axis]+array.shape[axis+1:])
	return out

def max(array,axis=None,keepdims=False,out=None):
	if axis is None: return array.flatten().max(axis=0,out=out)
	ai = np.expand_dims(np.argmax(array.value, axis=axis), axis=axis)
	out = np.take_along_axis(array,ai,axis=axis)
	if not keepdims: out = out.reshape(array.shape[:axis]+array.shape[axis+1:])
	return out

def add(a,b,out=None,where=True): 
	if out is None: return a+b if is_ad(a) else b+a
	else: result=_tuple_first(out); result[where]=a[where]+_getitem(b,where); return result

def subtract(a,b,out=None,where=True):
	if out is None: return a-b if is_ad(a) else b.__rsub__(a) 
	else: result=_tuple_first(out); result[where]=a[where]-_getitem(b,where); return result

def multiply(a,b,out=None,where=True):
	if out is None: return a*b if is_ad(a) else b*a
	else: result=_tuple_first(out); result[where]=a[where]*_getitem(b,where); return result

def true_divide(a,b,out=None,where=True): 
	if out is None: return a/b if is_ad(a) else b.__rtruediv__(a)
	else: result=_tuple_first(out); result[where]=a[where]/_getitem(b,where); return result


def maximum(a,b): 	
	from . import where
	return where(a>b,a,b)
def minimum(a,b): 	
	from . import where
	return where(a<b,a,b)


def prod(arr,axis=None,dtype=None,out=None,keepdims=False,initial=None):
	"""Attempt to reproduce numpy prod function. (Rather inefficiently, and I presume partially)"""

	shape_orig = arr.shape

	if axis is None:
		arr = arr.flatten()
		axis = (0,)
	elif isinstance(axis,numbers.Number):
		axis=(axis,)


	if axis!=(0,):
		d = len(axis)
		rd = tuple(range(len(axis)))
		arr = np.moveaxis(arr,axis,rd)
		shape1 = (np.prod(arr.shape[d:],dtype=int),)+arr.shape[d:]
		arr = arr.reshape(shape1)

	if len(arr)==0:
		return initial

	if dtype!=arr.dtype and dtype is not None:
		if initial is None:
			initial = dtype(1)
		elif dtype!=initial.dtype:
			initial = initial*dtype(1)


	out = functools.reduce(operator.mul,arr) if initial is None else functools.reduce(operator.mul,arr,initial)

	if keepdims:
		shape_kept = tuple(1 if i in axis else ai for i,ai in enumerate(shape_orig)) if out.size>1 else (1,)*len(shape_orig)
		out = out.reshape(shape_kept) 

	return out



# Elementary functions and their derivatives

def pow1(x,n):	return (x**n,n*x**(n-1))
def pow2(x,n):	return (x**n,n*x**(n-1),(n*(n-1))*x**(n-2))

def log1(x): 	return (np.log(x),1./x)
def log2(x):	y=1./x; return (np.log(x),y,-y**2)

def exp1(x): 	e=np.exp(x); return (e,e)
def exp2(x): 	e=np.exp(x); return (e,e,e)

def abs1(x):	return (np.abs(x),np.sign(x))
def abs2(x):	return (np.abs(x),np.sign(x),np.zeros(x.shape))

def sin1(x):	return (np.sin(x),np.cos(x))
def sin2(x):	s=np.sin(x); return (s,np.cos(x),-s)

def cos1(x): 	return (np.cos(x),-np.sin(x))
def cos2(x):	c=np.cos(x); return (c,-np.sin(x),-c)

def tan1(x):	t=np.tan(x); return (t,1.+t**2)
def tan2(x):	t=np.tan(x); u=1.+t**2; return (t,u,2.*u*t)

def arcsin1(x): return (np.arcsin(x),(1.-x**2)**-0.5)
def arcsin2(x): y=1.-x**2; return (np.arcsin(x),y**-0.5,x*y**-1.5)

def arccos1(c): return (np.arccos(x),-(1.-x**2)**-0.5)
def arccos2(c): y=1.-x**2; return (np.arccos(x),-y**-0.5,-x*y**-1.5)

def _arctan1(x): return (np.arctan(x),1./(1+x**2))
def _arctan2(x): y=1./(1.+x**2); return (np.arctan(x),y,-2.*x*y**2)

# No implementation of arctan2, or hypot, which have two args

def sinh1(x):	return (np.sinh(x),np.cosh(x))
def sinh2(x):	s=np.sinh(x); return (s,np.cosh(x),s)

def cosh1(x):	return (np.cosh(x),np.sinh(x))
def cosh2(x):	c=np.cosh(x); return (c,np.sinh(x),c)

def tanh1(x):	t=np.tanh(x); return (t,1.-t**2)
def tanh2(x):	t=np.tanh(x); u=1.-t**2; return (t,u,-2.*u*t)

def arcsinh1(x): return (np.arcsinh(x),(1.+x**2)**-0.5)
def arcsinh2(x): y=1.+x**2; return (np.arcsinh(x),y**-0.5,-x*y**-1.5)

def arccosh1(c): return (np.arccos(x),(x**2-1.)**-0.5)
def arccosh2(c): y=x**2-1.; return (np.arccos(x),y**-0.5,-x*y**-1.5)

def _arctanh1(x): return (np.arctan(x),1./(1-x**2))
def _arctanh2(x): y=1./(1-x**2); return (np.arctan(x),y,2.*x*y**2)


