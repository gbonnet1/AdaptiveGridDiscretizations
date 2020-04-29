import numpy as np
import numbers
import functools
import operator

from . import numpy_like as npl
from .ad_generic import is_ad,asarray
from . import cupy_generic
from . import functional
# Elementary functions and their derivatives
# No implementation of arctan2, or hypot, which have two args
class Taylor1: # first order Taylor expansions
	def pow(x,n):	return (x**n,n*x**(n-1))
	def log(x): 	return (np.log(x),1./x)
	def exp(x): 	e=np.exp(x); return (e,e)
	def abs(x):	return (np.abs(x),np.sign(x))
	def sin(x):	return (np.sin(x),np.cos(x))
	def cos(x): 	return (np.cos(x),-np.sin(x))
	def tan(x):	t=np.tan(x); return (t,1.+t**2)
	def arcsin(x): return (np.arcsin(x),(1.-x**2)**-0.5)
	def arccos(c): return (np.arccos(x),-(1.-x**2)**-0.5)
	def arctan(x): return (np.arctan(x),1./(1+x**2))
	def sinh(x):	return (np.sinh(x),np.cosh(x))
	def cosh(x):	return (np.cosh(x),np.sinh(x))
	def tanh(x):	t=np.tanh(x); return (t,1.-t**2)
	def arcsinh(x): return (np.arcsinh(x),(1.+x**2)**-0.5)
	def arccosh(c): return (np.arccos(x),(x**2-1.)**-0.5)
	def arctanh(x): return (np.arctanh(x),1./(1-x**2))

class Taylor2: # second order Taylor expansions of classical functions
	def pow(x,n):	return (x**n,n*x**(n-1),(n*(n-1))*x**(n-2))
	def log(x):	y=1./x; return (np.log(x),y,-y**2)
	def exp(x): 	e=np.exp(x); return (e,e,e)
	def abs(x):	return (np.abs(x),np.sign(x),np.zeros_like(x))
	def sin(x):	s=np.sin(x); return (s,np.cos(x),-s)
	def cos(x):	c=np.cos(x); return (c,-np.sin(x),-c)
	def tan(x):	t=np.tan(x); u=1.+t**2; return (t,u,2.*u*t)
	def arcsin(x): y=1.-x**2; return (np.arcsin(x),y**-0.5,x*y**-1.5)
	def arccos(c): y=1.-x**2; return (np.arccos(x),-y**-0.5,-x*y**-1.5)
	def arctan(x): y=1./(1.+x**2); return (np.arctan(x),y,-2.*x*y**2)
	def sinh(x):	s=np.sinh(x); return (s,np.cosh(x),s)
	def cosh(x):	c=np.cosh(x); return (c,np.sinh(x),c)
	def tanh(x):	t=np.tanh(x); u=1.-t**2; return (t,u,-2.*u*t)
	def arcsinh(x): y=1.+x**2; return (np.arcsinh(x),y**-0.5,-x*y**-1.5)
	def arccosh(c): y=x**2-1.; return (np.arccos(x),y**-0.5,-x*y**-1.5)
	def arctanh(x): y=1./(1-x**2); return (np.arctanh(x),y,2.*x*y**2)

def _tuple_first(a): 	return a[0] if isinstance(a,tuple) else a
def _getitem(a,where):  return a if (where is True and not cupy_generic.isndarray(a)) else a[where]

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

def maximum(a,b): return np.where(a>b,a,b)
def minimum(a,b): return np.where(a<b,a,b)

class baseAD:

	def _init_cupy(self):
		if self.cupy_based():
			import cupy as cp
			baseAD_cupy = functional.class_rebase(baseAD,(cp.ndarray,),"baseAD_cupy")
			x = cp.array([np.nan],dtype=np.float32)
			super(baseAD_cupy,self).__init__(**cupy_generic.cupy_init_kwargs(x))

	@property
	def shape(self): return self.value.shape
	@property
	def ndim(self): return self.value.ndim
	@property
	def size(self): return self.value.size	
	def flatten(self):	return self.reshape( (self.size,) )
	def squeeze(self,axis=None): return self.reshape(self.value.squeeze(axis).shape)
	@property
	def T(self):	return self if self.ndim<2 else self.transpose()

	@classmethod
	def stack(cls,elems,axis=0):
		return cls.concatenate(tuple(npl.expand_dims(e,axis=axis) for e in elems),axis)

	@property
	def dtype(self): return self.value.dtype
	def __len__(self): return len(self.value)
	def _ndarray(self): return type(self.value)
	def cupy_based(self): return self._ndarray() is not np.ndarray
	def isndarray(self,other): return isinstance(other,self._ndarray()) # same array module
	@classmethod
	def is_ad(cls,other): return isinstance(other,cls)
	@classmethod
	def new(cls,*args,**kwargs):
		return cls(*args,**kwargs)	

	@classmethod
	def Taylor(cls): return Taylor1 if cls.order()==1 else Taylor2

	def sqrt(self):			return self**0.5
	def __pow__(self,n):	return self._math_helper(self.Taylor().pow(self.value,n))
	def log(self):			return self._math_helper(self.Taylor().log(self.value))
	def exp(self):			return self._math_helper(self.Taylor().exp(self.value))
	def abs(self):			return self._math_helper(self.Taylor().abs(self.value))
	def sin(self):			return self._math_helper(self.Taylor().sin(self.value))
	def cos(self):			return self._math_helper(self.Taylor().cos(self.value))
	def tan(self):			return self._math_helper(self.Taylor().tan(self.value))
	def arcsin(self):		return self._math_helper(self.Taylor().arcsin(self.value))
	def arccos(self):		return self._math_helper(self.Taylor().arccos(self.value))
	def arctan(self):		return self._math_helper(self.Taylor().arctan(self.value))
	def sinh(self):			return self._math_helper(self.Taylor().sinh(self.value))
	def cosh(self):			return self._math_helper(self.Taylor().cosh(self.value))
	def tanh(self):			return self._math_helper(self.Taylor().tanh(self.value))
	def arcsinh(self):		return self._math_helper(self.Taylor().arcsinh(self.value))
	def arccosh(self):		return self._math_helper(self.Taylor().arccosh(self.value))
	def arctanh(self):		return self._math_helper(self.Taylor().arctanh(self.value))

		# See https://docs.scipy.org/doc/numpy/reference/ufuncs.html
	def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):

		# Return an np.ndarray for piecewise constant functions
		if ufunc in [
		# Comparison functions
		np.greater,np.greater_equal,
		np.less,np.less_equal,
		np.equal,np.not_equal,

		# Math
		np.floor_divide,np.rint,np.sign,np.heaviside,

		# Floating functions
		np.isfinite,np.isinf,np.isnan,np.isnat,
		np.signbit,np.floor,np.ceil,np.trunc
		]:
			inputs_ = (a.value if self.is_ad(a) else a for a in inputs)
			return self.value.__array_ufunc__(ufunc,method,*inputs_,**kwargs)


		if method=="__call__":

			# Reimplemented
			if ufunc==np.maximum: return maximum(*inputs,**kwargs)
			if ufunc==np.minimum: return minimum(*inputs,**kwargs)

			# Math functions
			if ufunc==np.sqrt:		return self.sqrt()
			if ufunc==np.log:		return self.log()
			if ufunc==np.exp:		return self.exp()
			if ufunc==np.abs:		return self.abs()
			if ufunc==np.sin:		return self.sin()
			if ufunc==np.cos:		return self.cos()
			if ufunc==np.tan:		return self.tan()
			if ufunc==np.arcsin:	return self.arcsin()
			if ufunc==np.arccos:	return self.arccos()
			if ufunc==np.arctan:	return self.arctan()
			if ufunc==np.sinh:		return self.sinh()
			if ufunc==np.cosh:		return self.cosh()
			if ufunc==np.tanh:		return self.tanh()
			if ufunc==np.arcsinh:	return self.arcsinh()
			if ufunc==np.arccosh:	return self.arccosh()
			if ufunc==np.arctanh:	return self.arctanh()

			# Operators
			if ufunc==np.add: return add(*inputs,**kwargs)
			if ufunc==np.subtract: return subtract(*inputs,**kwargs)
			if ufunc==np.multiply: return multiply(*inputs,**kwargs)
			if ufunc==np.true_divide: return true_divide(*inputs,**kwargs)


		return NotImplemented

	def __array_function__(self,func,types,args,kwargs):
		return npl._array_function_overload(self,func,types,args,kwargs)


	# Support for +=, -=, *=, /=, <, <=, >, >=, ==, !=
	def __iadd__(self,other): return add(self,other,out=self)
	def __isub__(self,other): return subtract(self,other,out=self)
	def __imul__(self,other): return multiply(self,other,out=self)
	def __itruediv__(self,other): return true_divide(self,other,out=self)

	def __lt__(self,other): return np.less(self,other)
	def __le__(self,other): return np.less_equal(self,other)
	def __gt__(self,other): return np.greater(self,other)
	def __ge__(self,other): return np.greater_equal(self,other)
	def __eq__(self,other): return np.equal(self,other)
	def __ne__(self,other): return np.not_equal(self,other)

	def argmin(self,*args,**kwargs): return self.value.argmin(*args,**kwargs)
	def argmax(self,*args,**kwargs): return self.value.argmax(*args,**kwargs)

	def min(array,axis=None,keepdims=False,out=None):
		if axis is None: return array.reshape(-1).min(axis=0,out=out)
		ai = npl.expand_dims(np.argmin(array.value, axis=axis), axis=axis)
		out = np.take_along_axis(array,ai,axis=axis)
		if not keepdims: out = out.reshape(array.shape[:axis]+array.shape[axis+1:])
		return out

	def max(array,axis=None,keepdims=False,out=None):
		if axis is None: return array.reshape(-1).max(axis=0,out=out)
		ai = npl.expand_dims(np.argmax(array.value, axis=axis), axis=axis)
		out = np.take_along_axis(array,ai,axis=axis)
		if not keepdims: out = out.reshape(array.shape[:axis]+array.shape[axis+1:])
		return out

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


# ---- cupy ad needs cp.ndarray as base class ----

def _new(cls):
	@functools.wraps(cls.__init__)
	def new(value,*args,**kwargs): 
		value = asarray(value)
		if cupy_generic.from_cupy(value):
			import cupy as cp
			baseAD_cupy = functional.class_rebase(baseAD,(cp.ndarray,),"baseAD_cupy")
			cls_cupy = functional.class_rebase(cls,(baseAD_cupy,),cls.__name__+"_cupy")
			return cls_cupy(value,*args,**kwargs)
		else:
			return cls(value,*args,**kwargs)
	return new
