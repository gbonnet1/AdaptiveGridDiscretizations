# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .cupy_generic import cupy_init_kwargs,cupy_rebase
from . import ad_generic
from . import cupy_support as npl
from . import numpy_like
from . import misc
from . import Dense

_add_dim = misc._add_dim; _add_dim2 = misc._add_dim2; _add_coef=misc._add_coef;

class denseAD2(np.ndarray):
	"""
	A class for dense forward second order automatic differentiation
	"""

	# Construction
	# See : https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
	def __new__(cls,value,coef1=None,coef2=None,broadcast_ad=False):
		# Case where one should just reproduce value
		if cls.is_ad(value):
			assert coef1 is None and coef2 is None 
			if cls.cupy_based: value,coef1,coef2 = value.value,value.coef1,value.coef2
			else: return value
		if ad_generic.is_ad(value):
			raise ValueError("Attempting to cast between different AD types")

		# Create instance 
		value = ad_generic.asarray(value)
		if cls.cupy_based():
			denseAD2_cupy = cupy_rebase(denseAD2)
			obj = super(denseAD2_cupy,cls).__new__(cls,**cupy_init_kwargs(value))
		else: 
			obj = value.view(denseAD2)

		# Add attributes
		shape = value.shape
		shape1 = shape+(0,)
		shape2 = shape+(0,0)
		obj.coef1 = (npl.zeros_like(value,shape=shape1) if coef1 is None 
			else misc._test_or_broadcast_ad(coef1,shape,broadcast_ad))
		obj.coef2 = (npl.zeros_like(value,shape=shape2) if coef2 is None 
			else misc._test_or_broadcast_ad(coef2,shape,broadcast_ad,2))
		return obj

	def __init__(self,value,*args,**kwargs):
		if self.cupy_based():
			if self.is_ad(value): value = value.value
			denseAD2_cupy = cupy_rebase(denseAD2)
			super(denseAD2_cupy,self).__init__(**cupy_init_kwargs(value))

	@classmethod
	def _ndarray(cls):
		return cls.__bases__[0]
	@classmethod
	def cupy_based(cls):
		return cls._ndarray() is not np.ndarray
	@classmethod
	def isndarray(cls,other): 
		"""Wether argument is an ndarray from a compatible module (numpy or cupy)"""
		return isinstance(other,cls._ndarray())
	@classmethod
	def is_ad(cls,other): return isinstance(other,cls)

	@classmethod
	def new(cls,*args,**kwargs):
		return cls(*args,**kwargs)
	@property
	def value(self): 
		"""Returns the base ndarray, without AD information"""
		if self.cupy_based(): return self.view()
		else: return self.view(np.ndarray)
	def copy(self,order='C'):
		return self.new(self.value.copy(order=order),
			self.coef1.copy(order=order),self.coef2.copy(order=order))

	# Representation 
	def __iter__(self):
		for value,coef1,coef2 in zip(self.value,self.coef1,self.coef2):
			yield self.new(value,coef1,coef2)

	def __str__(self):
		return "denseAD2("+str(self.value)+","+misc._prep_nl(str(self.coef1))+","+misc._prep_nl(str(self.coef2)) +")"
	def __repr__(self):
		return "denseAD2("+repr(self.value)+","+misc._prep_nl(repr(self.coef1))+","+misc._prep_nl(repr(self.coef2)) +")"

	# Operators
	def __add__(self,other):
		if self.is_ad(other):
			return self.new(self.value+other.value,_add_coef(self.coef1,other.coef1),_add_coef(self.coef2,other.coef2))
		else:
			return self.new(self.value+other, self.coef1, self.coef2, broadcast_ad=True)

	def __sub__(self,other):
		if self.is_ad(other):
			return self.new(self.value-other.value,_add_coef(self.coef1,-other.coef1),_add_coef(self.coef2,-other.coef2))
		else:
			return self.new(self.value-other, self.coef1, self.coef2, broadcast_ad=True)

	def __mul__(self,other):
		if self.is_ad(other):
			mixed = npl.expand_dims(self.coef1,axis=-1)*npl.expand_dims(other.coef1,axis=-2)
			return self.new(self.value*other.value, _add_coef(_add_dim(self.value)*other.coef1,_add_dim(other.value)*self.coef1),
				_add_coef(_add_coef(_add_dim2(self.value)*other.coef2,_add_dim2(other.value)*self.coef2),_add_coef(mixed,np.moveaxis(mixed,-2,-1))))
		elif self.isndarray(other):
			return self.new(self.value*other,_add_dim(other)*self.coef1,_add_dim2(other)*self.coef2)
		else:
			return self.new(self.value*other,other*self.coef1,other*self.coef2)

	def __truediv__(self,other):
		if self.is_ad(other):
			return self.__mul__(other.__pow__(-1))
		elif self.isndarray(other):
			inv = 1./other
			return self.new(self.value*inv,_add_dim(inv)*self.coef1,_add_dim2(inv)*self.coef2)
		else:
			inv = 1./other
			return self.new(self.value*inv,self.coef1*inv,self.coef2*inv)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other):	return self.__pow__(-1).__mul__(other)


	def __neg__(self):		return self.new(-self.value,-self.coef1,-self.coef2)

	# Math functions
	def _math_helper(self,deriv): # Inputs : a=f(x), b=f'(x), c=f''(x), where x=self.value
		a,b,c=deriv
		mixed = npl.expand_dims(self.coef1,axis=-1)*npl.expand_dims(self.coef1,axis=-2)
		return self.new(a,_add_dim(b)*self.coef1,_add_dim2(b)*self.coef2+_add_dim2(c)*mixed)
	
	def sqrt(self):			return self**0.5
	def __pow__(self,n):	return self._math_helper(misc.pow2(self.value,n))
	def log(self):			return self._math_helper(misc.log2(self.value))
	def exp(self):			return self._math_helper(misc.exp2(self.value))
	def abs(self):			return self._math_helper(misc.abs2(self.value))
	def sin(self):			return self._math_helper(misc.sin2(self.value))
	def cos(self):			return self._math_helper(misc.cos2(self.value))
	def tan(self):			return self._math_helper(misc.tan2(self.value))
	def arcsin(self):		return self._math_helper(misc.arcsin2(self.value))
	def arccos(self):		return self._math_helper(misc.arccos2(self.value))
	def arctan(self):		return self._math_helper(misc._arctan2(self.value))
	def sinh(self):			return self._math_helper(misc.sinh2(self.value))
	def cosh(self):			return self._math_helper(misc.cosh2(self.value))
	def tanh(self):			return self._math_helper(misc.tanh2(self.value))
	def arcsinh(self):		return self._math_helper(misc.arcsinh2(self.value))
	def arccosh(self):		return self._math_helper(misc.arccosh2(self.value))
	def arctanh(self):		return self._math_helper(misc._arctanh2(self.value))

	@classmethod
	def compose(cls,a,t):
		assert cls.is_ad(a) and all(cls.is_ad(b) for b in t)
		b = np.moveaxis(cls.concatenate(t,axis=0),0,-1)
		coef1 = (_add_dim(a.coef1)*b.coef1).sum(axis=-2)
		coef2_pure = (_add_dim2(a.coef1)*b.coef2).sum(axis=-3)
		shape_factor = b.shape[:-1]
		mixed = ( b.coef1.reshape(shape_factor+(a.size_ad,1,b.size_ad,1))
			     *b.coef1.reshape(shape_factor+(1,a.size_ad,1,b.size_ad)) )
		coef2_mixed = (_add_dim2(a.coef2)*mixed).sum(axis=-3).sum(axis=-3)
		return cls.new(a.value,coef1,coef2_pure+coef2_mixed)

	#Indexing
	@property
	def size_ad(self):  return self.coef1.shape[-1]

	def to_first(self): return Dense.denseAD(self.value,self.coef1)
	def gradient(self,i=None): 
		"""Returns the gradient, or the i-th component of the gradient if specified."""
		return np.moveaxis(self.coef1,-1,0) if i is None else self.coef1[...,i]
	def hessian(self,i=None,j=None): 
		"""Returns the hessian, or component (i,j) of the hessian if specified."""
		return np.moveaxis(self.coef2,(-2,-1),(0,1)) if i is None else self.coef2[...,i,j]

	def __getitem__(self,key):
		ekey1,ekey2 = misc.key_expand(key,1),misc.key_expand(key,2)
		return self.new(self.value[key], self.coef1[ekey1], self.coef2[ekey2])

	def __setitem__(self,key,other):
		ekey1,ekey2 = misc.key_expand(key,1),misc.key_expand(key,2)
		if self.is_ad(other):
			osad = other.size_ad
			if osad==0: return self.__setitem__(key,other.value)
			elif self.size_ad==0: 
				self.coef1=npl.zeros_like(self.value,shape=self.coef1.shape[:-1]+(osad,))
				self.coef2=npl.zeros_like(self.value,shape=self.coef2.shape[:-2]+(osad,osad))
			self.value[key] = other.value
			self.coef1[ekey1] = other.coef1
			self.coef2[ekey2] = other.coef2
		else:
			self.value[key] = other
			self.coef1[ekey1] = 0.
			self.coef2[ekey2] = 0.


	def reshape(self,shape,order='C'):
		shape = misc._to_tuple(shape)
		shape1 = shape+(self.size_ad,)
		shape2 = shape+(self.size_ad,self.size_ad)
		return self.new(self.value.reshape(shape,order=order),
			self.coef1.reshape(shape1,order=order), self.coef2.reshape(shape2,order=order))

	def flatten(self):	return self.reshape( (self.size,) )
	def squeeze(self,axis=None): return self.reshape(self.value.squeeze(axis).shape)

	def broadcast_to(self,shape):
		shape1 = shape+(self.size_ad,)
		shape2 = shape+(self.size_ad,self.size_ad)
		return self.new(np.broadcast_to(self.value,shape), 
			np.broadcast_to(self.coef1,shape1), np.broadcast_to(self.coef2,shape2))

	def pad(self,pad_width,*args,constant_values=0,**kwargs):
		return self.new(
			np.pad(self.value,pad_width,*args,constant_values=constant_values,**kwargs),
			np.pad(self.coef1,pad_width+((0,0),),*args,constant_values=0,**kwargs),
			np.pad(self.coef2,pad_width+((0,0),(0,0),),*args,constant_values=0,**kwargs),)

	@property
	def T(self):	return self if self.ndim<2 else self.transpose()
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes1 = tuple(axes) +(self.ndim,)
		axes2 = tuple(axes) +(self.ndim,self.ndim+1)
		return self.new(self.value.transpose(axes),
			self.coef1.transpose(axes1),self.coef2.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		if axis<0: axis=self.ndim+axis
		out = self.new(self.value.sum(axis,**kwargs), 
			self.coef1.sum(axis,**kwargs), self.coef2.sum(axis,**kwargs))
		return out

	prod = misc.prod

	def min(self,*args,**kwargs): return misc.min(self,*args,**kwargs)
	def max(self,*args,**kwargs): return misc.max(self,*args,**kwargs)
	def argmin(self,*args,**kwargs): return self.value.argmin(*args,**kwargs)
	def argmax(self,*args,**kwargs): return self.value.argmax(*args,**kwargs)

	def sort(self,*varargs,**kwargs):
		from . import sort
		self=sort(self,*varargs,**kwargs)


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
			if ufunc==np.maximum: return misc.maximum(*inputs,**kwargs)
			if ufunc==np.minimum: return misc.minimum(*inputs,**kwargs)

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
			if ufunc==np.add: return self.add(*inputs,**kwargs)
			if ufunc==np.subtract: return self.subtract(*inputs,**kwargs)
			if ufunc==np.multiply: return self.multiply(*inputs,**kwargs)
			if ufunc==np.true_divide: return self.true_divide(*inputs,**kwargs)


		return NotImplemented
	
	def __array_function__(self,func,types,args,kwargs):
		return numpy_like._array_function_overload(self,func,types,args,kwargs)

	# Static methods

	# Support for +=, -=, *=, /=
	@staticmethod
	def add(*args,**kwargs): return misc.add(*args,**kwargs)
	@staticmethod
	def subtract(*args,**kwargs): return misc.subtract(*args,**kwargs)
	@staticmethod
	def multiply(*args,**kwargs): return misc.multiply(*args,**kwargs)
	@staticmethod
	def true_divide(*args,**kwargs): return misc.true_divide(*args,**kwargs)

	@classmethod
	def stack(cls,elems,axis=0):
		return cls.concatenate(tuple(npl.expand_dims(e,axis=axis) for e in elems),axis)

	@classmethod
	def concatenate(cls,elems,axis=0):
		axis1,axis2 = (axis,axis) if axis>=0 else (axis-1,axis-2)
		elems2 = tuple(cls(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		assert all((e.size_ad==size_ad or e.size_ad==0) for e in elems2)
		return cls( 
		np.concatenate(tuple(e.value for e in elems2), axis=axis), 
		np.concatenate(tuple(e.coef1 if e.size_ad==size_ad else 
			npl.zeros_like(e.coef1,shape=e.shape+(size_ad,)) for e in elems2),axis=axis1),
		np.concatenate(tuple(e.coef2 if e.size_ad==size_ad else 
			npl.zeros_like(e.coef2,shape=e.shape+(size_ad,size_ad)) for e in elems2),axis=axis2))

	def apply_linear_operator(self,op):
		return self.new(op(self.value),
			misc.apply_linear_operator(op,self.coef1,flatten_ndim=1),
			misc.apply_linear_operator(op,self.coef2,flatten_ndim=2))
# -------- End of class denseAD2 -------

# -------- Factory method -----

new = ad_generic._new(denseAD2) # Factory function

def identity(*args,**kwargs):
	arr = Dense.identity(*args,**kwargs)
	return new(arr.value,arr.coef,
		npl.zeros_like(arr.value,shape=arr.shape+(arr.size_ad,arr.size_ad)))

def register(*args,**kwargs):
	return Dense.register(*args,**kwargs,ident=identity)
