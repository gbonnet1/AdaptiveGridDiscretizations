import numpy as np
from . import misc
from . import Dense

_add_dim = misc._add_dim; _pad_last = misc._pad_last; _concatenate=misc._concatenate;

class spAD(np.ndarray):
	"""
	A class for sparse forward automatic differentiation
	"""

	# Construction
	# See : https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
	def __new__(cls,value,coef=None,index=None,broadcast_ad=False):
		if isinstance(value,spAD):
			assert coef is None and index is None
			return value
		obj = np.asarray(value).view(spAD)
		shape = obj.shape
		shape2 = shape+(0,)
		assert ((coef is None) and (index is None)) or (coef.shape==index.shape)
		obj.coef  = (np.full(shape2,0.) if coef is None 
			else misc._test_or_broadcast_ad(coef,shape,broadcast_ad) ) 
		obj.index = (np.full(shape2,0)  if index is None 
			else misc._test_or_broadcast_ad(index,shape,broadcast_ad) )
		return obj

#	def __array_finalize__(self,obj): pass

	def copy(self,order='C'):
		return spAD(self.value.copy(order=order),self.coef.copy(order=order),self.index.copy(order=order))
	def __copy__(self): return self.copy(order='K')
	def __deepcopy__(self,*args): 
		return spAD(self.value.__deepcopy__(*args),self.coef.__deepcopy__(*args),self.index.__deepcopy__(*args))

	# Representation 
	def __iter__(self):
		for value,coef,index in zip(self.value,self.coef,self.index):
			yield spAD(value,coef,index)

	def __str__(self):
		return "spAD"+str((self.value,self.coef,self.index))
	def __repr__(self):
		return "spAD"+repr((self.value,self.coef,self.index))	

	# Operators
	def __add__(self,other):
		if isinstance(other,spAD):
			value = self.value+other.value
			return spAD(value, _concatenate(self.coef,other.coef,value.shape), _concatenate(self.index,other.index,value.shape))
		else:
			return spAD(self.value+other, self.coef, self.index, broadcast_ad=True)

	def __sub__(self,other):
		if isinstance(other,spAD):
			value = self.value-other.value
			return spAD(self.value-other.value, _concatenate(self.coef,-other.coef,value.shape), _concatenate(self.index,other.index,value.shape))
		else:
			return spAD(self.value-other, self.coef, self.index, broadcast_ad=True)

	def __mul__(self,other):
		if isinstance(other,spAD):
			value = self.value*other.value
			coef1,coef2 = _add_dim(other.value)*self.coef,_add_dim(self.value)*other.coef
			index1,index2 = np.broadcast_to(self.index,coef1.shape),np.broadcast_to(other.index,coef2.shape)
			return spAD(value,_concatenate(coef1,coef2),_concatenate(index1,index2))
		elif isinstance(other,np.ndarray):
			value = self.value*other
			coef = _add_dim(other)*self.coef
			index = np.broadcast_to(self.index,coef.shape)
			return spAD(value,coef,index)
		else:
			return spAD(self.value*other,other*self.coef,self.index)

	def __truediv__(self,other):
		if isinstance(other,spAD):
			return spAD(self.value/other.value,
				_concatenate(self.coef*_add_dim(1/other.value),other.coef*_add_dim(-self.value/other.value**2)),
				_concatenate(self.index,other.index))
		elif isinstance(other,np.ndarray):
			return spAD(self.value/other,self.coef*_add_dim(1./other),self.index)
		else:
			return spAD(self.value/other,self.coef/other,self.index)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other): 	
		value = other/self.value
		coef = self.coef*_add_dim(-other/self.value**2)
		index = np.broadcast_to(self.index,coef.shape)
		return spAD(value,coef,index)

	def __neg__(self):		return spAD(-self.value,-self.coef,self.index)

	# Math functions
	def _math_helper(self,deriv):
		a,b=deriv
		return spAD(a,_add_dim(b)*self.coef,self.index)

	def sqrt(self):			return self**0.5
	def __pow__(self,n):	return self._math_helper(misc.pow1(self.value,n))
	def log(self):			return self._math_helper(misc.log1(self.value))
	def exp(self):			return self._math_helper(misc.exp1(self.value))
	def abs(self):			return self._math_helper(misc.abs1(self.value))
	def sin(self):			return self._math_helper(misc.sin1(self.value))
	def cos(self):			return self._math_helper(misc.cos1(self.value))
	def tan(self):			return self._math_helper(misc.tan1(self.value))
	def arcsin(self):		return self._math_helper(misc.arcsin1(self.value))
	def arccos(self):		return self._math_helper(misc.arccos1(self.value))
	def arctan(self):		return self._math_helper(misc._arctan1(self.value))
	def sinh(self):			return self._math_helper(misc.sinh1(self.value))
	def cosh(self):			return self._math_helper(misc.cosh1(self.value))
	def tanh(self):			return self._math_helper(misc.tanh1(self.value))
	def arcsinh(self):		return self._math_helper(misc.arcsinh1(self.value))
	def arccosh(self):		return self._math_helper(misc.arccosh1(self.value))
	def arctanh(self):		return self._math_helper(misc._arctanh1(self.value))

	@staticmethod
	def compose(a,t):
		assert isinstance(a,Dense.denseAD) and all(isinstance(b,spAD) for b in t)
		lens = tuple(len(b) for b in t)
		assert a.size_ad == sum(lens)
		t = tuple(np.moveaxis(b,0,-1) for b in t)
		a_coefs = np.split(a.coef,np.cumsum(lens[:-1]),axis=-1)
		def FlattenLast2(arr): return arr.reshape(arr.shape[:-2]+(np.prod(arr.shape[-2:],dtype=int),))
		coef = tuple(_add_dim(c)*b.coef for c,b in zip(a_coefs,t) )
		coef = np.concatenate( tuple(FlattenLast2(c) for c in coef), axis=-1)
		index = np.broadcast_to(np.concatenate( tuple(FlattenLast2(b.index) for b in t), axis=-1),coef.shape)
		return spAD(a.value,coef,index)

	#Indexing
	@property
	def value(self): return self.view(np.ndarray)
	@property
	def size_ad(self):  return self.coef.shape[-1]

	def __getitem__(self,key):
		ekey = misc.key_expand(key)
		return spAD(self.value[key], self.coef[ekey], self.index[ekey])

	def __setitem__(self,key,other):
		ekey = misc.key_expand(key)
		if isinstance(other,spAD):
			self.value[key] = other.value
			pad_size = max(self.coef.shape[-1],other.coef.shape[-1])
			if pad_size>self.coef.shape[-1]:
				self.coef = _pad_last(self.coef,pad_size)
				self.index = _pad_last(self.index,pad_size)
			self.coef[ekey] = _pad_last(other.coef,pad_size)
			self.index[ekey] = _pad_last(other.index,pad_size)
		else:
			self.value[key] = other
			self.coef[ekey] = 0.

	def reshape(self,shape,order='C'):
		shape = misc._to_tuple(shape)
		shape2 = shape+(self.size_ad,)
		return spAD(self.value.reshape(shape,order=order),self.coef.reshape(shape2,order=order), self.index.reshape(shape2,order=order))

	def flatten(self):	return self.reshape( (self.size,) )
	def squeeze(self,axis=None): return self.reshape(self.value.squeeze(axis).shape)

	def broadcast_to(self,shape):
		shape2 = shape+(self.size_ad,)
		return spAD(np.broadcast_to(self.value,shape), np.broadcast_to(self.coef,shape2), np.broadcast_to(self.index,shape2))

	@property
	def T(self):	return self if self.ndim<2 else self.transpose()
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes) +(self.ndim,)
		return spAD(self.value.transpose(axes),self.coef.transpose(axes2),self.index.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		value = self.value.sum(axis,**kwargs)
		shape = value.shape +(self.size_ad * self.shape[axis],)
		coef = np.moveaxis(self.coef, axis,-1).reshape(shape)
		index = np.moveaxis(self.index, axis,-1).reshape(shape)
		out = spAD(value,coef,index)
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
			inputs_ = (a.value if isinstance(a,spAD) else a for a in inputs)
			return super(spAD,self).__array_ufunc__(ufunc,method,*inputs_,**kwargs)


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


	# Conversion
	def bound_ad(self):
		return 1+np.max(self.index,initial=-1)
	def to_dense(self,dense_size_ad=None):
		def mvax(arr): return np.moveaxis(arr,-1,0)
		if dense_size_ad is None: dense_size_ad = self.bound_ad()
		coef = np.zeros(self.shape+(dense_size_ad,))
		for c,i in zip(mvax(self.coef),mvax(self.index)):
			coef_new = _add_dim(c)+np.take_along_axis(coef,_add_dim(i),axis=-1)
			np.put_along_axis(coef,_add_dim(i),coef_new,axis=-1)
		return Dense.denseAD(self.value,coef)

	#Linear algebra
	def triplets(self):
		coef = self.coef.flatten()
		row = np.broadcast_to(_add_dim(np.arange(self.size).reshape(self.shape)), self.index.shape).flatten()
		column = self.index.flatten()

		pos=coef!=0
		return (coef[pos],(row[pos],column[pos]))

	def solve(self,raw=False):
		"""
		Assume that the spAD instance represents the variable y = x + A*delta,
		where delta is a symbolic perturbation. 
		Solves the system x + A*delta = 0, assuming compatible shapes.
		"""
		mat = self.triplets()
		rhs = -np.array(self).flatten()
		return (mat,rhs) if raw else misc.spsolve(mat,rhs).reshape(self.shape)

	"""
	def diagonal(self,identity_var=None):
		coef = np.moveaxis(self.coef,-1,0)
		index = np.moveaxis(self.index,-1,0)
		diag = np.zeros(self.shape)
		rg = (np.arange(self.size).reshape(self.shape)
			if identity_var is None else 
			np.squeeze(identity_var.index,axis=-1))
		for c,i in zip(coef,index):
			pos = i==rg
			diag[pos]+=c[pos]
		return diag
	"""

	def is_elliptic(self,tol=None,identity_var=None):
		"""
		Tests wether the variable encodes a (linear) degenerate elliptic scheme.
		Output :
		- sum of the coefficients at each position (must be non-negative for 
		degenerate ellipticity, positive for strict ellipticity)
		- maximum of off-diagonal coefficients at each position (must be non-positive)
		Output (if tol is specified) : 
		- min_sum >=-tol and max_off <= tol
		Side effect warning : AD simplification, which is also possibly costly
		"""
		self.simplify_ad()
		min_sum = self.coef.sum(axis=-1)

		rg = (np.arange(self.size).reshape(self.shape+(1,))
			if identity_var is None else identity_var.index)
		coef = self.coef.copy()
		coef[self.index==rg] = -np.inf
		coef[coef==0.] = -np.inf
		max_off = coef.max(axis=-1)

		if tol is None: return min_sum,max_off
		return min_sum.min()>=-tol and max_off.max()<=tol

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

	@staticmethod
	def stack(elems,axis=0):
		return spAD.concatenate(tuple(np.expand_dims(e,axis=axis) for e in elems),axis)

	@staticmethod
	def concatenate(elems,axis=0):
		axis1 = axis if axis>=0 else axis-1
		elems2 = tuple(spAD(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		return spAD( 
		np.concatenate(tuple(e.value for e in elems2), axis=axis), 
		np.concatenate(tuple(_pad_last(e.coef,size_ad)  for e in elems2),axis=axis1),
		np.concatenate(tuple(_pad_last(e.index,size_ad) for e in elems2),axis=axis1))


	# Memory optimization
	def simplify_ad(self):
		if self.size_ad==0: return # Nothing to simplify
		if len(self.shape)==0: # Add dimension to scalar-like arrays
			other = self.reshape((1,))
			other.simplify_ad()
			other = other.reshape(tuple())
			self.coef,self.index = other.coef,other.index
			return
		bad_index = np.iinfo(self.index.dtype).max
		bad_pos = self.coef==0
		self.index[bad_pos] = bad_index
		ordering = self.index.argsort(axis=-1)
		self.coef = np.take_along_axis(self.coef,ordering,axis=-1)
		self.index = np.take_along_axis(self.index,ordering,axis=-1)

		cum_coef = np.full(self.shape,0.)
		indices = np.full(self.shape,0)
		size_ad = self.size_ad
		self.coef = np.moveaxis(self.coef,-1,0)
		self.index = np.moveaxis(self.index,-1,0)
		prev_index = np.copy(self.index[0])

		for i in range(size_ad):
			 # Note : self.index, self.coef change during iterations
			ind,co = self.index[i],self.coef[i]
			pos_new_index = np.logical_and(prev_index != ind,ind!=bad_index)
			pos_old_index = np.logical_not(pos_new_index)
			prev_index[pos_new_index] = ind[pos_new_index]
			cum_coef[pos_new_index]=co[pos_new_index]
			cum_coef[pos_old_index]+=co[pos_old_index]
			indices[pos_new_index]+=1
			indices_exp = np.expand_dims(indices,axis=0)
			np.put_along_axis(self.index,indices_exp,prev_index,axis=0)
			np.put_along_axis(self.coef,indices_exp,cum_coef,axis=0)

		indices[self.index[0]==bad_index]=-1
		indices_max = np.max(indices,axis=None)
		size_ad_new = indices_max+1
		self.coef  = self.coef[:size_ad_new]
		self.index = self.index[:size_ad_new]
		if size_ad_new==0:
			self.coef  = np.moveaxis(self.coef,0,-1)
			self.index = np.moveaxis(self.index,0,-1)
			return

		coef_end  = self.coef[ np.maximum(indices_max,0)]
		index_end = self.index[np.maximum(indices_max,0)]
		coef_end[ indices<indices_max] = 0.
		index_end[indices<indices_max] = -1
		while np.min(indices,axis=None)<indices_max:
			indices=np.minimum(indices_max,1+indices)
			indices_exp = np.expand_dims(indices,axis=0)
			np.put_along_axis(self.coef, indices_exp,coef_end,axis=0)
			np.put_along_axis(self.index,indices_exp,index_end,axis=0)

		self.coef  = np.moveaxis(self.coef,0,-1)
		self.index = np.moveaxis(self.index,0,-1)
		self.coef  = self.coef.reshape( self.shape+(size_ad_new,))
		self.index = self.index.reshape(self.shape+(size_ad_new,))

		self.index[self.index==-1]=0 # Corresponding coefficient is zero anyway.

			


# -------- End of class spAD -------

# -------- Factory method -----

def identity(shape=None,constant=None,shift=0):
	shape,constant = misc._set_shape_constant(shape,constant)
	shape2 = shape+(1,)
	return spAD(constant,np.full(shape2,1.),np.arange(shift,shift+np.prod(shape,dtype=int)).reshape(shape2))

def register(inputs,iterables=None,shift=0,ident=identity):
	if iterables is None:
		iterables = (tuple,)
	def reg(a):
		nonlocal shift
		a,to_ad = misc.ready_ad(a)
		if to_ad:
			result = ident(constant=a,shift=shift)
			shift += result.size
			return result
		else:
			return a
	return misc.map_iterables(reg,inputs,iterables)

