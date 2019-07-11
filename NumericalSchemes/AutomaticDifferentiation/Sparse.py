import numpy as np
from . import misc

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
		obj.coef  = (np.full(shape2,0.) if coef is None 
			else misc._test_or_broadcast_ad(coef,shape,broadcast_ad) ) 
		obj.index = (np.full(shape2,0)  if index is None 
			else misc._test_or_broadcast_ad(index,shape,broadcast_ad) )
		return obj

#	def __array_finalize__(self,obj): pass

	def copy(self,order='C'):
		return spAD(self.value.copy(order=order),self.coef.copy(order=order),self.index.copy(order=order))

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
			return spAD(self.value+other.value, _concatenate(self.coef,other.coef), _concatenate(self.index,other.index))
		else:
			return spAD(self.value+other, self.coef, self.index, broadcast_ad=True)

	def __sub__(self,other):
		if isinstance(other,spAD):
			return spAD(self.value-other.value, _concatenate(self.coef,-other.coef), _concatenate(self.index,other.index))
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
	def __rtruediv__(self,other): 	return spAD(other/self.value,self.coef*_add_dim(-other/self.value**2),self.index)

	def __neg__(self):		return spAD(-self.value,-self.coef,self.index)

	# Math functions
	def __pow__(self,n): 	return spAD(self.value**n, _add_dim(n*self.value**(n-1))*self.coef,self.index)
	def sqrt(self):		 	return self**0.5
	def log(self):			return spAD(np.log(self.value), self.coef*_add_dim(1./self.value), self.index)
	def exp(self):			return spAD(np.exp(self.value), self.coef*_add_dim(np.exp(self.value)), self.index)
	def abs(self):			return spAD(np.abs(self.value), self.coef*_add_dim(np.sign(self.value)), self.index)

	# Trigonometry
	def sin(self):			return spAD(np.sin(self.value), self.coef*_add_dim(np.cos(self.value)), self.index)
	def cos(self):			return spAD(np.cos(self.value), self.coef*_add_dim(-np.sin(self.value)), self.index)

	#Indexing
	@property
	def value(self): return self.view(np.ndarray)
	@property
	def size_ad(self):  return self.coef.shape[-1]

	def __getitem__(self,key):
		return spAD(self.value[key], self.coef[key], self.index[key])

	def __setitem__(self,key,other):
		if isinstance(other,spAD):
			self.value[key] = other.value
			pad_size = max(self.coef.shape[-1],other.coef.shape[-1])
			if pad_size>self.coef.shape[-1]:
				self.coef = _pad_last(self.coef,pad_size)
				self.index = _pad_last(self.index,pad_size)
			self.coef[key] = _pad_last(other.coef,pad_size)
			self.index[key] = _pad_last(other.index,pad_size)
		else:
			self.value[key] = other
			self.coef[key]  = 0.
#			self.index[key] = 0

	def reshape(self,shape,order='C'):
		shape2 = (shape if isinstance(shape,tuple) else (shape,))+(self.size_ad,)
		return spAD(self.value.reshape(shape,order=order),self.coef.reshape(shape2,order=order), self.index.reshape(shape2,order=order))

	def flatten(self):	
		return self.reshape( (self.size,) )

	def broadcast_to(self,shape):
		shape2 = shape+(self.size_ad,)
		return spAD(np.broadcast_to(self.value,shape), np.broadcast_to(self.coef,shape2), np.broadcast_to(self.index,shape2))

	@property
	def T(self):	return self if self.ndim<2 else self.transpose()
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes) +(self.ndim,)
		return spAD(self.value.transpose(axes),self.coef.transpose(axes2),self.index.transpose(axes2))

	def triplets(self):
		coef = self.coef.flatten()
		row = np.broadcast_to(_add_dim(np.arange(self.size).reshape(self.shape)), self.index.shape).flatten()
		column = self.index.flatten()

		pos=coef!=0
		return (coef[pos],(row[pos],column[pos]))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		value = self.value.sum(axis,**kwargs)
		shape = value.shape +(self.size_ad * self.shape[axis],)
		coef = np.moveaxis(self.coef, axis,-1).reshape(shape)
		index = np.moveaxis(self.index, axis,-1).reshape(shape)
		out = spAD(value,coef,index)
		return out

#	def prod(self,axis=None,out=None,**kwargs):
#		if axis is None: return self.flatten().prod(axis=0,out=out,**kwargs)
#		result = reduce( 
#		cprod = np.cumprod(self.value,axis=axis)
#		cprod_rev = n

	def min(self,*args,**kwargs):
		return misc.min(self,*args,**kwargs)

	def max(self,*args,**kwargs):
		return misc.max(self,*args,**kwargs)

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
			if ufunc==np.sqrt: return self.sqrt()
			if ufunc==np.log: return self.log()
			if ufunc==np.exp: return self.exp()
			if ufunc==np.abs: return self.abs()

			# Trigonometry
			if ufunc==np.sin: return self.sin()
			if ufunc==np.cos: return self.cos()

			# Operators
			if ufunc==np.add: return self.add(*inputs,**kwargs)
			if ufunc==np.subtract: return self.subtract(*inputs,**kwargs)
			if ufunc==np.multiply: return self.multiply(*inputs,**kwargs)
			if ufunc==np.true_divide: return self.true_divide(*inputs,**kwargs)


		return NotImplemented


	# Numerical 
	def solve(self):
		import scipy.sparse; import scipy.sparse.linalg
		return - scipy.sparse.linalg.spsolve(
        scipy.sparse.coo_matrix(self.triplets()).tocsr(),
        np.array(self).flatten()).reshape(self.shape)

	def bound_ad(self):
		return 1+np.max(self.index,initial=-1)
	def to_dense(self,dense_size_ad=None):
		def mvax(arr): return np.moveaxis(arr,-1,0)
		from . import Dense
		coef = np.zeros(self.shape+(self.bound_ad() if dense_size_ad is None else dense_size_ad,))
		for c,i in zip(mvax(self.coef),mvax(self.index)):
			np.put_along_axis(coef,_add_dim(i),np.take_along_axis(coef,_add_dim(i),axis=-1)+c,axis=-1)
		return Dense.denseAD(self.value,coef)

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
		elems2 = tuple(spAD(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		return spAD( 
		np.stack(tuple(e.value for e in elems2), axis=axis), 
		np.stack(tuple(_pad_last(e.coef,size_ad)  for e in elems2),axis=axis),
		np.stack(tuple(_pad_last(e.index,size_ad) for e in elems2),axis=axis))

	# Memory optimization
	def simplify_ad(self):
		if len(self.shape)==0: # Workaround for scalar-like arrays
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

# -------- Some utility functions, for internal use -------

def _concatenate(a,b): 	return np.concatenate((a,b),axis=-1)
def _add_dim(a):		return np.expand_dims(a,axis=-1)	
def _pad_last(a,pad_total):
		return np.pad(a, pad_width=((0,0),)*(a.ndim-1)+((0,pad_total-a.shape[-1]),), mode='constant', constant_values=0)
def _tuple_first(a): return a[0] if isinstance(a,tuple) else a

# -------- Factory method -----

def identity(shape=None,constant=None,shift=0):
	shape,constant = misc._set_shape_constant(shape,constant)
	shape2 = shape+(1,)
	return spAD(constant,np.full(shape2,1.),np.arange(shift,shift+np.prod(shape)).reshape(shape2))
