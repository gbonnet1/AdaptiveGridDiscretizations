import numpy as np
import copy
from . import misc
from . import Sparse

class reverseAD(object):
	"""
	A class for reverse first order automatic differentiation
	"""

	def __init__(self):
		self.deepcopy_states = False
		self._size_ad = 0
		self._size_rev = 0
		self._states = []
		self._shapes_ad = tuple()

	@property
	def size_ad(self): return self._size_ad
	@property
	def size_rev(self): return self._size_rev

	# Variable creation
	def identity(self,*args,**kwargs):
		"""Creates and register a new AD variable"""
		result = Sparse.identity(*args,**kwargs,shift=self.size_ad)
		self._shapes_ad += ([self.size_ad,result.shape],)
		self._size_ad += result.size
		return result

	def _identity_rev(self,*args,**kwargs):
		"""Creates and register an AD variable with negative indices, 
		used as placeholders in reverse AD"""
		result = Sparse.identity(*args,**kwargs,shift=self.size_rev)
		self._size_rev += result.size
		result.index = -result.index-1
		return result

	def _index_rev(self,index):
		"""Turns the negative placeholder indices into positive ones, 
		for sparse matrix creation."""
		index=index.copy()
		pos = index<0
		index[pos] = -index[pos]-1+self.size_ad
		return index

	# Applying a function
	def apply(self,func,*args,**kwargs):
		"""
		Applies a function on the given args, saving adequate data
		for reverse AD.
		"""
		_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse.spAD)
		if len(corresp)==0: return f(args,kwargs)
		_output = func(*_args,**_kwargs)
		output,shapes = misc._apply_output_helper(self,_output)
		self._states.append((shapes,func,
			copy.deepcopy(args) if self.deepcopy_states else args,
			copy.deepcopy(kwargs) if self.deepcopy_states else kwargs))
		return output

	def apply_linear_mapping(self,matrix,rhs,niter=1):
		return self.apply(linear_mapping_with_adjoint(matrix,niter=niter),rhs)
	def apply_linear_inverse(self,solver,matrix,rhs,niter=1):
		return self.apply(linear_inverse_with_adjoint(solver,matrix,niter=niter),rhs)
	def simplify(self,rhs):
		return self.apply(identity_with_adjoint,rhs)

	def iterate(self,func,var,*args,**kwargs):
		"""
		Input: function, variable to be updated, niter, nrec, optional args
		Iterates a function, saving adequate data for reverse AD. 
		If nrec>0, a recursive strategy is used to limit the amount of data saved.
		"""
		niter = kwargs.pop('niter')
		nrec = 0 if niter<=1 else kwargs.pop('nrec',0)
		assert nrec>=0
		if nrec==0:
			for i in range(niter):
				var = self.apply(func,
					var if self.deepcopy_states else copy.deepcopy(var),
					*args,**kwargs)
			return var
		else:
			assert False #TODO
		"""
			def recursive_iterate():
				other = reverseAD()
				return other.iterate(func,
			niter_top = int(np.ceil(niter**(1./(1+nrec))))
			for rec_iter in (niter//niter_top,)*niter_top + (niter%niter_top,)
				
				var = self.apply(recursive_iterate,var,*args,**kwargs,niter=rec_iter,nrec=nrec-1)

		for 
		"""


	# Adjoint evaluation pass
	def gradient(self,a):
		coef = Sparse.spAD(a.value,a.coef,self._index_rev(a.index)).to_dense().coef
		size_total = self.size_ad+self.size_rev
		if coef.size<size_total:  coef = misc._pad_last(coef,size_total)
		for outputshapes,func,args,kwargs in reversed(self._states):
			co_output = misc._to_shapes(coef[self.size_ad:],outputshapes)
			_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse.spAD)
			co_args = func(*_args,**_kwargs,co_output=co_output)
			for a_value,a_adjoint in co_args:
				for a_sparse,a_value2 in corresp:
					if a_value is a_value2:
						val,(row,col) = a_sparse.triplets()
						coef_contrib = misc.spapply(
							(val,(self._index_rev(col),row)),
							a_adjoint)
						# Possible improvement : shift by np.min(self._index_rev(col)) to avoid adding zeros
						coef[:coef_contrib.shape[0]] += coef_contrib
						break
		return coef[:self.size_ad]

	def to_inputshapes(self,a):
		return misc._to_shapes(a,self._shapes_ad)
# End of class reverseAD

def empty():
	return reverseAD()

# Elementary operators with adjoints

def linear_inverse_with_adjoint(solver,matrix,niter=1):
	from . import apply_linear_inverse
	def operator(x):	return apply_linear_inverse(solver,matrix,  x,niter=niter)
	def adjoint(x): 	return apply_linear_inverse(solver,matrix.T,x,niter=niter)
	def method(u,co_output=None,co_output2=None):
		if not(co_output2 is None):		return [(u,adjoint(co_output),adjoint(co_output2))]
		elif not(co_output is None):	return [(u,adjoint(co_output))]
		else:	return operator(u)
	return method

def linear_mapping_with_adjoint(matrix,niter=1):
	from . import apply_linear_mapping
	def operator(x):	return apply_linear_mapping(matrix,  x,niter=niter)
	def adjoint(x): 	return apply_linear_mapping(matrix.T,x,niter=niter)
	def method(u,co_output=None,co_output2=None):
		if not(co_output2 is None):		return [(u,adjoint(co_output),adjoint(co_output2))]
		elif not(co_output is None):	return [(u,adjoint(co_output))]
		else:	return operator(u)
	return method

def identity_with_adjoint(u,co_output=None,co_output2=None):
	if not(co_output2 is None):		return [(u,co_output,co_output2)]
	elif not(co_output is None):	return [(u,co_output)]
	else:	return u