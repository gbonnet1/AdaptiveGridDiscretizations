# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import scipy.sparse

from .. import AutomaticDifferentiation as ad
from .. import LinearParallel as lp
from .hamiltonian_base import HamiltonianBase,fixedpoint

class MetricHamiltonian(HamiltonianBase):

	def __init__(self,metric,inv_inner=None,**kwargs):
		"""
		Hamiltonian defined by an interpolated metric, which is dualized and interpolated.
		H(q,p) = (1/2) F^*_q(p)^2.
		- metric : dual defines the hamiltonian
		- kwargs : passed to metric.dual().set_interpolation 
		"""
		super(MetricHamiltonian,self).__init__(inv_inner,vdim=metric.vdim)
		self._dualmetric = metric.dual()
		self._dualmetric.set_interpolation(**kwargs)

	def _H(self,q,p): return self._dualmetric.at(q).norm2(p)
	
	def _DqH(self,q,p): 
		q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
		return self._dualmetric.at(q_ad).norm2(p).gradient()

	def _DpH(self,q,p): return self._dualmetric.at(q).gradient2(p)

class GenericHamiltonian(HamiltonianBase):

	def __init__(self,func, disassociate_ad=False, **kwargs):
		"""
		Hamiltonian defined by a function.
		- disassociate_ad (optional) : hide AD information when calling f
		- shape_free (optional) : shape of position and momentum variables
		"""
		super(GenericHamiltonian,self).__init__(**kwargs)
		self._func = func
		self.disassociate_ad = disassociate_ad

	def _H(self,q,p): return self._func(q,p)

	def _identity_ad(self,x,noad=None):
		x_ad = ad.Dense.identity(constant=x,shape_free=self.shape_free) 
		if self.disassociate_ad: 
			x_dis = ad.disassociate(x_ad,shape_free=self.shape_free)
			if noad is None: return x_dis
			else:return (x_dis,ad.disassociate(type(x_ad)(noad),shape_free=self.shape_free)) 
		else: 
			return x_ad if noad is None else (x_ad,noad)

	def _gradient_ad(self,x):
		if self.disassociate_ad: x=ad.associate(x)
		return x.gradient()

	def _DqH(self,q,p):
		q_ad,p = self._identity_ad(q,noad=p)
		return self._gradient_ad(self._H(q_ad,p))

	def _DpH(self,q,p):
		p_ad,q = self._identity_ad(p,noad=q)
		return self._gradient_ad(self._H(q,p_ad))

class SeparableHamiltonian(HamiltonianBase):

	def __init__(self,Hq,Hp,**kwargs):
		"""
		Separable Hamiltonian defined by a pair of functions.
		- Hq,Hp : two functions, defining the hamiltonian
		- shape_free (optional) : shape of position and momentum variables
		- inv_inner (optional): inverse inner product, used for gradient normalization
		"""
		super(SeparableHamiltonian,self).__init__(**kwargs,is_separable=True)
		self.Hq = Hq
		self.Hp = Hp

	def _H(self,q,p): return self.Hq(q) + self.Hp(p)

	def _DqH(self,q,_):
		q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
		return self.Hq(q_ad).gradient()

	def _DpH(self,_,p):
		p_ad = ad.Dense.identity(constant=p,shape_free=self.shape_free)
		return self.Hp(p_ad).gradient()

class QuadraticHamiltonian(HamiltonianBase):

	def __init__(self,Mq,Mp,**kwargs):
		"""
		Quadratic Hamiltonian, defined by a pair of linear operators.
		(Expected to be symmetric, semi-definite.)
		"""
		super(QuadraticHamiltonian,self).__init__(**kwargs,is_separable=True)
		self.Mq = Mq
		self.Mp = Mp
	
	def flat(self,x): 
		return x.reshape((self.size_free,*x.shape[self.ndim_free:])) 

	def _H(self,q,p):
		q,p = map(self.flat,(q,p))
		return 0.5*(lp.dot_VV(q,ad.apply_linear_mapping(self.Mq,q)) 
			+ lp.dot_VV(p,ad.apply_linear_mapping(self.Mp,p)) )  

	def _DqH(self,q,_): return ad.apply_linear_mapping(self.Mq,self.flat(q))
	def _DpH(self,_,p): return ad.apply_linear_mapping(self.Mp,self.flat(p))

	@classmethod
	def spmat(cls,f,x,simplify_ad=None):
		"""
		Returns the sparse matrix associated to the hessian of f at x.
		Output of f is summed, if non-scalar.
		- simplify_ad (optional): wether to simplify the ad information 
		   before generating the sparse matrix
		- inv_inner : inverse inner product, for gradient normalization
		"""
		x_ad = ad.Sparse2.identity(constant=x)
		f_ad = f(x_ad) 
		if simplify_ad is None: simplify_ad = f_ad.ndim > 0
		for i in range(simplify_ad): f_ad.simplify_ad()
		f_ad = f_ad.sum()
		return ad.misc.tocsr(f_ad.triplets()) #scipy.sparse.coo_matrix(f_ad.triplets()).tocsc()

	def set_spmat(self,x,**kwargs):
		self.shape_free = x.shape
		if callable(self.Mq): self.Mq = self.spmat(self.Mq,x,**kwargs)
		if callable(self.Mp): self.Mp = self.spmat(self.Mp,x,**kwargs)
		if callable(self.inv_inner): self.inv_inner = self.spmat(self.inv_inner,x,**kwargs) 




