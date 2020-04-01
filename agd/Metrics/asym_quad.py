# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .base import Base
from .riemann import Riemann
from .rander import Rander
from . import misc
from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from ..FiniteDifferences import common_field

class AsymQuad(Base):

	def __init__(self,m,w):
		m,w = (ad.asarray(e) for e in (m,w))
		self.m,self.w =common_field((m,w),(2,1))

	def norm(self,v):
		v,m,w = common_field((ad.asarray(v),self.m,self.w),(1,2,1))
		return np.sqrt(lp.dot_VAV(v,m,v) + np.maximum(lp.dot_VV(w,v),0.)**2)

	def dual(self):
		M = lp.inverse(self.m+lp.outer_self(self.w))
		wInv = lp.solve_AV(self.m,self.w)
		W = -wInv/np.sqrt(1.+lp.dot_VV(self.w,wInv))
		return AsymQuad(M,W)

	@property
	def vdim(self): return len(self.m)

	@property
	def shape(self): return self.m.shape[2:]	

	def is_definite(self):
		return Riemann(self.m).is_definite()

	def anisotropy(self):
		eMax = Riemann(self.m+lp.outer_self(self.w)).eigvals().max(axis=0)
		eMin = Riemann(self.m).eigvals().min(axis=0)
		return np.sqrt(eMax/eMin)

	def cost_bound(self):
		return Riemann(self.m).cost_bound() + ad.Optimization.norm(w,ord=2,axis=0)

	def inv_transform(self,a):
		return AsymQuad(Riemann(self.m).inv_transform(a),lp.dot_VA(self.w,a))

	def flatten(self):
		return Rander(self.m,self.w).flatten()

	@classmethod
	def expand(cls,arr):
		rd = Rander.expand(arr)
		return cls(rd.m,rd.w)

	def model_HFM(self):
		return "AsymmetricQuadratic"+str(self.vdim)


	@classmethod
	def needle(cls,u,cost_parallel,cost_orthogonal):
		"""
		Defines a one-sided needle like metric, 
		if cost-parallel >= cost_orthogonal.

		Undefined behavior if if cost-parallel < cost_orthogonal.
		"""
		riem,_u = Riemann.needle(u,cost_parallel,cost_orthogonal,ret_u=True)
		return cls(riem.m,-(cost_orthogonal-cost_parallel)*_u)

	@classmethod
	def from_cast(cls,metric): 
		if isinstance(metric,cls):	return metric
		riemann = Riemann.from_cast(metric)
		return cls(riemann.m,(0.,)*riemann.vdim)

	def __iter__(self):
		yield self.m
		yield self.w