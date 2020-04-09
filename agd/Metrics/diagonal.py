# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .base import Base
from .isotropic import Isotropic
from .. import AutomaticDifferentiation as ad

class Diagonal(Base):
	"""
	A diagonal metric, defined through axis dependent costs.
	"""

	def __init__(self,costs):
		self.costs = ad.asarray(costs)

	@classmethod
	def from_speed(cls,speeds): return cls(1./speeds)
	def dual(self): return self.from_speed(self.costs)
	def with_costs(self,costs): return Diagonal(self.costs*costs)

	def norm(self,v):
		costs,v = fd.common_field((self.costs,v),depths=(1,1))
		return ad.Optimization.norm(cost*v,ord=2,axis=0)

	def is_definite(self): return np.all(self.costs>0.,axis=0)
	def anisotropy(self): return np.max(self.costs,axis=0)/np.min(self.costs,axis=0)
	def cost_bound(self): return np.max(self.costs,axis=0)

	@property
	def vdim(self): return len(self.costs)
	@property
	def shape(self): return self.costs.shape[1:]

	def flatten(self):      return self.costs
	@classmethod
	def expand(cls,arr):    return cls(arr)

	def model_HFM(self):
		return "Diagonal"+str(self.vdim)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		iso = Isotropic.from_cast(metric)
		shape = (iso.vdim,) + iso.shape
		return cls(ad.broadcast_to(iso.cost,shape))

	def __iter__(self):
		yield self.costs