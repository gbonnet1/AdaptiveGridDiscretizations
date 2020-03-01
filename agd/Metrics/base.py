import numpy as np
from .. import AutomaticDifferentiation as ad
from .. import LinearParallel as lp
from .. import FiniteDifferences as fd
from .. import Interpolation

class Base(object):
	"""
	Base class for a metric
	"""

	def norm(self,v):
		"""
		Norm defiend by the metric. 
		Expected to be 1-homogeneous w.r.t. v
		"""
		raise ValueError("""Error : norm must be specialized in subclass""")

	def gradient(self,v):
		"""
		Gradient of the norm defined by the metric.
		"""
		if ad.is_ad(v) or ad.is_ad(self,iterables=(Base,)):
			v_dis = ad.disassociate(v,shape_bound=v.shape[1:])
			grad_dis = self.disassociate().gradient(v_dis)
			return ad.associate(grad_dis)

		v_ad = ad.Dense.identity(constant=v,shape_free=(len(v),))
		return np.moveaxis(self.norm(v_ad).coef,-1,0)
	
	def dual(self):
		raise ValueError("dual is not implemented for this norm")

	@property
	def vdim(self):
		"""
		Ambient vector space dimension
		"""
		raise ValueError("ndim is not implemented for this norm")

	@property
	def shape(self):
		"""
		Dimensions of the underlying domain.
		Expected to be the empty tuple, or a tuple of length vdim.
		"""
		raise ValueError("Shape not implemented for this norm")
	
	def disassociate(self):
		def dis(x):
			if isinstance(x,np.ndarray) and x.shape[-self.vdim:]==self.shape:
				return ad.disassociate(x,shape_bound=self.shape)
			return x
		return self.from_generator(dis(x) for x in self)
# ---- Well posedness related methods -----
	def is_definite(self):
		"""
		Wether norm(u)=0 implies u=0. 
		"""
		raise ValueError("is_definite is not implemented for this norm")

	def anisotropy(self):
		"""
		Sharp upper bound on norm(u)/norm(v), 
		for any unit vectors u and v.
		"""
		raise ValueError("anisotropy is not implemented for this norm")

	def anisotropy_bound(self):
		"""
		Upper bound on norm(u)/norm(v), 
		for any unit vectors u and v.
		"""
		return self.anisotropy()
# ---- Causality and acuteness related methods ----

	def cos_asym(self,u,v):
		"""
		Generalized cosine defined by the metric, 
		asymmetric variant defined as 
		<grad F(u), v> / F(v)
		"""
		u,v=(ad.array(e) for e in (u,v))
		return lp.dot_VV(self.gradient(u),v)/self.norm(v)

	def cos(self,u,v):
		"""
		Generalized cosine defined by the metric.
		"""
		u,v=(ad.array(e) for e in (u,v))
		gu,gv=self.gradient(u),self.gradient(v)
		guu,guv = lp.dot_VV(gu,u),lp.dot_VV(gu,v)
		gvu,gvv = lp.dot_VV(gv,u),lp.dot_VV(gv,v)
		return np.minimum(guv/gvv,gvu/guu)

	def angle(self,u,v):
		c = ad.array(self.cos(u,v))
		mask=c < -1.
		c[mask]=0.
		result = ad.array(np.arccos(c))
		result[mask]=np.inf
		return result

# ---- Geometric transformations ----

	def inv_transform(self,a):
		"""
		Affine transformation of the norm. 
		The new unit ball is the inverse image of the previous one.
		"""
		raise ValueError("Affine transformation not implemented for this norm")

	def transform(self,a):
		"""
		Affine transformation of the norm.
		The new unit ball is the direct image of the previous one.
		"""
		return self.inv_transform(lp.inverse(a))

	def rotate(self,r):
		"""
		Rotation of the norm, by a given rotation matrix.
		The new unit ball is the direct image of the previous one.
		"""
		return self.transform(r)

	def rotate_by(self,*args,**kwargs):
		"""
		Rotation of the norm, based on rotation parameters : angle (and axis in 3D).
		"""
		return self.rotate(lp.rotation(*args,**kwargs))

# ---- Import and export ----

	def flatten(self):
		raise ValueError("Flattening not implemented for this norm")

	@classmethod
	def expand(cls,arr):
		raise ValueError("Expansion not implemented for this norm")

	def to_HFM(self):
		"""
		Formats a metric for the HFM library. 
		This may include flattening some symmetric matrices, 
		concatenating with vector fields, and moving the first axis last.
		"""
		return np.moveaxis(self.flatten(),0,-1)

	def model_HFM(self):
		raise ValueError("HFM name is not specified for this norm")

	@classmethod
	def from_HFM(cls,arr):
		return cls.expand(np.moveaxis(arr,-1,0))

	def __iter__(self):
		raise ValueError("__iter__ not implemented for this norm")
		
	@classmethod
	def from_generator(cls,gen):
		return cls(*gen)

# ---- Related with Lagrandian and Hamiltonian interpretation ----

	def norm2(self,v):
		"""
		Half squared norm.
		"""
		n = self.norm(v)
		return 0.5*n**2

	def gradient2(self,v):
		"""
		Gradient of the half squared norm.
		"""
		g = self.gradient(v)
		return lp.dot_VV(g,v)*g

	def set_interpolation(self,grid,**kwargs):
		"""
		Sets interpolation_data, required to specialize the norm 
		at a given position.
		Inputs:
			- grid (optional). Coordinate system (required on first call). 
			- kwargs. Passed to UniformGridInterpolation (includes order)
		"""
		assert self.vdim == len(grid)

		def make_interp(value):
			if not isinstance(value,np.ndarray): return value
			if value.shape[-self.vdim:]!=grid[0].shape: return value
			return Interpolation.UniformGridInterpolation(grid,value,**kwargs)

		self.interpolation_data = tuple(make_interp(value) for value in self)

	def at(self,x):
		"""
		Interpolates the metric to a given position, on a grid given beforehand.
		Inputs : 
			- x (optional). Place where interpolation is needed.
		"""
		return self.from_generator(
			field(x) if callable(field) else field
			for field in self.interpolation_data)
		# isinstance(field,Interpolation.UniformGridInterpolation)
	

#	def is_ad(self):
#		return ad.is_ad(self,iterables=(Base,))

#	def remove_ad(self):
#		return self.from_generator(ad.remove_ad(x) for x in self)
