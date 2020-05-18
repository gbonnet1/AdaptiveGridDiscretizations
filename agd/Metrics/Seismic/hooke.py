# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import itertools
import copy

from ... import AutomaticDifferentiation as ad
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from ... import Selling
from ...FiniteDifferences import common_field
from .. import misc
from ..riemann import Riemann
from .implicit_base import ImplicitBase
from .thomsen_data import HexagonalFromTEM

class Hooke(ImplicitBase):
	"""
	A norm defined by a Hooke tensor. 
	Often encountered in seismic traveltime tomography.
	"""
	def __init__(self,hooke,*args,**kwargs):
		super(Hooke,self).__init__(*args,**kwargs)
		self.hooke = hooke
		self._to_common_field()

	def is_definite(self):
		return Riemann(self.hooke).is_definite()

	@staticmethod
	def _vdim(hdim):
		"""Vector dimension from Hooke tensor size"""
		vdim = int(np.sqrt(2*hdim))
		if Hooke._hdim(vdim)!=hdim:
			raise ValueError("Incorrect hooke tensor")
		return vdim

	@staticmethod
	def _hdim(vdim):
		"""Hooke tensor size from vector dimension"""
		return (vdim*(vdim+1))//2

	@property
	def vdim(self):
		return self._vdim(len(self.hooke))

	@property
	def shape(self): return self.hooke.shape[2:]
	
	def model_HFM(self):
		d = self.vdim
		suffix = "" if self.inverse_transformation is None else "Topographic"
		return f"Seismic{suffix}{d}"

	def flatten(self):
		hooke = misc.flatten_symmetric_matrix(self.hooke)
		if self.inverse_transformation is None: 
			return hooke
		else: 
			inv_trans= self.inverse_transformation.reshape((self.vdim**2,)+self.shape)
			return np.concatenate((hooke,inv_trans),axis=0)

	@classmethod
	def expand(cls,arr):
		return cls(misc.expand_symmetric_matrix(arr))

	def __iter__(self):
		yield self.hooke
		for x in super(Hooke,self).__iter__():
			yield x

	def with_cost(self,cost):
		other = copy.copy(self)
		other.hooke = self.hooke * cost**2
		return other

	def _to_common_field(self,*args,**kwargs):
		self.hooke,self.inverse_transformation = fd.common_field(
			(self.hooke,self.inverse_transformation),(2,2),*args,**kwargs)

	def _dual_params(self,*args,**kwargs):
		return fd.common_field((self.hooke,),(2,),*args,**kwargs)

	def _dual_level(self,v,params=None,relax=0.):
		if params is None: params = self._dual_params(v.shape[1:])

		# Contract the hooke tensor and covector
		hooke, = params
		Voigt,Voigti = self._Voigt,self._Voigti
		d = self.vdim
		m = ad.asarray([[
			sum(v[j]*v[l] * hooke[Voigt[i,j],Voigt[k,l]] 
				for j in range(d) for l in range(d))
			for i in range(d)] for k in range(d)])

		# Evaluate det
		s = np.exp(-relax)
		xp = ad.cupy_generic.get_array_module(m)
		ident = fd.as_field(xp.eye(d,dtype=m.dtype),m.shape[2:],depth=2)
		return 1.-s -lp.det(ident - m*s) 

	def extract_xz(self):
		"""
		Extract a two dimensional Hooke tensor from a three dimensional one, 
		corresponding to a slice through the X and Z axes.
		"""
		assert self.vdim==3
		h=self.hooke
		return Hooke(ad.asarray([ 
			[h[0,0], h[0,2], h[0,4] ],
			[h[2,0], h[2,2], h[2,4] ],
			[h[4,0], h[4,2], h[4,4] ]
			]))

	@classmethod
	def from_VTI_2(cls,Vp,Vs,eps,delta):
		"""
		X,Z slice of a Vertical Transverse Isotropic medium
		based on Thomsen parameters
		"""
		c33=Vp**2
		c44=Vs**2
		c11=c33*(1+2*eps)
		c13=-c44+np.sqrt( (c33-c44)**2+2*delta*c33*(c33-c44) )
		zero = np.zeros_like(Vs)
		return cls(ad.asarray( [ [c11,c13,zero], [c13,c33,zero], [zero,zero,c44] ] ))

	@classmethod
	def from_Ellipse(cls,m):
		"""
		Rank deficient Hooke tensor,
		equivalent, for pressure waves, to the Riemannian metric defined by m ** -2.
		Shear waves are infinitely slow.
		"""
		assert(len(m)==2)
		a,b,c=m[0,0],m[1,1],m[0,1]
		return Hooke(ad.asarray( [ [a*a, a*b,a*c], [a*b, b*b, b*c], [a*c, b*c, c*c] ] ))

	@classmethod
	def from_cast(cls,metric): 
		if isinstance(metric,cls):	return metric
		riemann = Riemann.from_cast(metric)
		
		m = riemann.dual().m
		assert not ad.is_ad(m)
		from scipy.linalg import sqrtm
		return cls.from_Ellipse(sqrtm(m))

	def _iter_implicit(self):
		yield self.hooke

	@property	
	def _Voigt(self):
		"""Direct Voigt indices"""
		if self.vdim==2:   return np.array([[0,2],[2,1]])
		elif self.vdim==3: return np.array([[0,5,4],[5,1,3],[4,3,2]])
		else: raise ValueError("Unsupported dimension")
	@property
	def _Voigti(self):
		"""Inverse Voigt indices"""
		if self.vdim==2:   return np.array([[0,0],[1,1],[0,1]])
		elif self.vdim==3: return np.array([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]])
		else: raise ValueError("Unsupported dimension")


	def rotate(self,r):
		other = copy.copy(self)
		hooke,r = common_field((self.hooke,r),(2,2))
		Voigt,Voigti = self._Voigt,self._Voigti
		other.hooke = ad.array([ [ [
			hooke[Voigt[i,j],Voigt[k,l]]*r[ii,i]*r[jj,j]*r[kk,k]*r[ll,l]
			for (i,j,k,l) in itertools.product(range(self.vdim),repeat=4)]
			for (ii,jj) in Voigti] 
			for (kk,ll) in Voigti]
			).sum(axis=2)
		return other

	@staticmethod
	def _Mandel_factors(vdim,shape=tuple(),a=np.sqrt(2)):
		def f(k):	return 1. if k<vdim else a
		hdim = Hooke._hdim(vdim)
		factors = ad.array([[f(i)*f(j) for i in range(hdim)] for j in range(hdim)])
		return fd.as_field(factors,shape,conditional=False)

	def to_Mandel(self,a=np.sqrt(2)):
		"""Introduces the sqrt(2) and 2 factors involved in Mandel's notation"""
		return self.hooke*self._Mandel_factors(self.vdim,self.shape)

	@classmethod
	def from_Mandel(cls,mandel,a=np.sqrt(2)):
		"""Removes the sqrt(2) and 2 factors involved in Mandel's notation"""
		vdim = cls._vdim(len(mandel))
		return Hooke(mandel/cls._Mandel_factors(vdim,mandel.shape[2:],a))

	@classmethod
	def from_orthorombic(cls,a,b,c,d,e,f,g,h,i):
		z=0.*a
		return cls(ad.array([
		[a,b,c,z,z,z],
		[b,d,e,z,z,z],
		[c,e,f,z,z,z],
		[z,z,z,g,z,z],
		[z,z,z,z,h,z],
		[z,z,z,z,z,i]
		]))

	@classmethod
	def from_tetragonal(cls,a,b,c,d,e,f):
		return cls.from_orthorombic(a,b,c,a,c,d,e,e,f)

	@classmethod
	def from_hexagonal(cls,a,b,c,d,e):
		return cls.from_tetragonal(a,b,c,d,e,(a-b)/2)

	@classmethod 
	def from_Thomsen(cls,tem):
		"""Hooke tensor (m/s)^2 and density (g/cm^3)"""
		hex,ρ = HexagonalFromTEM(tem)
		return cls.from_hexagonal(*hex),ρ

	def is_TTI(self,tol=None):
		"""
		Determine if the metric is in a TTI form.
		"""
		# Original code by F. Desquilbet, 2020
		if tol is None: # small value (acts as zero for floats)
			tol = max(1e-9, MA(hooke)*1e-12)

		def small(arr): return np.max(np.abs(arr))<tol
		is_Sym = small(hooke-lp.transpose(hooke)) # symmetrical

		if metric.vdim==2:
			return is_Sym and small(hooke[2,0]) and small(hooke[2,1])
		if metric.vdim==3:
			return (is_Sym 
				and small((hooke[0,0]-hooke[0,1])/2-hooke[5,5])
				and all(small(hooke[i,j]-hooke[k,l]) 
					for ((i,j),(k,l)) in [((0,0),(1,1)), ((2,0),(2,1)), ((3,3),(4,4))])
				and all(small(hooke[i,j]) 
					for (i,j) in [(3,0),(4,0),(5,0),(3,1),(4,1),(5,1),
					(3,2),(4,2),(5,2),(4,3),(5,3),(5,4)]) ) 

	def _Voigt_m2v(self,m,sym=True):
		"""
		Turns a symmetric matrix into a vector, based on Voigt convention.
		- sym : True -> use the upper triangular part of m. 
		        Fals -> symmetrize the matrix m (adding its transpose).
		"""
		assert(self.inverse_transformation is None)
		m=ad.asarray(m)
		vdim = self.vdim
		assert(m.shape[:2]==(vdim,vdim))
		if vdim==1:
			return m[0]
		elif vdim==2:
			if sym: return ad.array((m[0,0],m[1,1],2*m[0,1]))
			else:   return ad.array((m[0,0],m[1,1],m[0,1]+m[1,0]))
		elif vdim==3:
			if sym: return ad.array((m[0,0],m[1,1],m[2,2],
				2*m[1,2],2*m[0,2],2*m[0,1]))
			else:   return ad.array((m1[0,0],m1[1,1],m[2,2],
				m[1,2]+m[2,1],m[0,2]+m[2,0],m[0,1]+m[1,0]))
		else:
			raise ValueError("Unsupported dimension")


	def dot_A(self,m,sym=True):
		"""
		Dot product associated with a Hooke tensor, which turns a strain tensor epsilon
		into a stress tensor sigma.
		- m : the strain tensor.
		"""
		
		v,hooke = fd.common_field((self._Voigt_m2v(m,sym),self.hooke),(1,2))
		w = lp.dot_AV(hooke,v)
		return ad.array( ((w[0],w[2]),(w[2],w[1])) )

	def dot_AA(self,m1,m2=None,sym=True):
		"""
		Inner product associated with a Hooke tensor, on the space of symmetric matrices.
		- m1 : first symmetric matrix
		- m2 : second symmetric matrix. Defaults to m1.
		"""
		if m2 is None: 
			v1,hooke = fd.common_field((self._Voigt_m2v(m1,sym),self.hooke),(1,2))
			v2=v1
		else: 
			v1,v2,hooke = fd.common_field(
				(self._Voigt_m2v(m1,sym),self._Voigt_m2v(m2,sym),self.hooke),(1,1,2))
		return lp.dot_VV(v1,lp.dot_AV(hooke,v2))

	def Selling(self):
		"""
		Returns a decomposition of the hooke tensor in the mathematical form
		hooke = sum_i rho_i m_i x m_i,
		where rho_i is a non-negative coefficient, m_i is symmetric and has integer 
		entries, and sum_i rho_i is maximal. 
		"""
		assert(self.vdim<=2)
		assert(self.inverse_transformation is None)
		coefs,offsets = Selling.Decomposition(self.hooke)
		if self.vdim==1: 
			moffsets = np.expand_dims(offsets,axis=0)
		elif self.vdim==2:
			moffsets = ad.array(((offsets[0],offsets[2]),(offsets[2],offsets[1])))
		else :
			raise ValueError("Unsupported dimension")
		return coefs,moffsets

	def apply_transform(self):
		"""
		Applies the transformation, if any stored, to the hooke tensor. 
		CAUTION : this feature is required for some applications to elasticity, 
		but is incompatible with the eikonal equation solver.
		"""
		r = self.inverse_transformation
		if r is None: return self
		other = copy.copy(self)
		other.inverse_transformation = None
		return other.rotate(lp.transpose(r)) 

	@classmethod
	def from_Lame(cls,Lambda,Mu,vdim=2):
		"""
		Constructs a Hooke tensor from the Lame coefficients, in dimension 2 or 3.
		"""
		assert(not (ad.is_ad(Lambda) or ad.is_ad(Mu)) )
		hdim = cls._hdim(vdim)
		hooke = np.zeros_like(Lambda,shape=(hdim,hdim))
		hooke[:vdim,:vdim] += Lambda
		for i in range(hdim): 
			hooke[i,i] += Mu*(1.+(i<vdim))
		return cls(hooke)



#Hooke tensor (m/s)^2 and density (g/cm^3)
Hooke.mica = Hooke.from_hexagonal(178.,42.4,14.5,54.9,12.2), 2.79
Hooke.stishovite = Hooke.from_tetragonal(453,211,203,776,252,302), 4.29
Hooke.olivine = Hooke.from_orthorombic(323.7,66.4,71.6,197.6,75.6,235.1,64.6,78.7,79.0), 3.311



