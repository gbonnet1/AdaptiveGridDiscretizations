# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import copy

from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ... import FiniteDifferences as fd
from .. import misc
from ..riemann import Riemann
from .implicit_base import ImplicitBase
from .thomsen_data import HexagonalFromTEM


class TTI(ImplicitBase):
	"""
	A family of reduced models, known as Tilted Transversally Anisotropic,
	and arising in seismic tomography.

	The dual unit ball is defined by an equation of the form
	l(X^2+Y^2,Z^2) + q(X^2+Y^2,Z^2) = 1,
	where l is linear and q is quadratic, where X,Y,Z are the coefficients of the input 
	vector, usually altered by a linear transformation.
	In two dimensions, ignore the Z^2 term.
	"""

	def __init__(self,linear,quadratic,vdim=None,**kwargs):
		super(TTI,self).__init__(**kwargs)
		self.linear = ad.asarray(linear)
		self.quadratic = ad.asarray(quadratic)
		assert len(self.linear) == 2
		assert self.quadratic.shape[:2] == (2,2)
		self._to_common_field()
		
		if vdim is None:
			if self.inverse_transformation is not None: vdim=len(self.inverse_transformation)
			elif self.linear.ndim>1: vdim = self.linear.ndim-1
			else: raise ValueError("Unspecified dimension")
		self._vdim=vdim

	@property
	def vdim(self): return self._vdim
	
	@property
	def shape(self): return self.linear.shape[1:]

	def _dual_level(self,v,params=None,relax=0.):
		l,q = self._dual_params(v.shape[1:]) if params is None else params
		v2 = v**2
		return lp.dot_VV(l,v2) + np.exp(-relax)*lp.dot_VAV(v2,q,v2) - 1.
	
	def cost_bound(self):
		# Ignoring the quadratic term for now.
		return self.Riemann_approx().cost_bound()

	def _dual_params(self,shape=None):
		return fd.common_field((self.linear,self.quadratic),depths=(1,2),shape=shape)

	def __iter__(self):
		yield self.linear
		yield self.quadratic
		for x in super(TTI,self).__iter__(): yield x

	def _to_common_field(self,shape=None):
		self.linear,self.quadratic,self.inverse_transformation = fd.common_field(
			(self.linear,self.quadratic,self.inverse_transformation),
			depths=(1,2,2),shape=shape)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		else: raise ValueError("No casting operations supported towards the TTI model")
		# Even cast from Riemann is non-canonical

	def model_HFM(self):
		return f"TTI{self.vdim}"

	def Extract_XZ(self):
		if len(self.shape)==3: raise ValueError("Three dimensional field")
		if self.inverse_transformation is not None:
			raise ValueError("Cannot extract XZ slice from tilted norms")
		other = copy.copy(self)
		other._vdim = 2
		return other

	def flatten(self,transposed_transformation=False):
		linear = self.linear
		quad = 2.*self.quadratic # Note the factor 2, used in HFM

		if self.inverse_transformation is None: 
			xp = ad.cupy_generic.get_array_module(linear)
			trans = fd.as_field(xp.eye(self.vdim,dtype=linear.dtype),self.shape,depth=2) 
		else: trans = self.inverse_transformation
		if transposed_transformation: trans = lp.transpose(lp.inverse(trans))

		return np.concatenate(
			(self.linear,misc.flatten_symmetric_matrix(quad),
				trans.reshape((self.vdim**2,)+self.shape)),
			axis=0)

	@classmethod
	def expand(cls,arr):
		vdim = np.sqrt(len(arr)-(2+3))
		assert(vdim==int(vdim))
		vdim = int(vdim)
		shape = arr.shape[1:]

		linear = arr[0:2]
		quadratic = 0.5*misc.expand_symmetric_matrix(arr[2:5])
		inv_trans = arr[5:].reshape((vdim,vdim)+shape)
		return cls(linear,quadratic,vdim=vdim,inverse_transformation=inv_trans)

	@classmethod
	def from_hexagonal(cls,c11,_,c13,c33,c44):
		linear = [c11+c44,c33+c44]
		mixed = 0.5*(c13**2-c11*c33)+c13*c44
		quadratic = [[-c11*c44,mixed],[mixed,-c33*c44]]
		return cls(linear,quadratic)

	@classmethod
	def from_thomsen(cls,tem):
		hex,ρ = HexagonalFromTEM(tem)
		return cls.from_tetragonal(*hex),ρ

"""
	@classmethod
	def from_Thomsen(cls,Thomsen, vdim=3, V_squared=False):
		if V_squared: Vp2,Vs2,eps,delta
		else: Vp,Vs,eps,delta = Thomsen; Vp2,Vs2=Vp**2,Vs**2

		a = -(1+2*eps)*Vp2*Vs2
		b = -Vp2*Vs2
		c = -(1+2*eps)*Vp2**2-Vs2**2+(Vp2-Vs2)*(Vp2*(1+2*delta)-Vs2)
		d = Vs2+(1+2*eps)*Vp2
		e = Vp2+Vs2

		return cls([d,e], [[a,c],[c,b]], vdim=vdim)
"""
"""
	def to_Thomsen(self,V_squared=False,safe=True):
		if safe and self.inverse_transformation is not None:
			raise ValueError("Thomsen parameters loose track of tilt")

		((a,c),(_,b)) = self.quadratic
		d,e = self.linear

		Vp2 = (e+np.sqrt(e**2+4*b))/2
		Vs2 = (e-np.sqrt(e**2+4*b))/2
		eps = -0.5*(1+a/(Vp2*Vs2))
		delta = 1/2*(-1+1/Vp2*(Vs2+(c+Vp2**2*(1+2*eps)+Vs2**2)/(Vp2-Vs2)))

		if safe:
			if self.inverse_transformation is not None:
				raise ValueError("Thomsen parameters loose track of tilt")
			other = self.from_Thomsen(Vp2,Vs2,eps,delta,V_squared=True)
			if not (np.allclose(self.linear,other.linear)
				and np.allclose(self.quadratic,other.quadratic)):
				raise ValueError("Thomsen parameters appear to loose information"
					"Use safe=False to ignore")

		if V_squared: return Vp2,Vs2,eps,delta
		else: return np.sqrt(Vp2),np.sqrt(Vs2),eps,delta
"""

"""
	@classmethod
	def from_Hooke(cls,metric):
		from .hooke import Hooke
		hooke = metric.hooke
		
		#assert(metric.is_reduced_VTI(metric)) #TODO
	
		if metric.vdim==2:

			Vp = np.sqrt(hooke[1,1])
			Vs = np.sqrt(hooke[2,2])
			eps = (hooke[0,0]-hooke[1,1])/(2*hooke[1,1])
			delt = ((hooke[0,1]+hooke[2,2])**2-(hooke[1,1]-hooke[2,2])**2)/(
					2*hooke[1,1]*(hooke[1,1]-hooke[2,2]))
			
			aa = -(1+2*eps)*Vp**2*Vs**2
			bb = -Vp**2*Vs**2
			cc = -(1+2*eps)*Vp**4-Vs**4+(Vp**2-Vs**2)*(Vp**2*(1+2*delt)-Vs**2)
			dd = Vs**2+(1+2*eps)*Vp**2
			ee = Vp**2+Vs**2

			linear = ad.array([dd,ee])
			quadratic = ad.array([[aa,cc],[cc,bb]])

		elif metric.vdim==3:
			
			Vp = np.sqrt(hooke[2,2])
			Vs = np.sqrt(hooke[3,3])
			eps = (hooke[0,0]-hooke[2,2])/(2*hooke[2,2])
			delt = ((hooke[0,2]+hooke[3,3])**2-(hooke[2,2]-hooke[3,3])**2)/(
					2*hooke[2,2]*(hooke[2,2]-hooke[3,3]))
			
			aa = -(1+2*eps)*Vp**2*Vs**2
			bb = -Vp**2*Vs**2
			cc = -(1+2*eps)*Vp**4-Vs**4+(Vp**2-Vs**2)*(Vp**2*(1+2*delt)-Vs**2)
			dd = Vs**2+(1+2*eps)*Vp**2
			ee = Vp**2+Vs**2

			linear = ad.array([dd,dd,ee])
			quadratic = ad.array([[aa,2*aa,cc],[2*aa,aa,cc],[cc,cc,bb]])

		else:
			raise ValueError("Unsupported dimension")

		return cls(linear,quadratic,vdim=metric.vdim,*super(Hooke,metric).__iter__())

"""
