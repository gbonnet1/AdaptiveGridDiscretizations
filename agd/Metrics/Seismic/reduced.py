# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ... import FiniteDifferences as fd
from .. import misc
from ..riemann import Riemann
from .implicit_base import ImplicitBase


class Reduced(ImplicitBase):
	"""
	A family of reduced models appearing in seismic tomography, 
	based on assuming that some of the coefficients of the Hooke tensor vanish.

	The dual ball is defined by an equation of the form 
	l(X^2,Y^2,Z^2) + q(X^2,Y^2,Z^2) + c X^2 Y^2 Z^2 = 1
	where l is linear, q is quadratic, and c is a cubic coefficient.
	X,Y,Z are the coefficients of the input vector, perhaps subject to a linear transformation.
	"""

	def __init__(self,linear,quadratic=None,cubic=None,*args,**kwargs):
		super(Reduced,self).__init__(*args,**kwargs)
		self.linear=ad.asarray(linear)
		self.quadratic=None if quadratic is None else ad.asarray(quadratic)
		self.cubic=None if cubic is None else ad.asarray(cubic)
		assert cubic is None or self.vdim==3
		self._to_common_field()

	@property
	def vdim(self):
		return len(self.linear)
	
	@property
	def shape(self):
		return self.linear.shape[1:]

	def _dual_level(self,v,params=None,relax=0.):
		if params is None: params = self._dual_params(v.shape[1:])
		s = np.exp(-relax)
		l,q,c = params
		v2 = v**2
		result = lp.dot_VV(l,v2) - 1.
		if q is not None:
			result += lp.dot_VAV(v2,q,v2)*s
		if c is not None:
			assert self.vdim==3
			result += v2.prod()*(c*s**2)
		return result

	def cost_bound(self):
		"""
		Upper bound on norm(u), for any unit vector u.
		"""
		cst = np.max(self.linear,axis=0) # Ignoring quadratic and cubic for now
		if self.inverse_transformation is not None: # Frobenius norm upper bounds operator norm 
			cst *= (self.inverse_transformation**2).sum(axis=(0,1))
		return np.sqrt(cst)

	def _dual_params(self,*args,**kwargs):
		return fd.common_field((self.linear,self.quadratic,self.cubic),(1,2,0),*args,**kwargs)

	def __iter__(self):
		yield self.linear
		yield self.quadratic
		yield self.cubic
		for x in super(Reduced,self).__iter__(): 
			yield x

	def _to_common_field(self,*args,**kwargs):
		self.linear,self.quadratic,self.cubic,self.inverse_transformation = fd.common_field(
			(self.linear,self.quadratic,self.cubic,self.inverse_transformation),
			(1,2,0,2),*args,**kwargs)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		riemann = Riemann.from_cast(metric)
		assert not ad.is_ad(riemann.m)
		e,a = np.linalg.eig(riemann.m)
		result = Reduced(e)
		result.inv_transform(a)
		return result

	def to_TTI3(self):
		"""
		Produces a three dimensional norm of TTI type (tilted transversally isotropic), 
		from two dimensional one.
		"""
		assert(self.inverse_transformation is None)
		assert(self.vdim==2)

		l = self.linear
		linear = ad.array((l[0],l[1],l[1]))
		if self.quadratic is None: return Reduced(linear)

		q=self.quadratic
		quadratic = ad.array( (
			(q[0,0],q[0,1],q[0,1]),
			(q[1,0],q[1,1],q[1,1]),
			(q[1,0],q[1,1],q[1,1]) ) )
		return Reduced(linear,quadratic,
			niter_sqp=self.niter_sqp,relax_sqp=self.relax_sqp)

	def is_TTI(self):
		"""Wether the norm is of tilted transversally isotropic type"""
		if self.vdim==2: return True
		assert(self.vdim==3)
		return ( 
			np.all(self.linear[1]==self.linear[2]) 
			and (self.quadratic is None or (
				np.all(self.quadratic[1,1]==self.quadratic[1,2])
				and np.all(self.quadratic[1,1]==self.quadratic[2,2])
				and np.all(self.quadratic[0,1]==self.quadratic[0,2]) ) )
			and (self.cubic is None or np.all(self.cubic==0.) ) 
			)

	def model_HFM(self):
		return f"TTI{self.vdim}"
	
	def flatten(self,transposed_transformation=False):
		assert(self.is_TTI()) # Only the TTI type is handle by HFM
		if self.quadratic is None: 
			quad = ad.cupy_support.zeros_like(self.linear, (2,2)+self.shape)
		else: quad = 2.*self.quadratic[0:2,0:2] # Note the factor 2, used in HFM

		xp = ad.cupy_generic.get_array_module(self.linear)
		if self.inverse_transformation is None: 
			trans = fd.as_field(xp.eye(self.vdim),self.shape,conditional=False) 
		else: trans = self.inverse_transformation
		if transposed_transformation: trans = lp.transpose(lp.inverse(trans))

		return np.concatenate(
			(self.linear[0:2],misc.flatten_symmetric_matrix(quad),
				trans.reshape((self.vdim**2,)+self.shape)),
			axis=0)

	@classmethod
	def expand(cls,arr):
		#Only TTI norms are supported
		vdim = np.sqrt(len(arr)-(2+3))
		shape = arr.shape[1:]
		assert(vdim==int(vdim))
		linear = arr[0:2]
		quadratic = 0.5*misc.expand_symmetric_matrix(arr[2:5])
		inv_trans = arr[5:].reshape((vdim,vdim)+shape)
		if vdim==2:
			return cls(linear,quadratic,inverse_transformation=inv_trans)
		else:
			assert(vdim==3)
			return cls(ad.array([linear[0],linear[1],linear[1]]),
				ad.array([[quadratic[0,0],quadratic[0,1],quadratic[0,1]],
					[quadratic[1,0],quadratic[1,1],quadratic[1,1]],
					[quadratic[1,0],quadratic[1,1],quadratic[1,1]]]),
				inverse_transformation=inv_trans)


	@classmethod
	def from_Hooke(cls,metric):
		"""
		Generate reduced algebraic form from full Hooke tensor.
		Warning : Hooke to Reduced conversion requires that some 
		coefficients of the Hooke tensor vanish, and may induce approximations.
		"""
		from .hooke import Hooke
		hooke = metric.hooke
		if metric.vdim==2:
			linear = ad.array([hooke[0,0],hooke[1,1]])
			quadratic = ad.array([[hooke[0,0],hooke[0,2]],[hooke[2,0],hooke[2,2]]])
			cubic = None
			raise ValueError("TODO : correct implementation")
		elif metric.vdim==3:
			linear = ad.array([hooke[0,0],hooke[1,1],hooke[2,2]])
			quadratic = None
			cubic = None
			if np.any((hooke[1,3]!=0.,hooke[2,4]!=0.)):
				raise ValueError("Impossible conversion")
			raise ValueError("TODO : correct implementation")
		else:
			raise ValueError("Unsupported dimension")

		return cls(linear,quadratic,cubic,*super(Hooke,metric).__iter__())

	@classmethod
	def from_Thomsen(cls,Vp,Vs,eps,delta,rho=1):
		"""
		Constructs a VTI (vertical transversely anisotropic) metric from Thomsen 
		parameters. See  "Weak elastic anisotropy" (Thomsen, 1986).
		- Vp (m/s)
		- Vs (m/s)
		- eps (dimensionless)
		- delta (dimensionless)
		- rho (g/cm^3)
		"""
		aa = -(1+2*eps)*Vp**2*Vs**2
		bb = -Vp**2*Vs**2
		cc = -(1+2*eps)*Vp**4-Vs**4+(Vp**2-Vs**2)*(Vp**2*(1+2*delt)-Vs**2)
		dd = Vs**2+(1+2*eps)*Vp**2
		ee = Vp**2+Vs**2

		linear = ad.array([dd,dd,ee])
		quadratic = ad.array([[aa,2*aa,cc],[2*aa,aa,cc],[cc,cc,bb]])
		cubic = None

		# TODO: Tensor normalization
		assert(rho==1)

		return cls(linear,quadratic,cubic)

	@classmethod
	def ThomsenExample(cls,i=None):
		"""
		List of VTI examples taken from article "Weak elastic anisotropy" (Thomsen, 1986), 
		with Thomsen parameters [Vp,Vs,epsilon,delta]. 
		"""
		# Code by F. Desquilbet, 2020
		# TODO : put the densities, and the names
		Vp_Vs_eps_delta_tab = np.array([
			[3368,1829,0.110,-0.035],
		    [4529,2703,0.034,0.211],
		    [4476,2814,0.097,0.091],
		    [4099,2346,0.077,0.010],
		    [4972,2899,0.056,-0.003],
		    [4349,2571,0.091,0.148],
		    [3928,2055,0.334,0.730],
		    [4539,2706,0.060,0.143],
		    [4449,2585,0.091,0.565],
		    [4672,2833,0.023,0.002],
		    [3794,2074,0.189,0.204],
		    [5460,3219,0.000,-0.264],
		    [4418,2587,0.053,0.158],
		    [4405,2542,0.080,-0.003],
		    [5073,2998,0.010,0.012],
		    [4869,2911,0.033,0.040],
		    [4296,2471,0.081,0.129],
		    
		    [3383,2438,0.065,0.059],
		    [3688,2774,0.081,0.057],
		    [3901,2682,0.137,-0.012],
		    [4237,3018,0.036,-0.039],
		    [4846,3170,0.063,0.008],
		    [4633,3231,-0.026,-0.033],
		    [4359,3048,0.172,0.000],
		    [3962,2926,0.055,-0.089],
		    [3749,2621,0.128,0.078],
		    [1875,826,0.225,0.100],
		    [1058,387,0.215,0.315],
		    [4130,2380,0.085,0.120],
		    [4721,2890,0.135,0.205],
		    [2074,869,0.110,0.090],
		    [2106,887,0.195,0.175],
		    [2202,969,0.015,0.060],
		    [3048,1490,0.255,-0.050],
		    
		    [3377,1490,0.200,-0.075],
		    [4231,2539,0.200,0.100],
		    [4167,2432,0.040,0.010],
		    [4404,2582,0.025,0.055],
		    [4206,2664,0.002,0.020],
		    [3810,2368,0.030,0.045],
		    [3292,1768,0.195,-0.220],
		    [5029,2987,-0.005,-0.015],
		    [4877,2941,0.045,-0.045],
		    [4846,1856,0.020,-0.030],
		    [4420,2091,1.12,-0.235],
		    [6096,4481,-0.096,0.273],
		    [5334,3353,0.369,0.579],
		    [4054,1341,1.222,-0.388],
		    [6340,4389,0.097,0.586],
		    [3627,1676,-0.038,-0.164],
		    [2868,1350,0.97,0.09],
		    
		    [3009,1654,0.013,-0.001],
		    [3009,1654,0.059,-0.001],
		    [3306,1819,0.134,0.000],
		    [3306,1819,0.169,0.000],
		    [2745,1508,0.103,-0.001],
		    [1409,780,0.022,0.018],
		    [1911,795,1.161,-0.140]
		    ])

		if i is None: 
			return Vp_Vs_eps_delta_tab
		else:
			return cls.from_Thomsen(*Vp_Vs_eps_delta_tab[i])


	@classmethod
	def from_Hooke(cls,metric):
		"""
		Generate reduced algebraic form from full Hooke tensor.
		!! Warning !!  Hooke to Reduced conversion requires that some 
		coefficients of the Hooke tensor vanish, which is currently not checked,
		and may induce approximations.
		"""
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
			cubic = None

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
			cubic = None

		else:
			raise ValueError("Unsupported dimension")

		return cls(linear,quadratic,cubic,*super(Hooke,metric).__iter__())




