"""
This module implements some basic functionality for solving ODEs derived from a 
Hamiltonian in a manner compatible with automatic differentiation.
(Flow computation, symplectic schemes, etc)

Recall that Hamilton's equations read 
dq/dt =  dH/dp
dp/dt = -dH/dq

Given a metric F, the corresponding canonical Hamiltonian is defined as 
H(q,p) = (1/2) F^*_q(p)^2.
In written words, the Hamiltonian is the half square of the dual metric.
"""

import scipy.sparse
import numpy as np
from copy import copy

from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from .base import Base

lo = ad.left_operand

def fixedpoint(f,x,tol=1e-9,nitermax=100):
	"""
	Iterates the function f on the data x until a fixed point is found, 
	up to prescribed tolerance, or the maximum number of iterations is reached.
	"""
	norm_infinity = ad.Optimization.norm_infinity
	x_old = x
	for i in range(nitermax):
		x=f(x)
		if norm_infinity(x-x_old)<tol: break
		x_old = x
	return x


class Hamiltonian(object):
	def __init__(self, H, shape_free=None, grid=None):
		"""
		Inputs:
		- H : the hamiltonian, which may be either:
			* a metric
		 	* a callable function
			* a pair of callable functions, for a separable hamiltonian.
				(in that case, may also be scalars or matrices, for quadratic hamiltonians)
		- shape_free (optional) : the shape of the position and impulsion
		- grid : if not None, used dual metric interpolated on the grid
		"""		
		if isinstance(H,list): H = tuple(H)
		if grid is not None:
			assert isinstance(H,Base)
			H = H.dual()
			H.set_interpolation(grid)
		self._H = H
		if self.is_separable: assert len(H)==2

		if self.is_metric:
			assert shape_free is None
			self.vdim = self._H.vdim
		else:
			self.shape_free = shape_free

	@property
	def is_separable(self):
		return isinstance(self._H,tuple)
	@property
	def is_metric(self):
		return isinstance(self._H,Base)

	@property
	def vdim(self):
		"""
		Dimension of space of positions. 
		(Also equals the dimension of the space of impulsions)
		Returns none for a Hamiltonian acting on scalars.
		"""
		if shape_free is None: raise ValueError("Unspecified space dimensions")
		elif len(self.shape_free)==0: return None
		elif len(self.shape_free)==1: return self.shape_free[0]
		else: raise ValueError(f"Hamiltonian acts on tensors of shape {shape_free}")

	@vdim.setter
	def vdim(self,vdim):
		if vdim is None:
			self.shape_free = tuple()
		else:
			self.shape_free = (vdim,)	

	def _checkdim(self,x):
		"""
		Checks the dimension of the input variable agains shape_free.
		(Sets shape_free if it is undefined, assuming depth one of x)
		"""
		if self.shape_free is None:
			self.vdim = x.shape[0]
		assert self.shape_free==tuple() or x.shape[:len(self.shape_free)]==self.shape_free

	def separable_quadratic_set_sparse_matrices(self):
		"""
		If the hamiltonian is separable and quadratic, replace the callable functions
		with sparse matrices, for faster evaluation.
		"""
		assert self.is_separable
		x_ad = ad.Sparse2.identity(shape=self.shape_free)
		self._H = tuple(scipy.sparse.coo_matrix(f(x_ad).triplets()).tocsc()
			if callable(f) else f
			for f in self._H)

	def H(self,q,p):
		"""
		Evaluates the Hamiltonian, for a given position and impulsion.
		"""
		if self.is_separable:
			def value(f,x):
				"""Evaluates a function with the given structure."""
				if callable(f): return f(x)
				else: return 0.5*lp.dot_VV(x,ad.apply_linear_mapping(f,x))

			return sum(lo(value(h,x)) for (h,x) in zip(self._H,(q,p)) )
		elif self.is_metric:
			return self._H.at(q).norm2(p)
		else:
			return self._H(q,p)

	def _gradient_ad(self,x):
		"""
		Extracts the gradient from an AD variable and reshapes as required.
		"""
		g = x.gradient()
		return g.reshape(self.shape_free+g.shape[1:])

	def _gradient(self,f,x):
		"""
		Differentiates a function with the given structure.
		"""
		if callable(f):
			x_ad = ad.Dense.identity(constant=x,shape_free=self.shape_free) 
			return self._gradient_ad( f(x_ad) )
		else:
			return ad.apply_linear_mapping(f,x)

	def DqH(self,q,p):
		"""
		Differentiates the Hamiltonian, w.r.t. position.
		"""
		self._checkdim(q)
		if self.is_separable:
			return self._gradient(self._H[0],q)

		q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
		if self.is_metric:
			return self._gradient_ad(self._H.at(q_ad).norm2(p))
		else: 
			return self._gradient_ad(self._H(q_ad,p))

	def DpH(self,q,p):
		"""
		Differentiates the Hamiltonian, w.r.t. impulsion.
		"""
		self._checkdim(p)
		if self.is_separable:
			return self._gradient(self._H[1],p)
		elif self.is_metric:
			return self._H.at(q).gradient2(p)
		else: 
			p_ad = ad.Dense.identity(constant=p,shape_free=self.shape_free)
			return self._gradient_ad(self._H(q,p_ad))

	def flow(self,q,p):
		"""
		Symplectic gradient of the Hamiltonian.
		"""
		return (self.DpH(q,p),-self.DqH(q,p))

	def integrate(self,q,p,scheme,niter,T=1,path=False):
		"""
		Solves Hamilton's equations by running the scheme niter times.
		Inputs : 
			- q,p : Initial position and impulsion.
			- scheme : ODE integration scheme. (string or callable)
			- niter : number of steps
			- T : total time
			- path : wether to return the intermediate steps
		Output : 
			- q,p if path is False. 
			Otherwise [q0,...,qn],[p0,...,pn],[t0,..tn], with n=niter, tn=T.
		"""
		if isinstance(scheme,str):
			schemes = self.nonsymplectic_schemes()
			schemes.update(self.symplectic_schemes())
			scheme = schemes[scheme]

		dt = T/niter
		q,p = copy(q),copy(p)
		if path: Q,P = [copy(q)],[copy(p)]

		for i in range(niter):
			q,p = scheme(q,p,dt)
			if path: Q.append(copy(q)); P.append(copy(p))

		if path: 
			ndim_free = len(self.shape_free)
			return (ad.stack(Q,axis=ndim_free),
				ad.stack(P,axis=ndim_free),
				np.linspace(0,T,niter+1))
		return q,p

	def nonsymplectic_schemes(self):
		"""
		Standard ODE integration schemes
		"""
		def Euler(q,p,dt):
			dq,dp = self.flow(q,p)
			return q+dt*dq, p+dt*dp

		def RK2(q,p,dt):
			dq1,dp1 = self.flow(q, p)
			dq2,dp2 = self.flow(q+0.5*dt*dq1, p+0.5*dt*dp1)
			return q+dt*dq2, p+dt*dp2

		def RK4(q,p,dt):
			dq1,dp1 = self.flow(q, p)
			dq2,dp2 = self.flow(q+0.5*dt*dq1, p+0.5*dt*dp1)
			dq3,dp3 = self.flow(q+0.5*dt*dq2, p+0.5*dt*dp2)
			dq4,dp4 = self.flow(q+dt*dq3, p+dt*dp3)
			return q+dt*(dq1+2*dq2+2*dq3+dq4)/6., p+dt*(dp1+2*dp2+2*dp3+dp4)/6.

		return {"Euler":Euler,"Runge-Kutta-2":RK2,"Runge-Kutta-4":RK4}

	def incomplete_schemes(self,solver=None):
		"""
		Incomplete schemes, updating only position or impulsion. 
		Inputs : 
			- solver (optional). Numerical solver used for the implicit steps.
			Defaults to a basic fixed point solver: "fixedpoint" in the same package.
		"""
		def Expl_q(q,p,dt):
			return q + dt*self.DpH(q,p)

		def Expl_p(q,p,dt):
			return p - dt*self.DqH(q,p)

		if self.is_separable:
			return {"Explicit-q":Expl_q,"Explicit-p":Expl_p,
			"Implicit-q":Expl_q,"Implicit-p":Expl_p}

		if solver is None:
			solver=fixedpoint

		def Impl_q(q,p,dt):
			def f(q_): return q+dt*self.DpH(q_,p)
			return solver(f,q)
		def Impl_p(q,p,dt):
			def f(p_): return p-dt*self.DqH(q,p_)
			return solver(f,p)
		
		return {"Explicit-q":Expl_q,"Explicit-p":Expl_p,
			"Implicit-q":Impl_q,"Implicit-p":Impl_p}

	def symplectic_schemes(self,**kwargs):
		"""
		Symplectic schemes, alternating updates of position and impulsion.
		The first updated variable is indicated with a suffix.

		In the non-separable cases, some of the updates are implicit, otherwise
		they are explicit.
		Inputs : 
			- kwargs. Passed to self.incomplete_schemes
		"""
		incomp = self.incomplete_schemes(**kwargs)
		if self.is_separable:
			Expl_q,Expl_p = tuple(incomp[s] for s in 
			("Explicit-q","Explicit-p"))

			def Euler_p(q,p,dt):
				p = Expl_p(q,p,dt)
				q = Expl_q(q,p,dt)
				return q,p
				
			def Euler_q(q,p,dt):
				q = Expl_q(q,p,dt)
				p = Expl_p(q,p,dt)
				return q,p 
				
			def Verlet_p(q,p,dt):
				p=Expl_p(q,p,dt/2)
				q=Expl_q(q,p,dt)
				p=Expl_p(q,p,dt/2)
				return q,p
				
			def Verlet_q(q,p,dt):
				q=Expl_q(q,p,dt/2)
				p=Expl_p(q,p,dt)
				q=Expl_q(q,p,dt/2)
				return q,p

			def Ruth3_p(q,p,dt):
				"""Ronald Ruth 1983, as of Wikipedia"""
				c1,c2,c2 = 1., -2./3., 2./3.
				d1,d2,d3 = -1./24, 3./4, 7./24
				p=Expl_p(q,p,c1*dt)
				q=Expl_q(q,p,d1*dt)
				p=Expl_p(q,p,c2*dt)
				q=Expl_q(q,p,d2*dt)
				p=Expl_p(q,p,c3*dt)
				q=Expl_q(q,p,d3*dt)
				return q,p

			def Ruth4_p(q,p,dt):
				"""Ronald Ruth 1983, as of Wikipedia"""
				t = 2.**(1./3.)
				c1 = 1/(2*(2-t)); c2 = (1-t)*c1; c3,c4 = c2,c1
				d1 = 2*c1; d2 = -t/d1; d3,d4 = d1,0
				p=Expl_p(q,p,c1*dt)
				q=Expl_q(q,p,d1*dt)
				p=Expl_p(q,p,c2*dt)
				q=Expl_q(q,p,d2*dt)
				p=Expl_p(q,p,c3*dt)
				q=Expl_q(q,p,d3*dt)
				p=Expl_p(q,p,c4*dt) 
				#q=Expl_q(q,p,d4*dt)
				return q,p


			return {"Euler-p":Euler_p,"Euler-q":Euler_q,
			"Verlet-p":Verlet_p,"Verlet-q":Verlet_q,
			"Ruth3-p":Ruth3_p,"Ruth4-p":Ruth4_p}


		else: # Non separable case
			Expl_q,Expl_p,Impl_q,Impl_p = tuple(incomp[s] for s in 
				("Explicit-q","Explicit-p","Implicit-q","Implicit-p"))

			def Euler_p(q,p,dt):
				p=Impl_p(q,p,dt)
				q=Expl_q(q,p,dt)
				return q,p

			def Euler_q(q,p,dt):
				q=Impl_q(q,p,dt)
				p=Expl_p(q,p,dt)
				return q,p

			def Verlet_p(q,p,dt):
				p=Impl_p(q,p,dt/2.)
				q=Expl_q(q,p,dt/2.)
				p=Impl_q(q,p,dt/2.)
				q=Expl_p(q,p,dt/2.)
				return q,p

			def Verlet_q(q,p,dt):
				q=Impl_q(q,p,dt/2.)
				p=Expl_p(q,p,dt/2.)
				p=Impl_p(q,p,dt/2.)
				q=Expl_q(q,p,dt/2.)
				return q,p
				
			return {"Euler-p":Euler_p,"Euler-q":Euler_q,
			"Verlet-p":Verlet_p,"Verlet-q":Verlet_q}


