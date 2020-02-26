import numpy as np
import itertools
import scipy.linalg

from . import AutomaticDifferentiation as ad


"""
This file implements some basic spline interpolation methods,
in a manner compatible with automatic differentiation.
"""

class _spline_univariate(object):
	"""
	A univariate spline of a given order.
	"""
	def __init__(self,order):
		assert isinstance(order,int)
		self.order=order
		if not (order>=1 and order<=3):
			raise ValueError("Unsupported spline order")

	def __call__(self,x):
		x=ad.array(x)
		if   self.order==1: return self._call1(x)
		elif self.order==2: return self._call2(x)
		elif self.order==3: return self._call3(x)
		assert False

	def nodes(self):
		"""
		For each node, the spline is nonzero on ]node,node+1[.
		"""
		if   self.order==1: return range(-1,1)
		elif self.order==2: return range(-1,2)
		elif self.order==3: return range(-2,2)
		assert False

	def _call1(self,x):
		"""
		A piecewise linear spline, defined over [-1,1].
		"""
		return np.maximum(0.,1.-np.abs(x))

	def _call2(self,x):
		"""
		A piecewise quadratic spline function, defined over [-1,2]
		"""
		result = ad.zeros_like(x)
	
		# interval [-1,0[
		pos = np.logical_and(x>=-1,x<0)
		x_ = x[pos]+1
		result[pos] = x_**2

		# interval [0,1[
		pos = np.logical_and(x>=0,x<1)
		x_ = x[pos]-0.5
		result[pos] = 1.5 - 2.*x_**2

		# interval [1,2[
		pos = np.logical_and(x>=1,x<2)
		x_ = 2.-x[pos]
		result[pos] = x_**2 

		return result

	def _call3(self,x):
		"""
		A piecewise cubic spline function, defined over [-2,2]
		"""
		result = ad.zeros_like(x)

		def f(y): return 3.*y**2 - y**3

		# interval [-2,-1[
		pos = np.logical_and(x>=-2,x<-1)
		result[pos] = f(x[pos]+2)

		# interval [-1,0[
		pos = np.logical_and(x>=-1,x<0)
		result[pos] = 4.-f(-x[pos])

		#interval [0,1[
		pos = np.logical_and(x>=0,x<1)
		result[pos] = 4-f(x[pos])

		#interval [1,2[
		pos = np.logical_and(x>=1,x<2)
		result[pos] = f(2.-x[pos])

		return result


	def make_coefs(self,values,periodic,overwrite_values=False):
		"""
		Produces the node coefficients corresponding to given values.
		!! Call convention : interpolation is along the first axis. !!
		"""
		if self.order==1: 
			return values

		n = len(values)
		if periodic: 
			raise ValueError("Periodic interpolation is not supported for degree > 1")

		if self.order==2:
			band = np.zeros( (2,n) )
			band[0,:] = self.__call__(0.)
			band[1,:-1]  = self.__call__(1.)
			return scipy.linalg.solve_banded((1,0),band,values,
				overwrite_ab=True,overwrite_b=overwrite_values) 
		elif self.order==3:
			band = np.zeros( (2,n) )
			band[0,1:] = self.__call__(1.)
			band[1,:]  = self.__call__(0.)
			return scipy.linalg.solveh_banded(band,values,
				overwrite_ab=True,overwrite_b=overwrite_values) 



class _spline_tensor(object):
	"""
	A tensor product of univariate splines.
	"""
	def __init__(self,orders):
		assert isinstance(orders,tuple)
		self.splines = tuple(_spline_univariate(order) for order in orders)

	@property
	def order(self):
		return tuple(spline.order for spline in self.splines)
	@property
	def vdim(self):
		return len(self.splines)

	def __call__(self,x):
		return np.prod( tuple(spline(xi) for (xi,spline) in zip(x,self.splines)) ,axis=0)

	def nodes(self):
		"""
		for each node, the spline is non-zero on node+]0,1[**vdim
		"""
		_nodes = tuple(spline.nodes() for spline in self.splines)
		return np.array(tuple(itertools.product(*_nodes))).T

	def make_coefs(self,values,periodic,overwrite_values=False):
		"""
		Produces the node coefficients corresponding to given values.
		!! Call convention : interpolation is along the first axes. !!
		"""
		assert len(periodic)==len(self.splines)
		for i,(spline,per) in enumerate(zip(self.splines,periodic)):
			values = np.moveaxis(spline.make_coefs(
				np.moveaxis(values,i,0),per,overwrite_values=overwrite_values),0,i)
		return values

def _append_dims(x,ndim):
	return np.reshape(x,x.shape+(1,)*ndim)

class UniformGridInterpolation(object):
	"""
	Interpolates values on a uniform grid, in arbitrary dimension, using splines of 
	a given order.
	"""

	def __init__(self,grid,values=None,order=1,periodic=False):
		"""
		- grid (ndarray) : must be a uniform grid. E.g. np.meshgrid(aX,aY,indexing='ij')
		 where aX,aY have uniform spacing.
		- values (ndarray) : interpolated values.
		- order (int, tuple of ints) : spline interpolation order (<=3), along each axis.
		- periodic (bool, tuple of bool) : wether periodic interpolation, along each axis.
		"""
		grid = ad.array(grid)
		self.shape = grid.shape[1:]
		self.origin = grid.__getitem__((slice(None),)+(0,)*self.vdim)
		self.scale = grid.__getitem__((slice(None),)+(1,)*self.vdim) - self.origin
		

		if order is None: order = 1
		if isinstance(order,int): order = (order,)*self.vdim
		self.spline = _spline_tensor(order)
		assert self.spline.vdim == self.vdim
		self.nodes = self.spline.nodes()

		if periodic is None: periodic=False
		if isinstance(periodic,bool): periodic = (periodic,)*self.vdim
		self.periodic = periodic

		self.coef = None if values is None else self.make_coefs(values)

	@property
	def vdim(self):
		"""
		Dimension of the interpolation points.
		"""
		return len(self.shape)

	@property
	def odim(self):
		"""
		Dimension of the interpolated objects.
		"""
		return self.coef.ndim - self.vdim

	def __call__(self,x):
		"""
		Interpolates the data at the position x.
		"""
		x=ad.array(x)
		assert len(x)==self.vdim
		pdim = x.ndim-1 # Number of dimensions of position
		# Rescale the coordinates in reference rectangle
		origin,scale = (_append_dims(e,pdim) for e in (self.origin,self.scale))
		y = np.expand_dims((x - origin)/scale,axis=1)
		# Bottom left pixel
		yf = np.floor(y).astype(int)
		# All pixels supporting an active spline
		ys = yf - _append_dims(self.nodes,pdim)
		
		# Spline coefficients, taking care of out of domain 
		ys_ = ys.copy()
		out = np.full(x.shape[1:],False)
		for i,(d,per) in enumerate(zip(self.shape,self.periodic)):
			if per: 
				ys_[i] = ys_[i]%d
			else: 
				bad = np.logical_or(ys_[i]<0,ys_[i]>=d)
				out = np.logical_or(out,bad)
				ys_[i,bad] = 0 

		coef = self.coef[tuple(ys_)]
		coef[out]=0.
		odim = self.odim
		coef = np.moveaxis(coef,range(-odim,0),range(odim))
		
		# Spline weights
		weight = self.spline(y-ys)

		return (coef*weight).sum(axis=odim)

	def set_values(self,values):
		self.coef = self.make_coefs(values)

	def make_coefs(self,values,overwrite_values=False):
		values = ad.array(values)
		odim = values.ndim - self.vdim
		# Internally, interpolation is along the first axes.
		# (Contrary to external interface)
		val = np.moveaxis(values,range(odim),range(-odim,0))

		return self.spline.make_coefs(val,periodic=self.periodic,
			overwrite_values=overwrite_values)












