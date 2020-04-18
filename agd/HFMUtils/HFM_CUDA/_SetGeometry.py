# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import copy

from . import misc
from . import inf_convolution
from .. import Grid
from ... import FiniteDifferences as fd
from ... import AutomaticDifferentiation as ad
from ... import Metrics

# This file implements some member functions of the Interface class of HFM_CUDA

def CostMetric(self,x):
	return self._CostMetric.at(x)

def SetGeometry(self):
	if self.verbosity>=1: print("Prepating the domain data (shape,metric,...)")
	eikonal = self.kernel_data['eikonal']
	policy = eikonal.policy

	# Domain shape and grid scale
	self.shape = tuple(self.GetValue('dims',
		help="dimensions (shape) of the computational domain").astype(int))

	self.periodic_default = (False,False,True) if self.isCurvature else (False,)*self.ndim
	self.periodic = self.GetValue('periodic',default=self.periodic_default,
		help="Apply periodic boundary conditions on some axes")
	self.shape_o = tuple(misc.round_up(self.shape,self.shape_i))
	if policy.bound_active_blocks is True: 
		policy.bound_active_blocks = 12*np.prod(self.shape_o) / np.max(self.shape_o)
	
	# Set the discretization gridScale(s)
	if self.isCurvature:
		self.h_base = self.GetValue('gridScale',array_float=True,
			help="Scale of the physical (not angular) grid.")
		self.h_per = self.caster(2.*np.pi / self.shape[2] )
		self.h = self.caster((self.h_base,self.h_base,self.h_per))

	elif self.HasValue('gridScale') or self.isCurvature:
		self.h = cp.broadcast_to(self.GetValue('gridScale',array_float=True,
			help="Scale of the computational grid"), (self.ndim,))

	else:
		self.h = self.GetValue('gridScales',array_float=True,
			help="Axis independent scales of the computational grid")

	self.h_broadcasted = fd.as_field(self.h,self.shape,depth=1)


	# Get the metric 
	if   self.model_=='Diagonal':metricClass = Metrics.Diagonal
	elif self.model_=='Riemann': metricClass = Metrics.Riemann
	elif self.model_=='Rander' : metricClass = Metrics.Rander
	elif self.model_=='TTI':     metricClass = Metrics.Seismic.Reduced

	if self.model_=='Isotropic':
		self._metric = Metrics.Diagonal(cp.ones(self.ndim,dtype=self.float_t))
		self._dualMetric = None
	elif self.isCurvature: 
		pass
	else:
		self._metric = self.GetValue('metric',default=None,verbosity=3,
			help="Metric of the minimal path model")
		self._dualMetric = self.GetValue('dualMetric',default=None,verbosity=3,
			help="Dual metric of the minimal path model")
		for key,value in (('_metric',self._metric),('_dualMetric',self._dualMetric)):
			if ad.cupy_generic.isndarray(value): 
				setattr(self,key,metricClass.from_HFM(value))

	self.drift = self.GetValue('drift', default=None, verbosity=3, array_float=True,
		help="Drift introduced in the eikonal equation, becoming F^*(grad u - drift)=1")

	# Set the geometry

	if self.isCurvature:
		# Geometry defined using the xi, kappa and theta parameters
		self.xi = self.GetValue('xi',array_float=True,
			help="Cost of rotation for the curvature penalized models")
		self.kappa = self.GetValue('kappa',default=0.,array_float=True,
			help="Rotation bias for the curvature penalized models")
		self.theta = self.GetValue('theta',default=0.,verbosity=3,array_float=True,
			help="Deviation from horizontality, for the curvature penalized models")

		# Scale h_base is taken care of through the 'cost' field
		h_ratio = self.h_per/self.h_base
		self.xi    *= h_ratio
		self.kappa /= h_ratio
		# Large arrays are passed as geometry data, and scalar entries as module constants
		geom = []
		def is_var(e): return isinstance(e,cp.ndarray) and e.ndim>0
		traits = eikonal.traits
		traits['xi_var_macro']    = int(is_var(self.xi))
		traits['kappa_var_macro'] = int(is_var(self.kappa))
		traits['theta_var_macro'] = int(is_var(self.theta))
		if not is_var(self.theta): traits['nTheta']=self.shape[2];
		if all(traits[e]==0 for e in ('xi_var_macro','kappa_var_macro','theta_var_macro')):
			traits['precomputed_scheme_macro']=1

		geom = [e for e in (1./self.xi,self.kappa,
			np.cos(self.theta),np.sin(self.theta)) if is_var(e)]
		if len(geom)>0: self.geom = ad.array(geom)
		else: self.geom = cp.zeros((0,)+self.shape, dtype=self.float_t)

	else:
		if self._metric is not None: self._metric = self._metric.with_costs(self.h)
		if self._dualMetric is not None:self._dualMetric=self._dualMetric.with_speeds(self.h)
		if self.drift is not None: self.drift *= self.h_broadcasted

		if self.model_=='Isotropic':
			# No geometry field. Metric passed as a module constant
			self.geom = cp.zeros((0,)+self.shape,dtype=self.float_t)
		elif self.model_=='Diagonal':
			self.geom = self.dualMetric.costs**2
		elif self.model_=='Riemann':
			self.geom = self.dualMetric.flatten()
		elif self.model_=='Rander':
			self.geom = Metrics.Riemann(self.metric.m).dual().flatten()
			if self.drift is None: self.drift = self.float_t(0.)
			self.drift += self.metric.w

	eikonal.args['geom'] = misc.block_expand(fd.as_field(self.geom,self.shape),
		self.shape_i,mode='constant',constant_values=np.inf,contiguous=True)
	if self.drift is not None:
		eikonal.args['drift'] = misc.block_expand(fd.as_field(self.drift,self.shape),
			self.shape_i,mode='constant',constant_values=np.nan,contiguous=True)

	# geometrical data related with geodesics 
	self.exportGeodesicFlow = self.GetValue('exportGeodesicFlow',default=False,
		help="Export the upwind geodesic flow (direction of the geodesics)")
	self.tips = self.GetValue('tips',default=None,array_float=True,
		help="Tips from which to compute the minimal geodesics")
	if self.isCurvature:
		self.unorientedTips=self.GetValue('unorientedTips',default=None,array_float=True,
			help="Compute a geodesic from the most favorable orientation")
	self.hasTips = (self.tips is not None 
		or (self.isCurvature and self.unorientedTips is not None))

	# Cost function
	if self.HasValue('speed'): 
		self.cost = 1. / self.GetValue('speed',array_float=True,
			help="speed = 1/cost (scales the metric, accepts AD)")
	else:
		self.cost = self.GetValue('cost',array_float=True,default=None,
			help="cost = 1/speed (scales the metric, accepts AD)")
		if self.cost is None: self.cost = cp.ones(self.shape,dtype=self.float_t)
	if not ad.is_ad(self.cost):
		costVariation = self.GetValue('costVariation',default=None,
			help="First order variation of the cost function")
		if costVariation is not None: self.cost = ad.Dense.new(self.cost,costVariation)
	if self.isCurvature: self.cost *= self.h_base
	self.cost = np.broadcast_to(self.cost,self.shape)

	# Cost related parameters
	if self.HasValue('atol') and self.HasValue('rtol'): tol = None
	else:
		tol = self.GetValue('tol',default="_Dummy",array_float=True,
			help="Convergence tolerance for the fixed point solver (determines atol, rtol)")
		float_resolution = np.finfo(self.float_t).resolution
		if isinstance(tol,str) and tol=="_Dummy":
			cost_bound = ad.remove_ad(self.cost)
			if not self.isCurvature: cost_bound *= self.metric.cost_bound()
			mean_cost_bound = np.mean(cost_bound)
			tol = mean_cost_bound * float_resolution * 5.
			self.hfmOut['keys']['default']['tol']=self.float_t(float(tol))
	policy.atol = self.GetValue('atol',default=tol,array_float=True,
		help="Absolute convergence tolerance for the fixed point solver")
	rtol_default = 0. if policy.multiprecision else float_resolution * 5.
	policy.rtol = self.GetValue('rtol',default=rtol_default, array_float=True,
		help="Relative convergence tolerance for the fixed point solver")

	if policy.bound_active_blocks:
		policy.minChg_delta_min = self.GetValue(
			'minChg_delta_min',default=float(np.min(self.h))/10.,
			help="Minimal threshold increase with bound_active_blocks method")

	# Walls
	walls = self.GetValue('walls',default=None,help='Obstacles in the domain')
	if walls is not None:
		wallDist_t = np.uint8
		wallDistBound = self.GetValue('wallsDistBound',default=10,
			help="Bound on the computed distance to the obstacles.\n"
			"(Ideally a sharp upper bound on the stencil width.)")
		wallDistMax_t = np.iinfo(wallDist_t).max
		wallDist = cp.full(self.shape, wallDistMax_t, dtype=wallDist_t)
		wallDist[walls]=0
		l1Kernel = inf_convolution.distance_kernel(1,self.ndim,dtype=wallDist_t,ord=1)
		wallDist = inf_convolution.inf_convolution(wallDist,l1Kernel,
			niter=wallDistBound,periodic=self.periodic,overwrite=True)
		# This value indicates 'far from wall', and visibility computation is bypassed
		wallDist[wallDist>wallsDistBound] = wallDistMax_t 
		self.wallDist = wallDist
		eikonal.args['wallDist'] = misc.block_expand(wallDist,self.shape_i,
			mode='constant',constant_values=np.iinfo(wallDist_t).max)
