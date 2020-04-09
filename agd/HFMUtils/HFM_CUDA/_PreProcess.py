# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import copy

from . import misc
from .. import Grid
from ... import FiniteDifferences as fd
from ... import AutomaticDifferentiation as ad
from ... import Metrics

# This file implements some member functions of the Interface class of HFM_CUDA

def Metric(self,x):
	if self.isCurvature: 
		raise ValueError("No metric available for curvature penalized models")
	if hasattr(self,'metric'): return self.metric.at(x)
	else: return self.dualMetric.at(x).dual()


def SetGeometry(self):
	if self.verbosity>=1: print("Prepating the domain data (shape,metric,...)")

	# Domain shape and grid scale
	self.shape = tuple(self.GetValue('dims',
		help="dimensions (shape) of the computational domain").astype(int))

	self.periodic_default = (False,False,True) if self.isCurvature else (False,)*self.ndim
	self.periodic = self.GetValue('periodic',default=self.periodic_default,
		help="Apply periodic boundary conditions on some axes")
	self.shape_o = tuple(misc.round_up(self.shape,self.shape_i))
	if self.bound_active_blocks is True: 
		self.bound_active_blocks = 12*np.prod(self.shape_o) / np.max(self.shape_o)
	
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

	self.drift = self.GetValue('drift', default=None, verbosity=3, array_float=True,
		help="Drift introduced in the eikonal equation, becoming F(grad u - drift)=1")

	# Get the metric 
	if   self.model.startswith('Riemann'): self.metricClass = Metrics.Riemann
	elif self.model.startswith('Rander') : self.metricClass = Metrics.Rander
	else: self.metricClass = None

	if self.metricClass is not None:
		self.metric = self.GetValue('metric',default=None,verbosity=3,
			help="Metric of the minimal path model")
		self.dualMetric = self.GetValue('dualMetric',default=None,verbosity=3,
			help="Dual metric of the minimal path model")


	self.cost_based = self.model.startswith('Isotropic') or self.isCurvature
	if self.cost_based:
		self.metric = self.GetValue('cost',None,verbosity=3,
			help="Cost function for the minimal paths. cost = 1/speed.")
		self.dualMetric = self.GetValue('speed',None,verbosity=3,
			help="Speed function for the minimal paths. speed = 1/cost.")
	else:

	# Import from HFM format, and make copy, if needed
	if self.model.startswith('Isotropic') or self.isCurvature: 
		metricClass = Metrics.Isotropic
	elif self.model.startswith('Riemann'): metricClass = Metrics.Riemann
	elif self.model.startswith('Rander') : metricClass = Metrics.Rander

	overwriteMetric = self.GetValue("overwriteMetric",default=False,
			help="Allow overwriting the metric or dualMetric")

	if ad.cupy_generic.isndarray(self.metric) or np.isscalar(self.metric): 
		self.metric = metricClass.from_HFM(self.metric)
	elif not overwriteMetric:
		self.metric = copy.deepcopy(self.metric)
	if ad.cupy_generic.isndarray(self.dualMetric) or np.isscalar(self.dualMetric): 
		self.dualMetric = metricClass.from_HFM(self.dualMetric)
	elif not overwriteMetric:
		self.dualMetric = copy.deepcopy(self.dualMetric)

	# Rescale 
	h_base = self.h[0] if self.isCurvature else self.h
	if self.metric is not None: 	self.metric =     self.metric.rescale(1/h_base)
	if self.dualMetric is not None: self.dualMetric = self.dualMetric.rescale(h_base)
	if self.drift is not None: self.drift*=self.h_broadcasted

	self.costVariation = self.GetValue('costVariation',default=None,
		help = "First order variation of the cost function "
		"(defined as 1 in the case of an anisotropic metric)")
	if self.cost_based: 
		# These models internally only use the cost function, not the speed function
		if self.metric is None: self.metric = self.dualMetric.dual()
		self.dualMetric = None
		if ad.is_ad(self.metric.cost):
			assert self.costVariation is None
			self.costVariation = self.metric.cost.coef
			self.metric = Metrics.Isotropic(self.metric.cost.value)
		# Internally, we always use a relative cost variation
		if self.costVariation is not None:
			self.costVariation/=cp.expand_dims(self.metric.cost,axis=-1)

	if self.isCurvature:
		self.xi = self.GetValue('xi',
			help="Cost of rotation for the curvature penalized models")
		self.kappa = self.GetValue('kappa',default=0.,
			help="Rotation bias for the curvature penalized models")
		self.theta = self.GetValue('theta',default=0.,verbosity=3,
			help="Deviation from horizontality, for the curvature penalized models")

		self.xi *= self.h_per
		self.kappa /= self.h_per

		self.geom = ad.array([e for e in 
			(self.metric.cost,self.xi,self.kappa,self.theta) if not np.isscalar(e)])

	else: # not self.isCurvature
		if self.model.startswith('Isotropic'):
			self.geom = self.metric.cost
		elif self.model.startswith('Riemann'):
			if self.dualMetric is None: self.dualMetric = self.metric.dual()
			self.geom = self.dualMetric.flatten()
		elif self.model.startswith('Rander'):
			if self.metric is None: self.metric = self.dualMetric.dual()
			self.geom = Metrics.Riemann(self.metric.m).dual().flatten()
			if self.drift is None: self.drift = self.metric.w
			else: self.drift += self.metric.w

		# TODO : remove. No need to create this grid for our interpolation
		grid = ad.array(np.meshgrid(*(cp.arange(s,dtype=self.float_t) 
			for s in self.shape), indexing='ij')) # Adimensionized coordinates
		self.metric.set_interpolation(grid,periodic=self.periodic) # First order interpolation


	self.block['geom'] = misc.block_expand(fd.as_field(self.geom,self.shape),
		self.shape_i,mode='constant',constant_values=np.inf,contiguous=True)
	if self.drift is not None:
		self.block['drift'] = misc.block_expand(fd.as_field(self.drift,self.shape),
			self.shape_i,mode='constant',constant_values=np.nan,contiguous=True)

	tol_msg = "Convergence tolerance for the fixed point solver"
	mean_cost_magnitude_msg = ("Upper bound on the magnitude of the cost function, "
	"or equivalent quantity for a metric, used to set the 'tol' and 'minChg_delta_min' "
	"parameters of the eikonal solver")

	tol = self.GetValue('tol',default="_Dummy",help=tol_msg)
	if isinstance(tol,str) and tol=="_Dummy":
		if self.metric is None: self.metric=self.dualMetric.dual()
		mean_cost_bound = float(np.mean(self.metric.cost_bound()))
		float_resolution = np.finfo(self.float_t).resolution
		tol = mean_cost_bound * float_resolution * 5.
		if not self.multiprecision: tol*=np.sum(self.shape)
		self.hfmOut['keys']['defaulted']['tol']=tol
	self.tol = self.float_t(tol)

	if self.bound_active_blocks:
		self.minChg_delta_min = self.GetValue('minChg_delta_min',default=float(self.h)/10.,
			help="Minimal threshold increase for bound_active_blocks variants")


	# geometrical data related with geodesics geodesics
	self.exportGeodesicFlow = self.GetValue('exportGeodesicFlow',default=False,
		help="Export the upwind geodesic flow (direction of the geodesics)")
	self.tips = self.GetValue('tips',default=None,array_float=True,
		help="Tips from which to compute the minimal geodesics")
	if self.isCurvature:
		self.unorientedTips=self.GetValue('unorientedTips',default=None,array_float=True,
			help="Compute a geodesic from the most favorable orientation")
	self.hasTips = (self.tips is not None 
		or self.isCurvature and self.unorientedTips is not None)


def SetValuesArray(self):
	if self.verbosity>=1: print("Preparing the values array (setting seeds,...)")
	values = cp.full(self.shape,np.inf,dtype=self.float_t)
	self.seeds = self.GetValue('seeds',
		help="Points from where the front propagation starts",array_float=True)
	assert self.seeds.ndim==2 and self.seeds.shape[1]==self.ndim
	self.seeds = Grid.PointFromIndex(self.hfmIn,self.seeds,to=True) # Adimensionize seed position
	if len(self.seeds)==1: self.seed=self.seeds[0]
	seedValues = cp.zeros(len(self.seeds),dtype=self.float_t)
	seedValues = self.GetValue('seedValues',default=seedValues,
		help="Initial value for the front propagation",array_float=True)
	if not ad.is_ad(seedValues):
		seedValueVariation = self.GetValue('seedValueVariation',default=None,
			help="First order variation of the seed values",array_float=True)
		if seedValueVariation is not None:
			denseAD_cupy = ad.cupy_generic.cupy_rebase(ad.Dense.denseAD)
			seedValues = denseAD_cupy.new(seedValues,seedValueVariation.T)
	seedRadius = self.GetValue('seedRadius',default=0.,
		help="Spread the seeds over a radius given in pixels, so as to improve accuracy.")


	if seedRadius==0.:
		seedIndices = np.round(self.seeds).astype(int)
		values[tuple(seedIndices.T)] = seedValues
		self.seedValues = seedValues
		self.seedIndices = seedIndices
	else:
		neigh = Grid.GridNeighbors(self.hfmIn,self.seed,seedRadius) # Geometry last
		r = seedRadius 
		aX = [cp.arange(int(np.floor(ci-r)),int(np.ceil(ci+r)+1)) for ci in self.seed]
		neigh =  ad.stack(cp.meshgrid( *aX, indexing='ij'),axis=-1)
		neigh = neigh.reshape(-1,neigh.shape[-1])
		neighValues = seedValues.repeat(len(neigh)//len(self.seeds)) # corrected below

		# Select neighbors which are close enough
		neigh = neigh[ad.Optimization.norm(neigh-self.seed,axis=-1) < r]

		# Periodize, and select neighbors which are in the domain
		nper = np.logical_not(self.periodic)
		inRange = np.all(np.logical_and(-0.5<=neigh[:,nper],
			neigh[:,nper]<cp.array(self.shape)[nper]-0.5),axis=-1)
		neigh = neigh[inRange,:]
		neighValues = neighValues[inRange,:]
		
		diff = (neigh - self.seed).T # Geometry first
#			neigh[:,self.periodic] = neigh[:,self.periodic] % self.shape[self.periodic]
		metric0 = self.Metric(self.seed)
		metric1 = self.Metric(neigh.T)
		self.seedValues = neighValues+0.5*(metric0.norm(diff) + metric1.norm(diff))
		self.seedIndices = neigh
		values[tuple(self.seedIndices.T)] = ad.remove_ad(self.seedValues)

	block_values = misc.block_expand(values,self.shape_i,
		mode='constant',constant_values=np.inf,contiguous=True)

	# Tag the seeds
	if np.prod(self.shape_i)%8!=0:
		raise ValueError('Product of shape_i must be a multiple of 8')
	self.seedTags = cp.isfinite(values)
	block_seedTags = misc.block_expand(self.seedTags,self.shape_i,
		mode='constant',constant_values=True,contiguous=True)
	block_seedTags = misc.packbits(block_seedTags,bitorder='little')
	block_seedTags = block_seedTags.reshape( self.shape_o + (-1,) )
	block_values[cp.isnan(block_values)] = np.inf

	self.block.update({'values':block_values,'seedTags':block_seedTags})

	# Handle multiprecision
	if self.multiprecision:
		block_valuesq = cp.zeros(block_values.shape,dtype=self.int_t)
		self.block.update({'valuesq':block_valuesq})

	if self.strict_iter_o:
		self.block['valuesNext']=block_values.copy()
		if self.multiprecision:
			self.block['valuesqNext']=block_valuesq.copy()

	if self.bound_active_blocks:
		minChg = cp.full(self.shape_o,np.inf,dtype=self.float_t)
		self.block['minChgPrev_o'] = minChg
		self.block['minChgNext_o'] = minChg.copy()

	for key,value in self.block.items():
		if value.dtype.type not in (self.float_t,self.int_t,np.uint8):
			raise ValueError(f"Inconsistent type {value.dtype.type} for key {key}")