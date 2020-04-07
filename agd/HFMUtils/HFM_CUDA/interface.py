# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import os
import time
import copy
from collections import OrderedDict

from . import kernel_traits
from . import misc
from . import cupy_module_helper
from .cupy_module_helper import GetModule

# Deferred implementation of Interface member functions
from . import _solvers 
from . import _GetGeodesics
from . import _PostProcess

from .. import Grid
from ... import FiniteDifferences as fd
from ... import AutomaticDifferentiation as ad
from ... import Metrics
from ... import LinearParallel as lp

class Interface(object):
	"""
	This class carries out the RunGPU function work. 
	It should not be used directly.
	"""
	def __init__(self,hfmIn):

		self.hfmIn = hfmIn
		if hfmIn['arrayOrdering'] != 'RowMajor':
			raise ValueError("Only RowMajor indexing supported")

		# Needed for GetValue
		self.help = hfmIn.get('help',[])
		self.hfmOut = {'keys':{
		'used':['exportValues','origin','arrayOrdering'],
		'defaulted':OrderedDict(),
		'visited':[],
		'help':OrderedDict()
		} }
		self.help_content = self.hfmOut['keys']['help']
		self.verbosity = 1

		self.verbosity = self.GetValue('verbosity',default=1,
			help="Choose the amount of detail displayed on the run")
		self.help = self.GetValue('help',default=[], # help on help...
			help="List of keys for which to display help")
		
		self.model = self.GetValue('model',help='Minimal path model to be solved.')
		# Unified treatment of standard and extended curvature models
		if self.model.endswith("Ext2"): self.model=self.model[:-4]+"2"

		self.ndim = len(hfmIn['dims'])
		
		self.returns=None
		self.xp = ad.cupy_generic.get_array_module(hfmIn,iterables=(dict,Metrics.Base))
		self.caster=ad.cupy_generic.array_float_caster(hfmIn,iterables=(dict,Metrics.Base))
		
	def HasValue(self,key):
		self.hfmOut['keys']['visited'].append(key)
		return key in self.hfmIn

	def GetValue(self,key,default="_None",verbosity=2,help=None,array_float=False):
		"""
		Get a value from a dictionnary, printing some help if requested.
		"""
		if key in self.help and key not in self.help_content:
			self.help_content[key]=help
			if self.verbosity>=1:
				if help is None: 
					print(f"Sorry : no help for key {key}")
				else:
					print(f"---- Help for key {key} ----")
					print(help)
					if isinstance(default,str) and default=="_None": 
						print("No default value")
					elif isinstance(default,str) and default=="_Dummy":
						print(f"see out['keys']['defaulted'][{key}] for default")
					else:
						print("default value :",default)
					print("-----------------------------")

		if key in self.hfmIn:
			self.hfmOut['keys']['used'].append(key)
			value = self.hfmIn[key]
			return self.caster(value) if array_float else value
		elif isinstance(default,str) and default == "_None":
			raise ValueError(f"Missing value for key {key}")
		else:
			assert key not in self.hfmOut['keys']['defaulted']
			self.hfmOut['keys']['defaulted'][key] = default
			if verbosity<=self.verbosity:
				print(f"key {key} defaults to {default}")
			return default

	def Warn(self,msg):
		if self.verbosity>=-1:
			print("---- Warning ----\n",msg,"\n-----------------\n")

	def Run(self):
		self.block={}

		self.SetKernelTraits()
		self.SetGeometry()
		self.SetValuesArray()
		self.SetKernel()
		self.SetSolver()
		self.PostProcess()
		self.GetGeodesics()
		self.FinalCheck()

		return self.hfmOut

	@property
	def isCurvature(self):
		if any(self.model.startswith(e) for e in ('Isotropic','Riemann','Rander')):
			return False
		if self.model in ['ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2']:
			return True
		raise ValueError("Unreconized model")

	def SetKernelTraits(self):
		if self.verbosity>=1: print("Setting the kernel traits.")
		if self.verbosity>=2: print("(Scalar,Int,shape_i,niter_i,...)")	
		traits = kernel_traits.default_traits(self)
		traits.update(self.GetValue('traits',default=traits,
			help="Optional trait parameters passed to kernel."))

		self.multiprecision = (self.GetValue('multiprecision',default=False,
			help="Use multiprecision arithmetic, to improve accuracy") or 
			self.GetValue('values_float64',default=False) )
		if self.multiprecision: 
			traits['multiprecision_macro']=1
			traits['strict_iter_o_macro']=1
			traits['strict_iter_i_macro']=1

		self.factoringRadius = self.GetValue('factoringRadius',default=0,
			help="Use source factorization, to improve accuracy")
		if self.factoringRadius: traits['factor_macro']=1

		order = self.GetValue('order',default=1,
			help="Use second order scheme to improve accuracy")
		if order not in {1,2}: raise ValueError(f"Unsupported scheme order {order}")
		if order==2: traits['order2_macro']=1
		self.order=order

		if not self.isCurvature: # Dimension generic models
			traits['ndim_macro'] = int(self.model[-1])
		if self.HasValue('drift') or self.model.startswith('Rander'):
			traits['drift_macro']=1

		self.bound_active_blocks = self.GetValue('bound_active_blocks',default=False,
			help="Limit the number of active blocks in the front. " 
			"Admissible values : (False,True, or positive integer)")
		if self.bound_active_blocks: 
			traits['minChg_freeze_macro']=1
			traits['pruning_macro']=1

		self.strict_iter_o = traits.get('strict_iter_o_macro',0)
		self.float_t = np.dtype(traits['Scalar']).type
		self.int_t   = np.dtype(traits['Int']   ).type
		self.shape_i = traits['shape_i']

		self.traits = traits

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
		
		if self.HasValue('gridScale'):
			self.h = self.GetValue('gridScale',array_float=True,
				help="Scale of the computational grid")
			# Gridscale for periodic dimension, in the curvature case
			if self.isCurvature: 
				self.h_per = self.caster(2.*np.pi / self.shape[2] )
				self.h = self.caster((self.h,self.h,self.h_per))
		else:
			self.h = self.GetValue('gridScales',array_float=True,
				help="Axis independent scales of the computational grid")

		self.h_broadcasted = np.broadcast_to(self.h,(self.ndim,))
		self.h_broadcasted = np.broadcast_to(np.reshape(self.h_broadcasted,
			(self.ndim,)+(1,)*self.ndim),(self.ndim,)+self.shape)

		self.drift = self.GetValue('drift', default=None, verbosity=3, array_float=True,
			help="Drift introduced in the eikonal equation, becoming F(grad u - drift)=1")

		# Get the metric or cost function
		self.cost_based = self.model.startswith('Isotropic') or self.isCurvature
		if self.cost_based:
			self.metric = self.GetValue('cost',None,verbosity=3,
				help="Cost function for the minimal paths. cost = 1/speed.")
			self.dualMetric = self.GetValue('speed',None,verbosity=3,
				help="Speed function for the minimal paths. speed = 1/cost.")
		else:
			self.metric = self.GetValue('metric',default=None,verbosity=3,
				help="Metric of the minimal path model")
			self.dualMetric = self.GetValue('dualMetric',default=None,verbosity=3,
				help="Dual metric of the minimal path model")

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
				self.costVariation/=self.xp.expand_dims(self.metric.cost,axis=-1)

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
			grid = ad.array(np.meshgrid(*(self.xp.arange(s,dtype=self.float_t) 
				for s in self.shape), indexing='ij')) # Adimensionized coordinates
			self.metric.set_interpolation(grid,periodic=self.periodic) # First order interpolation


		self.block['geom'] = misc.block_expand(fd.as_field(self.geom,self.shape),
			self.shape_i,mode='constant',constant_values=self.xp.inf,contiguous=True)
		if self.drift is not None:
			self.block['drift'] = misc.block_expand(fd.as_field(self.drift,self.shape),
				self.shape_i,mode='constant',constant_values=self.xp.nan,contiguous=True)

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
		xp = self.xp
		values = xp.full(self.shape,xp.inf,dtype=self.float_t)
		self.seeds = self.GetValue('seeds',
			help="Points from where the front propagation starts",array_float=True)
		assert self.seeds.ndim==2 and self.seeds.shape[1]==self.ndim
		self.seeds = Grid.PointFromIndex(self.hfmIn,self.seeds,to=True) # Adimensionize seed position
		if len(self.seeds)==1: self.seed=self.seeds[0]
		seedValues = xp.zeros(len(self.seeds),dtype=self.float_t)
		seedValues = self.GetValue('seedValues',default=seedValues,
			help="Initial value for the front propagation",array_float=True)
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
			aX = [xp.arange(int(np.floor(ci-r)),int(np.ceil(ci+r)+1)) for ci in self.seed]
			neigh =  ad.stack(xp.meshgrid( *aX, indexing='ij'),axis=-1)
			neigh = neigh.reshape(-1,neigh.shape[-1])
			neighValues = seedValues.repeat(len(neigh)//len(self.seeds)) # corrected below

			# Select neighbors which are close enough
			neigh = neigh[ad.Optimization.norm(neigh-self.seed,axis=-1) < r]

			# Periodize, and select neighbors which are in the domain
			nper = np.logical_not(self.periodic)
			inRange = np.all(np.logical_and(-0.5<=neigh[:,nper],
				neigh[:,nper]<xp.array(self.shape)[nper]-0.5),axis=-1)
			neigh = neigh[inRange,:]
			neighValues = neighValues[inRange,:]
			
			diff = (neigh - self.seed).T # Geometry first
#			neigh[:,self.periodic] = neigh[:,self.periodic] % self.shape[self.periodic]
			metric0 = self.Metric(self.seed)
			metric1 = self.Metric(neigh.T)
			self.seedValues = neighValues+0.5*(metric0.norm(diff) + metric1.norm(diff))
			self.seedIndices = neigh
			values[tuple(self.seedIndices.T)] = self.seedValues 

		block_values = misc.block_expand(values,self.shape_i,
			mode='constant',constant_values=np.inf,contiguous=True)

		# Tag the seeds
		if np.prod(self.shape_i)%8!=0:
			raise ValueError('Product of shape_i must be a multiple of 8')
		self.seedTags = xp.isfinite(values)
		block_seedTags = misc.block_expand(self.seedTags,self.shape_i,
			mode='constant',constant_values=True,contiguous=True)
		block_seedTags = misc.packbits(block_seedTags,bitorder='little')
		block_seedTags = block_seedTags.reshape( self.shape_o + (-1,) )
		block_values[xp.isnan(block_values)] = xp.inf

		self.block.update({'values':block_values,'seedTags':block_seedTags})

		# Handle multiprecision
		if self.multiprecision:
			block_valuesq = xp.zeros(block_values.shape,dtype=self.int_t)
			self.block.update({'valuesq':block_valuesq})

		if self.strict_iter_o:
			self.block['valuesNext']=block_values.copy()
			if self.multiprecision:
				self.block['valuesqNext']=block_valuesq.copy()

		if self.bound_active_blocks:
			minChg = xp.full(self.shape_o,np.inf,dtype=self.float_t)
			self.block['minChgPrev_o'] = minChg
			self.block['minChgNext_o'] = minChg.copy()

		for key,value in self.block.items():
			if value.dtype.type not in (self.float_t,self.int_t,np.uint8):
				raise ValueError(f"Inconsistent type {value.dtype.type} for key {key}")

	def SetModuleConstant(self,*args,module="Eikonal",**kwargs):
		if isinstance(module,str):
			if module=="Eikonal": module = (self.solver_module,self.flow_module)
			elif module=="All": module = (self.solver_module,self.flow_module,self.geo_module)
		else: module=(module,)

		for mod in module: cupy_module_helper.SetModuleConstant(mod,*args,**kwargs)

	def SetKernel(self):
		if self.verbosity>=1: print("Preparing the GPU kernel")
		if self.GetValue('dummy_kernel',default=False): return
		# Set a few last traits
		traits = self.traits
		if self.isCurvature:
			traits['xi_var_macro'] = int(not np.isscalar(self.xi))
			traits['kappa_var_macro'] = int(not np.isscalar(self.kappa))
			traits['theta_var_macro'] = int(not np.isscalar(self.theta))
		if self.periodic != self.periodic_default:
			traits['periodic_macro']=1
			traits['periodic_axes']=self.periodic

		self.solver_source = cupy_module_helper.traits_header(self.traits,
			join=True,size_of_shape=True,log2_size=True) + "\n"

		if self.isCurvature: 
			self.model_source = f'#include "{self.model}.h"\n'
		else: 
			model = self.model[:-1]+'_' # Dimension generic
			if model == 'Rander_': model = 'Riemann_' # Rander = Riemann + drift
			self.model_source = f'#include "{model}.h"\n' 

		self.cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = cupy_module_helper.getmtime_max(self.cuda_path)
		self.cuda_date_modified = f"// Date cuda code last modified : {date_modified}\n"
		self.cuoptions = ("-default-device", f"-I {self.cuda_path}"
			) + self.GetValue('cuoptions',default=tuple(),
			help="Options passed via cupy.RawKernel to the cuda compiler")

		source = self.solver_source+self.model_source+self.cuda_date_modified
		self.solver_module = GetModule(source,self.cuoptions)

		# Produce a second kernel for computing the geodesic flow
		flow_traits = traits.copy() 
		flow_traits.update({
			'pruning_macro':0,
			'minChg_freeze_macro':0,
			'niter_i':1,
		})

		self.forward_ad = self.costVariation is not None or ad.is_ad(self.seedValues)
		self.reverse_ad = self.HasValue('sensitivity')
		if self.forward_ad or self.reverse_ad:
			for key in ('flow_weights','flow_weightsum','flow_indices'):
				flow_traits[key+"_macro"]=1
		if self.hasTips: 
			for key in ('flow_vector','flow_weightsum'):
				flow_traits[key+"_macro"]=1

		flow_traits['flow_vector_macro'] = int(
			self.exportGeodesicFlow or (self.tips is not None) or 
			(self.isCurvature and (self.unorientedTips is not None)))

		self.flow_traits = flow_traits
		self.flow_source = cupy_module_helper.traits_header(flow_traits,
			join=True,size_of_shape=True,log2_size=True) + "\n"
		source = self.flow_source+self.model_source+self.cuda_date_modified
		self.flow_module = GetModule(source,self.cuoptions)

		# Set the constants
		float_t,int_t = self.float_t,self.int_t
		self.SetModuleConstant('tol',self.tol,float_t)

		self.size_o = np.prod(self.shape_o)
		self.SetModuleConstant('shape_o',self.shape_o,int_t)
		self.SetModuleConstant('size_o', self.size_o, int_t)

		size_tot = self.size_o * np.prod(self.shape_i)
		self.SetModuleConstant('shape_tot',self.shape,int_t) # Used for periodicity
		self.SetModuleConstant('size_tot', size_tot, int_t) # Used for geom indexing

		if self.multiprecision:
			# Choose power of two, significantly less than h
			h = float(np.min(self.h))
			self.multip_step = 2.**np.floor(np.log2(h/10)) 
			self.SetModuleConstant('multip_step',self.multip_step, float_t)
			self.multip_max = np.iinfo(self.int_t).max*self.multip_step/2
			self.SetModuleConstant('multip_max',self.multip_max, float_t)

		if self.factoringRadius:
			self.SetModuleConstant('factor_radius2',self.factoringRadius**2,float_t)
			self.SetModuleConstant('factor_origin',self.seed,float_t) # Single seed only
			factor_metric = self.Metric(self.seed).to_HFM()
			# The drift part of a Rander metric can be ignored for factorization purposes 
			if self.model.startswith('Rander'): factor_metric = factor_metric[:-self.ndim]
			self.SetModuleConstant('factor_metric',factor_metric,float_t)

		if self.order==2:
			order2_threshold = self.GetValue('order2_threshold',0.3,
				help="Relative threshold on second order differences / first order difference,"
				"beyond which the second order scheme deactivates")
			self.SetModuleConstant('order2_threshold',order2_threshold,float_t)
		
		if self.isCurvature:
			if np.isscalar(self.xi): self.SetModuleConstant('xi',self.xi,float_t)
			if np.isscalar(self.kappa): self.SetModuleConstant('kappa',self.kappa,float_t)

		if self.bound_active_blocks:
			self.minChgPrev_thres = np.inf
			self.minChgNext_thres = np.inf
			SetModuleConstant('minChgPrev_thres',self.minChgPrev_thres,float_t,module=solver_module)
			SetModuleConstant('minChgNext_thres',self.minChgNext_thres,float_t,module=solver_module)

	def Metric(self,x):
		if self.isCurvature: 
			raise ValueError("No metric available for curvature penalized models")
		if hasattr(self,'metric'): return self.metric.at(x)
		else: return self.dualMetric.at(x).dual()

	def KernelArgs(self,solver=True):

		if self.bound_active_blocks:
			self.block['minChgPrev_o'],self.block['minChgNext_o'] \
				=self.block['minChgNext_o'],self.block['minChgPrev_o']

		kernel_argnames = self.solver_kernel_argnames if solver else self.flow_kernel_argnames
		kernel_args = tuple(self.block[key] for key in kernel_argnames)

		if self.strict_iter_o:
			self.block['values'],self.block['valuesNext'] \
				=self.block['valuesNext'],self.block['values']
			if self.multiprecision:
				self.block['valuesq'],self.block['valuesqNext'] \
					=self.block['valuesqNext'],self.block['valuesq']

		return kernel_args

	def SetSolver(self):
		if self.verbosity>=1: print("Setup and run the eikonal solver")
		solver = self.GetValue('solver',default='AGSI',help="Choice of fixed point solver")
		self.nitermax_o = self.GetValue('nitermax_o',default=2000,
			help="Maximum number of iterations of the solver")
		
		kernel_argnames = ['values'] 
		if self.multiprecision: kernel_argnames.append('valuesq')
		if self.strict_iter_o: 
			kernel_argnames.append('valuesNext')
			if self.multiprecision: kernel_argnames.append('valuesqNext')
		kernel_argnames.append('geom')
		if self.drift is not None: kernel_argnames.append('drift')
		kernel_argnames.append('seedTags')

		self.solver_kernel_argnames = kernel_argnames
		self.flow_kernel_argnames = kernel_argnames.copy()

		if self.bound_active_blocks:
			self.solver_kernel_argnames.append('minChgPrev_o')
			self.solver_kernel_argnames.append('minChgNext_o')

		solver_start_time = time.time()
		if solver=='global_iteration':
			niter_o = _solvers.global_iteration(self)
		elif solver in ('AGSI','adaptive_gauss_siedel_iteration'):
			niter_o = _solvers.adaptive_gauss_siedel_iteration(self)
		else:
			raise ValueError(f"Unrecognized solver : {solver}")

		solverGPUTime = time.time() - solver_start_time
		self.hfmOut.update({
			'niter_o':niter_o,
			'solverGPUTime':solverGPUTime,
		})

		self.raiseOnNonConvergence = self.GetValue('raiseOnNonConvergence',default=True)
		if niter_o>=self.nitermax_o:
			nonconv_msg = (f"Solver {solver} did not reach convergence after "
				f"maximum allowed number {niter_o} of iterations")
			if self.raiseOnNonConvergence: raise ValueError(nonconv_msg)
			else: self.Warn(nonconv_msg)
		if self.verbosity>=1: print(f"GPU solve took {solverGPUTime} seconds,"
			f" in {niter_o} iterations.")

	
	PostProcess = _PostProcess.PostProcess
	GetGeodesics = _GetGeodesics.GetGeodesics
	
	def FinalCheck(self):
		self.hfmOut['keys']['unused'] = list(set(self.hfmIn.keys())-set(self.hfmOut['keys']['used']))
		if self.verbosity>=1 and self.hfmOut['keys']['unused']:
			print(f"!! Warning !! Unused keys from user : {self.hfmOut['keys']['unused']}")




























