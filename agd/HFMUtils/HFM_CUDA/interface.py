import numpy as np
import os
import time
from packaging.version import Version
import copy

from . import kernel_traits
from . import solvers
from . import misc
from ... import HFMUtils
from ... import FiniteDifferences as fd
from ... import AutomaticDifferentiation as ad
from ... import Metrics

class Interface(object):
	"""
	This class carries out the RunGPU function work. 
	It should not be used directly.
	"""
	def __init__(self,hfmIn):

		self.hfmIn = HFMUtils.dictIn(hfmIn)
		if hfmIn['arrayOrdering'] != 'RowMajor':
			raise ValueError("Only RowMajor indexing supported")

		# Needed for GetValue
		self.help = hfmIn.get('help',[])
		self.hfmOut = {
		'key used':[],
		'key defaulted':[],
		'key visited':[],
		'help content':{},
		}
		self.help_content = self.hfmOut['help content']

		self.verbosity = 1
		self.verbosity = self.GetValue('verbosity',default=1,
			help="Choose the amount of detail displayed on the run")
		
		self.model = self.GetValue('model',help='Minimal path model to be solved.')
		# Unified treatment of standard and extended curvature models
		if self.model.endswith("Ext2"): self.model=self.model[:-4]+"2"
		
		self.returns=None
		self.xp = ad.cupy_generic.get_array_module(**hfmIn)
		
	def HasValue(self,key):
		self.hfmOut['key visited'].append(key)
		return key in self.hfmIn

	def GetValue(self,key,default="_None",verbosity=2,help=None):
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
					print("-----------------------------")

		if key in self.hfmIn:
			self.hfmOut['key used'].append(key)
			return self.hfmIn[key]
		elif default != "_None":
			self.hfmOut['key defaulted'].append((key,default))
			if verbosity>=self.verbosity:
				print(f"key {key} defaults to {default}")
			return default
		else:
			raise ValueError(f"Missing value for key {key}")


	def Run(self,returns='out'):
		assert returns in ('in_raw','out_raw','out')
		self.returns=returns
		self.block={}
		self.in_raw = {'block':self.block}
		self.out_raw = {}

		self.SetKernelTraits()
		self.SetGeometry()
		self.SetValuesArray()
		self.SetKernel()
		self.SetSolver()
		self.PostProcess()

		return self.hfmOut

	@property
	def isCurvature(self):
		if any(self.model.startswith(e) for e in ('Isotropic','Riemann','Rander')):
			return True
		if self.model in ['ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2']:
			return False
		raise ValueError("Unreconized model")

	def SetKernelTraits(self):
		if self.verbosity>=1: print("Setting the kernel traits.")
		if self.verbosity>=2: print("(Scalar,Int,shape_i,niter_i,...)")	
		traits = kernel_traits.default_traits(self.model)
		traits.update(self.GetValue('traits',default=tuple(),
			help="Optional trait parameters passed to kernel"))

		self.multiprecision = (self.GetValue('multiprecision',default=False,
			help="Use multiprecision arithmetic, to tmprove accuracy") or 
			self.GetValue('values_float64',default=False) )
		if self.multiprecision: traits['multiprecision_macro']=1

		self.factoringRadius = self.GetValue('factoringRadius',default=0,
			help="Use source factorization, to improve accuracy")
		if self.factoringRadius: traits['factor_macro']=1

		order = self.GetValue('order',default=1,
			help="Use second order scheme to improve accuracy")
		if order not in {1,2}: raise ValueError(f"Unsupported scheme order {order}")
		if order==2: traits['order2_macro']=1
		self.order=order

		if not self.isCurvature:
			traits['ndim_macro'] = int(self.model[-1])

		self.periodic = self.GetValue('periodic',default=None,
			help="Apply periodic boundary conditions on some axes")
		if self.periodic is not None:
			traits['periodic_macro']=1
			traits['periodic']=periodic

		self.traits = traits
		self.in_raw['traits']=traits

		self.float_t = np.dtype(traits['Scalar']).type
		self.int_t   = np.dtype(traits['Int']   ).type
		self.shape_i = traits['shape_i']

	def SetGeometry(self):
		if self.verbosity>=1: print("Prepating the domain data (shape,metric,...)")

		# Domain shape and grid scale
		self.shape = self.GetValue('dims',
			help="dimensions (shape) of the computational domain").astype(int)
		self.shape_o = tuple(misc.round_up(self.shape,self.shape_i))

		if self.HasValue('gridScale'):
			self.h = self.GetValue('gridScale', default=None,
				help="Scale of the computational grid")
		else:
			self.h = self.GetValue('gridScales', default=None,
				help="Axis independent scales of the computational grid")
		self.drift = self.GetValue('drift', default=None, verbosity=3,
			help="Drift introduced in the eikonal equation, becoming F(grad u - drift)=1")

		# Get the metric or cost function
		if self.model.startswith('Isotropic') or self.isCurvature:
			self.metric = self.GetValue('cost',None,verbosity=3,
				help="Cost function for the minimal paths")
			self.dualMetric = self.GetValue('speed',None,verbosity=3,
				help="Speed function for the minimal paths")
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

		if ad.cupy_generic.isndarray(self.metric): 
			self.metric = metricClass.from_HFM(self.metric)
		elif not overwriteMetric:
			self.metric = copy.deepcopy(self.metric)
		if ad.cupy_generic.isndarray(self.dualMetric): 
			self.dualMetric = metricClass.from_HFM(self.dualMetric)
		elif not overwriteMetric:
			self.dualMetric = copy.deepcopy(self.dualMetric)

		# Rescale 
		if self.metric is not None: self.metric.rescale(1/self.h)
		if self.dualMetric: is not None: self.dualMetric.rescale(self.h)

		if self.isCurvature:
			if self.metric is None: self.metric = self.dualMetric.dual()
			self.xi = self.GetValue('xi',
				help="Cost of rotation for the curvature penalized models")
			self.kappa = self.GetValue('kappa',default=0.,
				help="Rotation bias for the curvature penalized models")
			self.theta = self.GetValue('theta',default=0.,verbosity=3,
				help="Deviation from horizontality, for the curvature penalized models")

			self.h_per = 2.*np.pi / self.shape[2] # Gridscale for periodic dimension
			self.xi *= self.h_per
			self.kappa /= self.h_per

			self.geom = ad.array([e for e in (self.metric,self.xi,self.kappa,self.theta)
				if not np.isscalar(e)])
			self.periodic = (False,False,True)

		else: # not self.isCurvature
			if self.model.startswith('Isotropic'):
				if self.metric is None: self.metric = self.dualMetric.dual()
				self.geom = self.metric.cost
			elif self.model.startswith('Riemann'):
				if self.dualMetric is None: self.dualMetric = self.metric.dual()
				self.geom = self.dualMetric.flatten()
			elif self.model.startswith('Rander'):
				if self.metric is None: self.metric = self.dualMetric.dual()
				self.geom = Metrics.Riemann(self.metric.m).dual().flatten()
				self.drift = self.metric.w

			# TODO : remove. No need to create this grid for our interpolation
			grid = self.xp.meshgrid(*(range(s) for s in self.shape), 
				indexing='ij',dtype=self.float_t) # Adimensionized coordinates
			self.metric.set_interpolation(grid,periodic=self.periodic) # First order interpolation

		self.block['geom'] = misc.block_expand(self.geom,self.shape_i,
			mode='constant',constant_values=self.xp.inf)
		if self.drift is not None:
			self.block['drift'] = misc.block_expand(self.drift,self.shape_i,
				mode='constant',constant_values=self.xp.nan)

	def SetValuesArray(self):
		if self.verbosity>=1: print("Preparing the values array (setting seeds,...)")
		xp = self.xp
		values = xp.full(self.shape,xp.inf,dtype=self.float_t)
		self.seeds = xp.array(self.GetValue('seeds', 
			help="Points from where the front propagation starts") )
		self.seeds = self.hfmIn.PointFromIndex(seeds,to=True) # Adimensionize seed position
		if len(self.seeds)==1: self.seed=self.seeds[0]
		seedValues = xp.zeros(len(self.seeds),dtype=self.float_t)
		seedValues = xp.array(self.GetValue('seedValues',default=seedValues,
			help="Initial value for the front propagation"))
		seedRadius = self.GetValue('seedRadius',default=0.,
			help="Spread the seeds over a radius given in pixels, so as to improve accuracy.")

		if seedRadius==0.:
			seedIndices = np.round(self.seeds).astype(int)
			values[tuple(seedIndices.T)] = seedValues
		else:
			neigh = self.hfmIn.GridNeighbors(self.seed,seedRadius) # Geometry last
			r = seedRadius 
			aX = [range(int(np.floor(ci-r)),int(np.ceil(ci+r)+1)) for ci in self.seed]
			neigh =  ad.stack(xp.meshgrid( *aX, indexing='ij'),axis=-1)
			neigh = neigh_index.reshape(-1,neigh_index.shape[-1])

			# Select neighbors which are close enough
			neigh = neigh[ad.Optimization.norm(neigh-self.seed,axis=-1) < r]

			# Periodize, and select neighbors which are in the domain
			nper = np.logical_not(self.periodic)
			inRange = np.all(np.logical_and(-0.5<=neigh[:,nper],
				neigh[:,nper]<self.shape[nper]-0.5),axis=-1)
			neigh = neigh[:,inRange]
			
			diff = (neigh - self.seed).T # Geometry first
#			neigh[:,self.periodic] = neigh[:,self.periodic] % self.shape[self.periodic]
			metric0 = self.Metric(self.seed)
			metric1 = self.Metric(neigh.T)
			values[tuple(neigh.T)] = 0.5*(metric0.norm(diff) + metric1.norm(diff))

		block_values = misc.block_expand(values,self.shape_i,mode='constant',constant_values=xp.nan)

		# Tag the seeds
		block_seedTags = xp.isfinite(block_values).reshape( self.shape_o + (-1,) )
		block_seedTags = misc.packbits(block_seedTags,bitorder='little')
		block_values[xp.isnan(block_values)] = xp.inf

		self.block.update({'values':block_values,'seedTags':block_seedTags})

		# Handle multiprecision
		if self.multiprecision:
			block_valuesq = xp.zeros(block_values.shape,dtype=self.int_t)
			self.block.update({'valuesq':block_valuesq})

	def SetKernel(self):
		if self.verbosity>=1: print("Preparing the GPU kernel")
		if self.GetValue('dummy_kernel',default=False): return
		in_raw = self.in_raw
		self.source = kernel_traits.kernel_source(self.model,self.traits)

		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = max(os.path.getmtime(os.path.join(cuda_path,file)) 
			for file in os.listdir(cuda_path))
		self.source += f"// Date cuda code last modified : {date_modified}\n"
		cuoptions = ("-default-device", f"-I {cuda_path}"
			) + self.GetValue('cuoptions',default=tuple(),
			help="Options passed via cupy.RawKernel to the cuda compiler")

		import cupy
		self.cupy_old = True # or Version(cupy.__version__) < Version("8") # Untested
		if self.cupy_old:
			self.module = cupy.core.core.compile_with_cache(self.source, 
				options=cuoptions, prepend_cupy_headers=False)
		else:
			self.module = cupy.RawModule(source,options=cuoptions)

		float_t,int_t = self.float_t,self.int_t
		tol = self.float_t(self.GetValue('tol',1e-8,
			help="Convergence tolerance for the fixed point solver"))
		self.SetModuleConstant('tol',tol,float_t)

		self.size_o = np.prod(self.shape_o)
		self.SetModuleConstant('shape_o',self.shape_o,int_t)
		self.SetModuleConstant('size_o', self.size_o, int_t)

		size_tot = self.size_o * np.prod(self.shape_i)
		self.SetModuleConstant('shape_tot',self.shape,int_t) # Used for periodicity
		self.SetModuleConstant('size_tot', size_tot, int_t) # Used for geom indexing

		if self.multiprecision:
			# Choose power of two, significantly less than h
			multip_step = 2.**np.floor(np.log2(self.h/50)) 
			self.SetModuleConstant('multip_step',multip_step, float_t)
			multip_max = np.iinfo(self.int_t).max*multip_step/2
			self.SetModuleConstant('multip_max',multip_max, float_t)

			self.multip_step=multip_step
			in_raw.update({'multip_step':multip_step,'multip_max':multip_max})

		if self.factoringRadius:
			self.SetModuleConstant('factor_radius2',self.factoringRadius**2,float_t)
			self.SetModuleConstant('factor_origin',self.seed,float_t) # Single seed only
			self.SetModuleConstant('factor_metric',self.Metric(self.seed).to_HFM(),float_t)

		if self.order==2:
			order2_threshold = self.GetValue('order2_threshold',0.2,
				help="Relative threshold on second order differences / first order difference,"
				"beyond which the second order scheme deactivates")
			self.SetModuleConstant('order2_threshold',order2_threshold,float_t)
		
		if self.isCurvature:
			is np.isscalar(self.xi): self.SetModuleConstant('xi',self.xi,float_t)
			if np.isscalar(self.kappa): self.SetModuleConstant('kappa',self.kappa,float_t)

		in_raw.update({
			'tol':tol,
			'shape_o':self.shape_o,
			'shape_tot':shape_tot,
			'source':self.source,
			})

	def Metric(self,x):
		if self.isCurvature: 
			raise ValueError("No metric available for curvature penalized models")
		if hasattr(self,'metric'): return self.metric.at(x)
		else: return self.dualMetric.at(x).dual()

	def SetModuleConstant(self,key,value,dtype):
		"""Sets a global constant in the cupy cuda module"""
		xp = self.xp
		value=xp.array(value,dtype=dtype)
		if self.cupy_old:
			#https://github.com/cupy/cupy/issues/1703
			b = xp.core.core.memory_module.BaseMemory()
			b.ptr = self.module.get_global_var(key)
			memptr = xp.cuda.MemoryPointer(b,0)
		else: 
			memptr = self.kernel.get_global(key)
		module_constant = xp.ndarray(value.shape, value.dtype, memptr)
		module_constant[...] = value


	def SetSolver(self):
		if self.verbosity>=1: print("Setup and run the eikonal solver")
		solver = self.GetValue('solver',help="Choice of fixed point solver")
		self.nitermax_o = self.GetValue('nitermax_o',default=2000,
			help="Maximum number of iterations of the solver")

		if self.returns=='in_raw': return {'in_raw':in_raw,'hfmOut':hfmOut}

		kernel_argnames = ['values','geom','seedTags']
		if self.multiprecision:  kernel_argnames.insert(1,'valuesq')
		if self.drift is not None: kernel_argnames.insert(-1,'drift')
		kernel_args = tuple(self.block[key] for key in kernel_argnames)

		solver_start_time = time.time()

		if solver=='global_iteration':
			niter_o = solvers.global_iteration(self,kernel_args)
		elif solver in ('AGSI','adaptive_gauss_siedel_iteration'):
			niter_o = solvers.adaptive_gauss_siedel_iteration(self,kernel_args)
		else:
			raise ValueError(f"Unrecognized solver : {solver}")

		solverGPUTime = time.time() - solver_start_time
		self.hfmOut.update({
			'niter_o':niter_o,
			'solverGPUTime':solverGPUTime,
		})

		if niter_o>=self.nitermax_o:
			nonconv_msg = (f"Solver {solver} did not reach convergence after "
				f"maximum allowed number {niter_o} of iterations")
			if self.GetValue('raiseOnNonConvergence',default=True):
				raise ValueError(nonconv_msg)
			else:
				print("---- Warning ----\n",nonconv_msg,"\n-----------------\n")

		if self.verbosity>=1: print(f"GPU solve took {solverGPUTime} seconds")

	def PostProcess(self):
		if self.verbosity>=1: print("Post-Processing")
		if self.returns=='out_raw': return {'out_raw':out_raw,'in_raw':in_raw,'hfmOut':hfmOut}
		print(self.shape,self.block['values'].shape)
		values = misc.block_squeeze(self.block['values'],self.shape)
		if self.multiprecision:
			valuesq = misc.block_squeeze(self.block['valuesq'],self.shape)
			if self.GetValue('values_float64',default=False,
				help="Export values using the float64 data type"):
				float64_t = np.dtype('float64').type
				self.hfmOut['values'] = (self.xp.array(values,dtype=float64_t) 
					+ float64_t(self.multip_step) * valuesq)
			else:
				self.hfmOut['values'] = (values 
					+ self.xp.array(valuesq,dtype=self.float_t)*self.multip_step)
		else:
			self.hfmOut['values'] = values
		return self.hfmOut



