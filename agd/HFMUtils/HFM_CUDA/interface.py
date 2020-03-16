import numpy as np
import os
import time
from packaging.version import Version

from . import kernel_traits
from . import solvers
from . import misc
from .. import Grid

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


		self.verbosity = hfmIn.get('verbosity',1)
		self.model = self.GetValue('model',help='Minimal path model to be solved')
		self.returns=None
		
	def GetValue(self,key,default=None,verbosity=2,help=None):
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
		elif default is not None:
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

	def SetKernelTraits(self):
		if self.verbosity>=1: print("Setting the kernel traits.")
		if self.verbosity>=2: print("(Scalar,Int,shape_i,niter_i,...)")	
		traits = kernel_traits.default_traits(self.model)
		traits.update(self.GetValue('traits',default=tuple(),
			help="Optional trait parameters passed to kernel"))
		self.traits = traits
		self.in_raw['traits']=traits

		self.float_t = np.dtype(traits['Scalar']).type
		self.int_t   = np.dtype(traits['Int']   ).type
		self.shape_i = traits['shape_i']

	def SetGeometry(self):
		if self.verbosity>=1: print("Prepating the domain data (shape,metric,...)")
		self.shape = self.GetValue('dims',
			help="dimensions (shape) of the computational domain").astype(int)
		self.shape_o = tuple(misc.round_up(self.shape,self.shape_i))
		self.h = self.GetValue('gridScale', help="Scale of the computational grid")
		metric = self.GetValue('cost',help="Cost function for the minimal paths")
		assert metric.dtype == self.float_t
		self.xp = misc.get_array_module(metric)
		self.block['metric'] = misc.block_expand(metric*self.h,self.shape_i,
			mode='constant',constant_values=self.xp.inf)

	def SetValuesArray(self):
		if self.verbosity>=1: print("Preparing the values array (setting seeds,...)")
		xp = self.xp
		values = xp.full(self.shape,xp.inf,dtype=self.float_t)
		seeds = self.GetValue('seeds', help="Points from where the front propagation starts")
		seedValues = xp.zeros(len(seeds),dtype=self.float_t)
		seedValues = self.GetValue('seedValues',default=seedValues,
			help="Initial value for the front propagation")
		seedRadius = self.GetValue('seedRadius',default=0.,
			help="Spreading the seeds over a few pixels can improve accuracy")

		if seedRadius==0.:
			seedIndices,_ = Grid.IndexFromPoint(self.hfmIn,seeds)
			values[tuple(seedIndices.T)] = seedValues
		else: 
			raise ValueError("Positive seedRadius not supported yet")

		block_values = misc.block_expand(values,self.shape_i,mode='constant',constant_values=xp.nan)

		# Tag the seeds
		block_seedTags = xp.isfinite(block_values).reshape( self.shape_o + (-1,) )
		block_seedTags = misc.packbits(block_seedTags,bitorder='little')
		block_values[xp.isnan(block_values)] = xp.inf

		self.block.update({'values':block_values,'seedTags':block_seedTags})

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

		shape_tot = np.array(self.shape_o)*np.array(self.shape_i)
		size_tot = np.prod(shape_tot)
		self.SetModuleConstant('shape_tot',shape_tot,int_t)
		self.SetModuleConstant('size_tot', size_tot, int_t)

		"""
		if not cupy_has_modules: 
			# Support outdated cupy version.
			# We use compile time constants, rather than cuda __constant__. 
			# This is silly, since it requires a recompilation each time the size is changed.
			kernel = cupy.RawKernel(self.source,'Update',options=cuoptions)
			self.kernels = {"Update":kernel}
"""

		in_raw.update({
			'tol':tol,
			'shape_o':self.shape_o,
			'shape_tot':shape_tot,
			'source':self.source
			})

		# TODO : factorization, multiprecision

		"""
	def SetCompileConstant(self,key,value,dtype):
		"Sets a compile time constant in the cupy cuda module source"
		if   dtype == self.float_t: self.source += "const Scalar "
		elif dtype == self.int_t:   self.source += "const Int "
		else: raise ValueError(f"Unrecognized dtype {dtype}")

		if isinstance(value,self.xp.ndarray):
			assert value.ndim==1
			self.source += f"{key}[{len(key)}] = " + "{" + ",".join(str(s) for s in shape_i) + "}"
		else:
			self.source += f"{key} = {value}"
		self.source += ";\n"
		"""

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

		kernel_args = tuple(self.block[key] for key in ('values','metric','seedTags'))

		solver_start_time = time.time()

		if solver=='global_iteration':
			niter_o = solvers.global_iteration(self,kernel_args)
		elif solver in ('AGSI','adaptive_gauss_siedel_iteration'):
			niter_o = solvers.adaptive_gauss_siedel_iteration(self,kernel_args)
		else:
			raise ValueError(f"Unrecognized solver : {solver}")

		solverGPUTime = time.time() - solver_start_time
		hfmOut.update({
			'niter_o':niter_o,
			'solverGPUTime':solverGPUTime,
		})
		out_raw.update({
			'block_values':block_values
		})

		if niter_o>=nitermax_o:
			nonconv_msg = (f"Solver {solver} did not reach convergence after "
				f"maximum allowed number {niter_o} of iterations")
			if self.GetValue('raiseOnNonConvergence',default=True):
				raise ValueError(nonconv_msg)
			else:
				print("---- Warning ----\n",nonconv_msg,"\n-----------------\n")

		if self.verbosity>=1: print("GPU solve took {solverGPUTime} seconds")

	def PostProcess(self):
		if self.verbosity>=1: print("Post-Processing")
		if returns=='out_raw': return {'out_raw':out_raw,'in_raw':in_raw,'hfmOut':hfmOut}
		values = misc.block_squeeze(block_values,shape)
		hfmOut['values'] = values
		return hfmOut



