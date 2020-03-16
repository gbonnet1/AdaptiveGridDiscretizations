import kernel_traits




class Interface(object):
	"""
	This class carries out the RunGPU function work. 
	It should not be used directly.
	"""


	def __init__(hfmIn):

		self.hfmIn = hfmIn
		if hfmIn['arrayOrdering'] != 'RowMajor':
			raise ValueError("Only RowMajor indexing supported")

		self.verbosity = 1
		self.verbosity = misc.GetValue('verbosity',default=1,
			help="Choose the amount of detail displayed on the run")

		self.hfmOut = {
		'key used':[],
		'key defaulted':[],
		'key visited':[],
		'help content':{},
		}

		self.help = hfmIn.get('help',[])
		self.verbosity = hfmIn.get('verbosity',1)
		self.model = misc.GetValue(hfmIn,'model',hfmOut,
			help='Minimal path model to be solved')
		self.returns=None
		
	@property
	def help_content(self): 
		return self.hfmOut['help content']
	

	def GetValue(key,default=None,verbosity=2,help=None):
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
			return dico[key]
		elif default is not None:
			self.hfmOut['key defaulted'].append((key,default))
			if verbosity>=self.verbosity:
				print(f"key {key} defaults to {default}")
			return default
		else:
			raise ValueError(f"Missing value for key {key}")


	def Run(returns='out'):
		assert returns in ('in_raw','out_raw','out')
		self.returns=returns
		self.block={}
		self.in_raw = {'block':self.block}

		self.SetKernelTraits()
		self.SetGeometry()
		self.SetValueArray()
		self.SetKernel()
		self.SetSolver()
		self.PostProcess()

		return self.hfmOut

	def SetKernelTraits():
		if self.verbosity>=1: print("Setting the kernel traits.")
		if self.verbosity>=2: print("(Scalar,Int,shape_i,niter_i,...)")	
		self.traits = kernel_traits.default_traits(model)
		self.traits.update(self.GetValue('traits',default=tuple(),
			help="Optional trait parameters passed to kernel"))
		hfmOut['traits']=self.traits

		self.float_t = np.dtype(traits['Scalar']).type
		self.int_t   = np.dtype(traits['Int']   ).type
		self.shape_i = traits['shape_i']

	def SetGeometry():
		if verbosity>=1: print("Prepating the domain data (shape,metric,...)")
		self.shape = self.GetValue('dims',
			help="dimensions (shape) of the computational domain").astype(int)
		self.shape_o = tuple(misc.round_up(shape,shape_i))
		self.h = self.GetValue('gridScale', help="Scale of the computational grid")
		metric = self.GetValue('cost',help="Cost function for the minimal paths")
		assert self.metric.dtype == float_t
		self.xp = misc.get_array_module(metric)
		self.block['metric'] = misc.block_expand(metric*h,shape_i,
			mode='constant',constant_values=xp.inf)

	def SetValuesArray():
		if verbosity>=1: print("Preparing the values array (setting seeds,...)")
		values = xp.full(shape,xp.inf,dtype=float_t)
		seeds = self.GetValue('seeds', help="Points from where the front propagation starts")
		seedValues = self.xp.zeros(len(seeds),dtype=float_t)
		seedValues = self.GetValue('seedValues',default=seedValues,
			help="Initial value for the front propagation")
		seedRadius = self.GetValue('seedRadius',default=0.,
			help="Spreading the seeds over a few pixels can improve accuracy")

		if seedRadius==0.:
			seedIndices,_ = Grid.IndexFromPoint(hfmIn,seeds)
			values[tuple(seedIndices.T)] = seedValues
		else: 
			raise ValueError("Positive seedRadius not supported yet")

		block_values = misc.block_expand(values,shape_i,mode='constant',constant_values=xp.nan)

		# Tag the seeds
		block_seedTags = xp.isfinite(block_values).reshape( shape_o + (-1,) )
		block_seedTags = misc.packbits(block_seedTags,bitorder='little')
		block_values[xp.isnan(block_values)] = xp.inf

		self.block.update({'values':block_values,'seedTags':block_seedTags})

	def SetKernel():
		if verbosity>=1: print("Preparing the GPU kernel")
		if self.GetValue('dummy_kernel',default=False): return
		in_raw = self.in_raw
		source = kernel_traits.kernel_source(model,traits)
		in_raw['source']=source

		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = max(os.path.getmtime(os.path.join(cuda_path,file)) 
			for file in os.listdir(cuda_path))
		source += f"// Date cuda code last modified : {date_modified}\n"
		cuoptions = ("-default-device", f"-I {cuda_path}"
			) + self.GetValue('cuoptions',default=tuple(),
			help="Options passed via cupy.RawKernel to the cuda compiler")
		import cupy
		self.kernel = cupy.RawModule(source,options=cuoptions)
		in_raw['kernel']=kernel

		tol = self.float_t(misc.GetValue(hfmIn,'tol',hfmOut,1e-8,
			help="Convergence tolerance for the fixed point solver"))

		float_t,int_t = self.float_t,self.int_t
		SetKernelConstant('tol',tol,float_t)
		SetKernelConstant('shape_o',self.shape_o,int_t)
		SetKernelConstant('size_o',xp.prod(self.size_o,int_t),int_t)

		shape_tot = self.shape_o*self.shape_i
		SetKernelConstant('shape_tot',shape_tot,int_t)
		SetKernelConstant('size_tot',xp.prod(shape_tot),int_t)

		# TODO : factorization, multiprecision

		in_raw.update({'tol':tol,'shape_o':shape_o,'shape_tot':shape_tot})

	def SetKernelConstant(key,value,dtype):
		xp = self.xp
		value=xp.array(value,dtype=dtype)
		memptr = self.kernel.get_global(key)
		# ...wrap it using cupy.ndarray with a known shape
		arr_ndarray = xp.ndarray(value.shape, value.dtype, memptr)
		# ...perform data transfer to initialize it
		arr_ndarray[...] = value


	def SetSolver():
		if verbosity>=1: print("Setup and run the eikonal solver")
		self.solver = self.GetValue('solver',help="Choice of fixed point solver")
		self.nitermax_o = self.GetValue('nitermax_o',default=2000,
			help="Maximum number of iterations of the solver")

		if returns=='in_raw': return {'in_raw':in_raw,'hfmOut':hfmOut}

		data_t = (float_t,int_t)
		shapes_io = (shape_i,shape_o)
		kernel_args = (block_values,block_metric,block_seedTags,tol)
		kernel_args = tuple(arg if isinstance(arg,xp.ndarray) else 
		xp.array(arg,kernel_traits.dtype(arg,data_t)) for arg in kernel_args)



