# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements some member functions of the Interface class.
"""

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
	self.size_i = np.prod(self.shape_i)

	self.traits['eikonal']=traits

	self.caster = lambda x : cp.asarray(x,dtype=self.float_t)

def SetKernel(self):
	if self.verbosity>=1: print("Preparing the GPU kernel")
	if self.GetValue('dummy_kernel',default=False): return
	# Set a few last traits
	traits = self.traits['eikonal']
	if self.isCurvature:
		traits['xi_var_macro'] = int(not np.isscalar(self.xi))
		traits['kappa_var_macro'] = int(not np.isscalar(self.kappa))
		traits['theta_var_macro'] = int(not np.isscalar(self.theta))
	if self.periodic != self.periodic_default:
		traits['periodic_macro']=1
		traits['periodic_axes']=self.periodic

	self.source['eikonal'] = cupy_module_helper.traits_header(traits,
		join=True,size_of_shape=True,log2_size=True) + "\n"

	if self.isCurvature: 
		self.source['model'] = f'#include "{self.model}.h"\n'
	else: 
		model = self.model[:-1]+'_' # Dimension generic
		if model == 'Rander_': model = 'Riemann_' # Rander = Riemann + drift
		self.source['model'] = f'#include "{model}.h"\n' 

	self.cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(self.cuda_path)
	self.cuda_date_modified = f"// Date cuda code last modified : {date_modified}\n"
	self.cuoptions = ("-default-device", f"-I {self.cuda_path}"
		) + self.GetValue('cuoptions',default=tuple(),
		help="Options passed via cupy.RawKernel to the cuda compiler")

	source = self.source['eikonal']+self.source['model']+self.cuda_date_modified
	self.module['eikonal'] = GetModule(source,self.cuoptions)

	# ---- Produce a second kernel for computing the geodesic flow ---
	self.traits['flow'] = self.traits['eikonal'].copy()
	traits = self.traits['flow']
	traits.update({
		'pruning_macro':0,
		'minChg_freeze_macro':0,
		'niter_i':1,
	})

	self.forward_ad = self.costVariation is not None or ad.is_ad(self.seedValues)
	self.reverse_ad = self.HasValue('sensitivity')
	if self.forward_ad or self.reverse_ad:
		for key in ('flow_weights','flow_weightsum','flow_indices'): 
			traits[key+"_macro"]=1
	if self.hasTips: 
		for key in ('flow_vector','flow_weightsum'): 
			traits[key+"_macro"]=1

	traits['flow_vector_macro'] = int(
		self.exportGeodesicFlow or (self.tips is not None) or 
		(self.isCurvature and (self.unorientedTips is not None)))

	self.source['flow'] = cupy_module_helper.traits_header(traits,
		join=True,size_of_shape=True,log2_size=True) + "\n"
	source = self.source['flow']+self.source['model']+self.cuda_date_modified
	self.module['flow'] = GetModule(source,self.cuoptions)

	# Set the constants
	def SetCst(*args):
		for kernelName in ('eikonal','flow'):
			SetModuleConstant(self.module['kernelName'],*args)

	float_t,int_t = self.float_t,self.int_t		
	SetCst('tol',self.tol,float_t)

	self.size_o = np.prod(self.shape_o)
	SetCst('shape_o',self.shape_o,int_t)
	SetCst('size_o', self.size_o, int_t)

	size_tot = self.size_o * np.prod(self.shape_i)
	SetCst('shape_tot',self.shape,int_t) # Used for periodicity
	SetCst('size_tot', size_tot,  int_t) # Used for geom indexing

	if self.multiprecision:
		# Choose power of two, significantly less than h
		h = float(np.min(self.h))
		self.multip_step = 2.**np.floor(np.log2(h/10)) 
		SetCst('multip_step',self.multip_step, float_t)
		self.multip_max = np.iinfo(self.int_t).max*self.multip_step/2
		SetCst('multip_max', self.multip_max,  float_t)

	if self.factoringRadius:
		SetCst('factor_radius2',self.factoringRadius**2,float_t)
		SetCst('factor_origin', self.seed,              float_t) # Single seed only
		factor_metric = self.Metric(self.seed).to_HFM()
		# The drift part of a Rander metric can be ignored for factorization purposes 
		if self.model.startswith('Rander'): factor_metric = factor_metric[:-self.ndim]
		SetCst('factor_metric',factor_metric,float_t)

	if self.order==2:
		order2_threshold = self.GetValue('order2_threshold',0.3,
			help="Relative threshold on second order differences / first order difference,"
			"beyond which the second order scheme deactivates")
		SetCst('order2_threshold',order2_threshold,float_t)
	
	if self.isCurvature:
		if np.isscalar(self.xi):    SetCst('xi',   self.xi,   float_t)
		if np.isscalar(self.kappa): SetCst('kappa',self.kappa,float_t)

	if self.bound_active_blocks:
		self.minChgPrev_thres = np.inf
		self.minChgNext_thres = np.inf
		mod = self.module['eikonal']
		SetModuleConstant(mod,'minChgPrev_thres',self.minChgPrev_thres,float_t)
		SetModuleConstant(mod,'minChgNext_thres',self.minChgNext_thres,float_t)

	# Set the kernel arguments
	self.solver = self.GetValue('solver',default='AGSI',
		help="Choice of fixed point solver (AGSI, global_iteration)")
	self.nitermax_o = self.GetValue('nitermax_o',default=2000,
		help="Maximum number of iterations of the solver")
	self.raiseOnNonConvergence = self.GetValue('raiseOnNonConvergence',default=True,
		help="Raise an exception of a solver fails to converge")

	
	kernel_argnames = ['values'] 
	if self.multiprecision: kernel_argnames.append('valuesq')
	if self.strict_iter_o: 
		kernel_argnames.append('valuesNext')
		if self.multiprecision: kernel_argnames.append('valuesqNext')
	kernel_argnames.append('geom')
	if self.drift is not None: kernel_argnames.append('drift')
	kernel_argnames.append('seedTags')

	self.kernel_argnames['flow'] = kernel_argnames.copy()

	if self.bound_active_blocks: kernel_argnames += ['minChgPrev_o','minChgNext_o']
	self.kernel_argnames['eikonal'] = kernel_argnames


def KernelArgs(self,kernelName):

	if self.bound_active_blocks:
		self.block['minChgPrev_o'],self.block['minChgNext_o'] \
			=self.block['minChgNext_o'],self.block['minChgPrev_o']

	kernel_argnames = self.kernel_argnames[kernelName]
	kernel_args = tuple(self.block[key] for key in kernel_argnames)

	if self.strict_iter_o and kernelName=='eikonal':
		self.block['values'],self.block['valuesNext'] \
			=self.block['valuesNext'],self.block['values']
		if self.multiprecision:
			self.block['valuesq'],self.block['valuesqNext'] \
				=self.block['valuesqNext'],self.block['valuesq']

	return kernel_args		self.solver = self.GetValue('solver',default='AGSI',help="Choice of fixed point solver")
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

		self.kernel_argnames['flow'] = kernel_argnames.copy()

		if self.bound_active_blocks: kernel_argnames += ['minChgPrev_o','minChgNext_o']
		self.kernel_argnames['eikonal'] = kernel_argnames

