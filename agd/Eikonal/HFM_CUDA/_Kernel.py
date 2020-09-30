# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import os
from collections import OrderedDict
import copy


from . import kernel_traits
from .cupy_module_helper import SetModuleConstant,GetModule
from . import cupy_module_helper
from ... import AutomaticDifferentiation as ad

"""
This file implements some member functions of the Interface class, related with the 
eikonal cuda kernel.
"""

def SetKernelTraits(self):
	"""
	Set the traits of the eikonal kernel.
	"""
	if self.verbosity>=1: print("Setting the kernel traits.")
	eikonal = self.kernel_data['eikonal']
	policy = eikonal.policy

	traits = kernel_traits.default_traits(self)
	traits.update(self.GetValue('traits',default=traits,
		help="Optional trait parameters for the eikonal kernel."))
	eikonal.traits = traits


	policy.multiprecision = (self.GetValue('multiprecision',default=False,
		help="Use multiprecision arithmetic, to improve accuracy") or 
		self.GetValue('values_float64',default=False) )
	if policy.multiprecision: 
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
	if self.model.startswith('Rander'):
		traits['drift_macro']=1

	policy.bound_active_blocks = self.GetValue('bound_active_blocks',default=False,
		help="Limit the number of active blocks in the front. " 
		"Admissible values : (False,True, or positive integer)")
	if policy.bound_active_blocks:
		traits['minChg_freeze_macro']=1
		traits['pruning_macro']=1

	policy.solver = self.GetValue('solver',default='AGSI',
		help="Choice of fixed point solver (AGSI, global_iteration)")
	if policy.solver=='global_iteration' and traits.get('pruning_macro',0):
		raise ValueError("Incompatible options found for global_iteration solver "
			"(bound_active_blocks, pruning)")

	policy.strict_iter_o = traits.get('strict_iter_o_macro',0)
	self.float_t  = np.dtype(traits['Scalar'] ).type
	self.int_t    = np.dtype(traits['Int']    ).type
	self.offset_t = np.dtype(traits['OffsetT']).type
	self.shape_i = traits['shape_i']
	self.size_i = np.prod(self.shape_i)
	self.caster = lambda x : cp.asarray(x,dtype=self.float_t)
	self.nscheme = kernel_traits.nscheme(self)
#	assert self.float_t == self.hfmIn.float_t # Not necessary for gpu_transfer

def SetKernel(self):
	"""
	Setup the eikonal kernel, and (partly) the flow kernel
	"""
	if self.verbosity>=1: print("Preparing the GPU kernel")
	modules = []

	# ---- Produce a first kernel, for solving the eikonal equation ----
	# Set a few last traits
	eikonal = self.kernel_data['eikonal']
	policy = eikonal.policy
	traits = eikonal.traits
	traits['import_scheme_macro'] = self.precompute_scheme
	if self.periodic != self.periodic_default:
		traits['periodic_macro']=1
		traits['periodic_axes']=self.periodic
	if self.model_=='Isotropic': traits['isotropic_macro']=1
	if 'wallDist' in eikonal.args: traits['walls_macro']=1
	policy.count_updates = self.GetValue('count_updates',default=False,
		help='Count the number of times each block is updated')

	integral_max = policy.multiprecision # Int_Max needed for multiprecision to avoid overflow
	eikonal.source = cupy_module_helper.traits_header(traits,
		join=True,size_of_shape=True,log2_size=True,integral_max=integral_max)+"\n"

	if self.isCurvature: 
		model_source = f'#include "{self.model}.h"\n'
	else: 
		model = self.model_ # Dimension generic
		if model == 'Diagonal': model = 'Isotropic' # Same file handles both
		elif   model in ('Rander','SubRiemann'): model = 'Riemann' # Rander = Riemann + drift
		model_source = f'#include "{model}_.h"\n' 
	
	self.cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(self.cuda_path)
	self.cuda_date_modified = f"// Date cuda code last modified : {date_modified}\n"
	self.cuoptions = ("-default-device", f"-I {self.cuda_path}",
		) + self.GetValue('cuoptions',default=tuple(),
		help="Options passed via cupy.RawKernel to the cuda compiler")

	eikonal.source += model_source+self.cuda_date_modified
	eikonal.module = GetModule(eikonal.source,self.cuoptions)
	modules.append(eikonal.module)

	# ---- Produce a second kernel for computing the geodesic flow ---
	flow = self.kernel_data['flow']
	flow.traits = {
		**eikonal.traits,
		'pruning_macro':0,
		'minChg_freeze_macro':0,
		'niter_i':1,
	}
	flow.policy = copy.copy(eikonal.policy) 
	flow.policy.nitermax_o = 1
	flow.policy.solver = 'global_iteration'

	if self.forwardAD or self.reverseAD:
		for key in ('flow_weights','flow_weightsum','flow_indices'): 
			flow.traits[key+"_macro"]=1
	if self.hasTips: 
		for key in ('flow_vector','flow_weightsum'): 
			flow.traits[key+"_macro"]=1
	if self.exportGeodesicFlow: flow.traits['flow_vector_macro']=1
	if self.model_=='Rander' and (self.forwardAD or self.reverseAD): 
		flow.traits['flow_vector_macro']=1

	flow.source = cupy_module_helper.traits_header(flow.traits,
		join=True,size_of_shape=True,log2_size=True,integral_max=integral_max) + "\n"
	flow.source += model_source+self.cuda_date_modified
	flow.module = GetModule(flow.source,self.cuoptions)
	modules.append(flow.module)

	# ---- Produce a third kernel for precomputing the stencils (if requested) ----
	if self.precompute_scheme:
		scheme = self.kernel_data['scheme']
		scheme.traits = {
			**eikonal.traits,
			'import_scheme_macro':0,
			'export_scheme_macro':1,
			}
		for key in ('strict_iter_o_macro','multiprecision_macro',
			'walls_macro','minChg_freeze_macro'):
			scheme.traits.pop(key,None)

		scheme.source = cupy_module_helper.traits_header(scheme.traits,
		join=True,size_of_shape=True,log2_size=True,integral_max=integral_max) + "\n"
		scheme.source += model_source+self.cuda_date_modified
		scheme.module = GetModule(scheme.source,self.cuoptions)

		modules.append(scheme.module)

	# Set the constants
	def SetCst(*args,modules=modules):
		for module in modules: SetModuleConstant(module,*args)

	float_t,int_t = self.float_t,self.int_t

	self.size_o = np.prod(self.shape_o)
	SetCst('shape_o',self.shape_o,int_t)
	SetCst('size_o', self.size_o, int_t)

	self.size_tot = self.size_o * np.prod(self.shape_i)
	SetCst('shape_tot',self.shape,int_t) # Used for periodicity
	SetCst('size_tot', self.size_tot,  int_t) # Used for geom indexing


	shape_geom_i,shape_geom_o = [s[self.geom_indep:] for s in (self.shape_i,self.shape_o)]
	if self.geom_indep: # Geometry only depends on a subset of coordinates
		size_geom_i,size_geom_o = [np.prod(s,dtype=int) for s in (shape_geom_i,shape_geom_o)]
		for key,value in [('size_geom_i',size_geom_i),('size_geom_o',size_geom_o),
			('size_geom_tot',size_geom_i*size_geom_o)]: SetCst(key,value,int_t)
	else: SetCst('size_geom_tot', self.size_tot,int_t)

	if policy.multiprecision:
		# Choose power of two, significantly less than h
		h = float(np.min(self.h))
		self.multip_step = 2.**np.floor(np.log2(h/10)) 
		SetCst('multip_step',self.multip_step, float_t, modules=(eikonal.module,flow.module))
		self.multip_max = np.iinfo(self.int_t).max*self.multip_step/2
		SetCst('multip_max', self.multip_max, float_t, modules=(eikonal.module,flow.module))

	if self.factoringRadius:
		SetCst('factor_radius2',self.factoringRadius**2,float_t)
		SetCst('factor_origin', self.seed,              float_t) # Single seed only
		factor_metric = ad.remove_ad(self.CostMetric(self.seed).to_HFM())
		# The drift part of a Rander metric can be ignored for factorization purposes 
		if self.model_=='Rander': factor_metric = factor_metric[:-self.ndim]
		elif self.model_=='Isotropic': factor_metric = factor_metric**2 
		SetCst('factor_metric',factor_metric,float_t)

	if self.order==2:
		order2_threshold = self.GetValue('order2_threshold',0.3,
			help="Relative threshold on second order differences / first order difference,"
			"beyond which the second order scheme deactivates")
		SetCst('order2_threshold',order2_threshold,float_t)		
	
	if self.model_ =='Isotropic':
		SetCst('weights', self.h**-2, float_t)
	if self.isCurvature:
		nTheta = self.shape[2]
		theta = self.hfmIn.Axes()[2]
		eps = self.GetValue('eps',default=0.1,array_float=tuple(),
			help='Relaxation parameter for the curvature penalized models')
		SetCst('decomp_v_relax',eps**2,float_t)

		if traits['xi_var_macro']==0:    SetCst('ixi',  self.ixi,  float_t) # ixi = 1/xi
		if traits['kappa_var_macro']==0: SetCst('kappa',self.kappa,float_t)
		if traits['theta_var_macro']==0: 
			SetCst('cosTheta_s',np.cos(theta),float_t)
			SetCst('sinTheta_s',np.sin(theta),float_t)

	if self.precompute_scheme:
		nactx = self.nscheme['nactx']
		# Convention 'geometry last turns' out to be much faster than the contrary.
		weights=cp.zeros((*shape_geom_o,*shape_geom_i,nactx),float_t)
		offsets=cp.zeros((*shape_geom_o,*shape_geom_i,nactx,self.ndim),self.offset_t)

		updateList_o = cp.arange(np.prod(shape_geom_o,dtype=int_t),dtype=int_t)
		dummy = cp.array(0,dtype=float_t) #; weights[0,0]=1; offsets[0,0,0]=2
		scheme.kernel = scheme.module.get_function("Update")
		# args : u_t,geom_t,seeds_t,rhs_t,..,..,..,updateNext_o
		args=(dummy,eikonal.args['geom'],dummy,dummy,weights,offsets,updateList_o,dummy)
		scheme.kernel((updateList_o.size,),(self.size_i,),args)

		eikonal.args['weights']=weights
		eikonal.args['offsets']=offsets

	# Set the kernel arguments
	policy.nitermax_o = self.GetValue('nitermax_o',default=2000,
		help="Maximum number of iterations of the solver")
	self.raiseOnNonConvergence = self.GetValue('raiseOnNonConvergence',default=True,
		help="Raise an exception if a solver fails to converge")

	# Sort the kernel arguments
	args = eikonal.args
	argnames = ('values','valuesq','valuesNext','valuesqNext',
		'geom','seedTags','rhs','wallDist','weights','offsets')
	eikonal.args = OrderedDict([(key,args[key]) for key in argnames if key in args])
#	print(eikonal.args['wallDist'].dtype)
	flow.args = eikonal.args.copy() # Further arguments added later


"""
			offset_t=self.offset_t
			scheme = self.kernel_data['scheme']
			scheme.traits = {'xi_var_macro':0,'kappa_var_macro':0,'theta_var_macro':0,
				'Scalar':float_t,'Int':int_t,'OffsetT':offset_t,
				'shape_i':self.shape_i,'nTheta':nTheta,'nFejer_macro':traits.get('nFejer_macro',5),
				'niter_i':1,'export_scheme_macro':1}
			scheme.source = cupy_module_helper.traits_header(scheme.traits,
			join=True,size_of_shape=True,log2_size=True,integral_max=integral_max) + "\n"
			scheme.source += model_source+self.cuda_date_modified
			scheme.module = GetModule(scheme.source,self.cuoptions)
			for args in ( ('shape_o',self.shape_o,int_t),('size_o',self.size_o,int_t),
				('shape_tot',self.shape,int_t),('size_tot',size_tot,int_t),
				('ixi',  self.ixi,float_t), ('kappa',self.kappa,float_t),
				('cosTheta_s',np.cos(theta),float_t), ('sinTheta_s',np.sin(theta),float_t),
				('decomp_v_relax',eps**2,float_t)):
				SetModuleConstant(scheme.module,*args)
			nactx = self.nscheme['nactx']
			weights = cp.ascontiguousarray(cp.zeros((nTheta,nactx),float_t))
			offsets = cp.ascontiguousarray(cp.zeros((nTheta,nactx,self.ndim),offset_t))
			updateList_o = cp.arange(int(np.ceil(nTheta/self.shape_i[2])),dtype=int_t)
			dummy = cp.array(0,dtype=float_t) #; weights[0,0]=1; offsets[0,0,0]=2
			scheme.kernel = scheme.module.get_function("Update")
			# args : u_t,geom_t,seeds_t,rhs_t,..,..,..,updateNext_o
			args=(dummy,dummy,dummy,dummy,weights,offsets,updateList_o,dummy)
			scheme.kernel((updateList_o.size,),(self.size_i,),args)
			SetCst('precomp_weights_s',weights,float_t)
			SetCst('precomp_offsets_s',offsets,offset_t)
			self.hfmOut.update({'scheme_weights':weights,'scheme_offsets':offsets})
"""
