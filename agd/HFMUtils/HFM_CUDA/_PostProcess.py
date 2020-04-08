# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from . import misc
from . import kernel_traits
from . import _solvers
from . import cupy_module_helper
from .graph_reverse import graph_reverse
from ... import FiniteDifferences as fd
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ...AutomaticDifferentiation import numpy_like as npl

# This file implements some member functions of the Interface class of HFM_CUDA

def PostProcess(self):
	if self.verbosity>=1: print("Post-Processing")
	values = misc.block_squeeze(self.block['values'],self.shape)
	if self.multiprecision:
		valuesq = misc.block_squeeze(self.block['valuesq'],self.shape)
		if self.GetValue('values_float64',default=False,
			help="Export values using the float64 data type"):
			float64_t = np.dtype('float64').type
			self.hfmOut['values'] = (values.astype(float64_t) 
				+ float64_t(self.multip_step) * valuesq)
		else:
			self.hfmOut['values'] = (values + valuesq.astype(self.float_t)*self.multip_step)
	else:
		self.hfmOut['values'] = values

	# Compute the geodesic flow, if needed, and related quantities
	shape_oi = self.shape_o+self.shape_i
	nact = kernel_traits.nact(self)
	ndim = self.ndim

	traits = self.traits['flow']
	kernel_argnames = self.kernel_argnames['flow']

	if traits.get('flow_weights_macro',False):
		kernel_argnames.append('flow_weights')
		self.block['flow_weights'] = self.xp.empty((nact,)+shape_oi,dtype=self.float_t)
	if traits.get('flow_weightsum_macro',False):
		kernel_argnames.append('flow_weightsum')
		self.block['flow_weightsum'] = self.xp.empty(shape_oi,dtype=self.float_t)
	if traits.get('flow_offsets_macro',False):
		kernel_argnames.append('flow_offsets')
		self.block['flow_offsets'] = self.xp.empty((ndim,nact,)+shape_oi,dtype=np.int8)
	if traits.get('flow_indices_macro',False):
		kernel_argnames.append('flow_indices')
		self.block['flow_indices'] = self.xp.empty((nact,)+shape_oi,dtype=self.int_t)
	if traits.get('flow_vector_macro',False):
		kernel_argnames.append('flow_vector')
		self.block['flow_vector'] = self.xp.ones((ndim,)+shape_oi,dtype=self.float_t)

	self.flow_needed = any(self.flow_traits.get(key+"_macro",False) for key in 
		('flow_weights','flow_weightsum','flow_offsets','flow_indices','flow_vector'))
	if self.flow_needed: self.Solve('flow',solver='global_iteration')

	self.flow = {}
	for key in self.block:
		if key.startswith('flow_'):
			self.flow[key]=misc.block_squeeze(self.block[key],self.shape,contiguous=True)

	if self.model.startswith('Rander') and 'flow_vector' in self.flow:
		if self.dualMetric is None: self.dualMetric = self.metric.dual()
		flow_orig = self.flow['flow_vector']
		m = fd.as_field(self.metric.m,self.shape,depth=2)
		w = fd.as_field(self.metric.w,self.shape,depth=1)
		eucl_gradient = lp.dot_AV(m,flow_orig)+w
		flow = self.dualMetric.gradient(eucl_gradient)
		flow[np.isnan(flow)]=0. # Vanishing flow yields nan after gradient
		flow = self.xp.ascontiguousarray(flow)
		self.flow['flow_vector_orig'],self.flow['flow_vector'] = flow_orig,flow

	if self.exportGeodesicFlow:
		self.hfmOut['flow'] = - self.flow['flow_vector'] * self.h_broadcasted

def SolveLinear(self,diag,indices,weights,rhs,chg,kernelName):
	"""
	A linear solver for the systems arising in automatic differentiation of the HFM.
	"""

	# Set the linear solver traits
	traits = {key:value for key,value in self.traits.items() 
		if key in ('ndim','shape_i','niter','pruning_macro','minchg_freeze_macro')}

	traits.update({
		'nrhs':len(rhs),
		'nindex':len(indices),
		})
	source = cupy_module_helper.traits_header(traits,join=True)+"\n"
	cuoptions = ("-default-device", f"-I {self.cuda_path}") 
	module = cupy_module_helper.GetModule(
		source+'#include "LinearUpdate.h"\n'+self.cuda_date_modified, self.cuoptions)

	self.traits[kernelName] = traits
	self.source[kernelName] = source
	self.module[kernelName] = module

	# Setup the kernel
	def SetCst(*args): cupy_module_helper.SetModuleConstant(module,*args)
	SetCst('shape_o', self.shape_o,       self.int_t)
	SetCst('size_o',  self.size_o,        self.int_t)
	SetCst('size_tot',np.prod(self.shape),self.int_t)

	float_res = np.finfo(self.float_t).resolution
	if not hasattr(self,linear_atol):
		self.linear_atol = self.GetValue('linear_atol',default=float_res*5
			help='Absolute convergence tolerance for the linear systems')
	if not hasattr(self,linear_rtol):
		self.linear_rtol = self.GetValue('linear_rtol',default=float_res*5,
			help='Relative convergence tolerance for the linear systems')

	SetCst('rtol',self.linear_rtol,self.float_t)
	SetCst('atol',self.linear_atol,self.float_t)

	# Call the kernel
	sol = cp.zeros_like(rhs)
	args = (('sol',sol),('rhs',rhs),('diag',diag),('indices',indices),('weights',weights))
	if self.bound_active_blocks: args.append(('chg',chg))
	kernel_argnames = []
	for key,value in args:
		ker = kernelName+"_"+key
		self.block[key]=cp.ascontiguousarray(value)
		kernel_argnames.append(key)

	self.kernel_argnames[kernelName] = kernel_argnames

	self.Solve(kernelName)
	return sol


def SolveAD(self):
	"""
	Forward and reverse differentiation of the HFM.
	"""
	if not (self.forward_ad or self.reverse_ad): return
	
	if self.bound_active_blocks:
		dist = self.block['values']
		if self.multiprecision: 
			dist += self.float_t(self.multip_step) * self.block['valuesq']
	else: dist=0

	diag = self.block['flow_weightsum'].copy() # diagonal preconditioner
	self.boundary = diag==0. #seeds, or walls, or out of domain
	diag[self.boundary]=1.
	
	indices = self.block['flow_indices_twolevel'] 
	weights = self.block['flow_weights']

	if self.forward_ad:
		# Get the rhs
		if self.costVariation is None:
			self.costVariation = xp.zeros(self.shape+self.seedValues.size_ad,
				dtype=self.float_t)
		rhs=self.costVariation 
		if ad.is_ad(self.seedValues):
			rhs[tuple(self.seedIndices.T)] = self.seedValues.coef

		rhs = misc.block_expand(np.moveaxis(rhs,-1,0),self.shape_i,
			mode='constant',constant_values=np.nan,contiguous=True)

		# solve
		valueVariation = self.SolveLinear(rhs,diag,indices,weights,dist,'forwardAD')
		self.hfmOut['valueVariation'] = valueVariation


	if self.reverse_ad:
		# Get the rhs
		rhs = self.GetValue('sensitivity',help='Reverse automatic differentiation')

		# Get the matrix structure
		invalid_index = np.iinfo(self.int_t).max
		indices[weights==0]=invalid_index
		indicesT,weightsT = graph_reverse(indices,weights,invalid=invalid_index)

		allSensitivity = self.SolveLinear(rhs,diag,indicesT,weightsT,-dist,'reverseAD')
		self.hfmOut['costSensitivity'] = allSensitivity #TODO : seedSensitivity







"""
# Failed attempt using a generic sparse linear solver.
def SolveAD(self)
	if self.forward_ad or self.reverse_ad:
		spmod=self.xp.cupyx.scipy.sparse
		xp=self.xp
		diag = self.flow['flow_weightsum'].copy() # diagonal preconditioner
		self.boundary = diag==0. #seeds, or walls, or out of domain
		diag[self.boundary]=1.
		coef = xp.concatenate((xp.expand_dims(diag,axis=0),
				-self.flow['flow_weights']),axis=0)
		diag_precond=True
		if diag_precond: coef/=diag
		size_tot = np.prod(self.shape) # Not same as solver size_tot
		rg = xp.arange(size_tot).reshape((1,)+self.shape)
		row = self.xp.broadcast_to(rg,coef.shape)
		col = xp.concatenate((rg,self.flow['flow_indices']),axis=0)

		self.triplets = (npl.flat(coef),(npl.flat(row),npl.flat(col))) 
		self.spmat = spmod.coo_matrix(self.triplets)

	if self.forward_ad:
		if self.costVariation is None:
			self.costVariation = self.xp.zeros(self.shape+self.seedValues.size_ad,
				dtype=self.float_t)
		rhs=self.costVariation 
		if ad.is_ad(self.seedValues):
			rhs[tuple(self.seedIndices.T)] = self.seedValues.coef
#			rhs/=xp.expand_dims(diag,axis=-1)
		rhs=rhs.reshape(size_tot,-1)

		# Solve the linear system
		csrmat = self.spmat.tocsr()
		# In contrast with scipy, lsqr must do one solve per entry. 
		# Note : lsqr also assumes rhs contiguity
		self.forward_solutions = [ 
			spmod.linalg.lsqr(csrmat,self.xp.ascontiguousarray(r)) for r in rhs.T] 
		self.hfmOut['valueVariation'] = self.xp.stack(
			[s[0].reshape(self.shape) for s in self.forward_solutions],axis=-1) 

	if self.reverse_ad:
		rhs = self.GetValue('sensitivity',help='Reverse automatic differentiation')
		hfmOut['valueSensitivity'] = spmod.linalg.lsqr(self.spmat.T.tocsr(),rhs)
"""