# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
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
	eikonal = self.kernel_data['eikonal']
	values = misc.block_squeeze(eikonal.args['values'],self.shape)
	if self.multiprecision:
		valuesq = misc.block_squeeze(eikonal.args['valuesq'],self.shape)
		if self.GetValue('values_float64',default=False,
			help="Export values using the float64 data type"):
			float64_t = np.dtype('float64').type
			self.hfmOut['values'] = (values.astype(float64_t) 
				+ float64_t(self.multip_step) * valuesq)
		else:
			self.hfmOut['values'] = (values+valuesq.astype(self.float_t)*self.multip_step)
	else:
		self.hfmOut['values'] = values

	# Compute the geodesic flow, if needed, and related quantities
	shape_oi = self.shape_o+self.shape_i
	nact = kernel_traits.nact(self)
	ndim = self.ndim

	flow = self.kernel_data['flow']
	if flow.traits.get('flow_weights_macro',False):
		flow.args['flow_weights']   = cp.empty((nact,)+shape_oi,dtype=self.float_t)
	if flow.traits.get('flow_weightsum_macro',False):
		flow.args['flow_weightsum'] = cp.empty(shape_oi,dtype=self.float_t)
	if flow.traits.get('flow_offsets_macro',False):
		flow.args['flow_offsets']   = cp.empty((ndim,nact,)+shape_oi,dtype=np.int8)
	if flow.traits.get('flow_indices_macro',False):
		flow.args['flow_indices']   = cp.empty((nact,)+shape_oi,dtype=self.int_t)
	if flow.traits.get('flow_vector_macro',False):
		flow.args['flow_vector']    = cp.empty((ndim,)+shape_oi,dtype=self.float_t)

	self.flow_needed = any(self.flow_traits.get(key+"_macro",False) for key in 
		('flow_weights','flow_weightsum','flow_offsets','flow_indices','flow_vector'))
	if self.flow_needed: self.Solve('flow')

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

	data = self.kernel_data[kernelName]
	eikonal = self.kernel_data['eikonal']

	# Set the linear solver traits
	data.traits = {key:value for key,value in eikonal.traits.items() 
		if key in ('ndim','shape_i','niter','pruning_macro','minchg_freeze_macro')}
	data.traits.update({'nrhs':len(rhs),'nindex':len(indices)})
	data.source = cupy_module_helper.traits_header(traits,join=True)+"\n"
	data.source += '#include "LinearUpdate.h"\n'+self.cuda_date_modified
	data.module = cupy_module_helper.GetModule(data.source, self.cuoptions)
	data.policy = eikonal.policy

	# Setup the kernel
	def SetCst(*args): cupy_module_helper.SetModuleConstant(data.module,*args)
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

	# We use a dummy initialization, to infinity, to track non-visited values
	sol = cp.full(rhs.shape,np.inf,dtype=self.float_t) 
	# Trigger is from the seeds (forward), or farthest points (reverse), excluding walls
	data.trigger = np.all(weights==0.,axis=0)

	# Call the kernel
	data.args = OrderedDict({
		'sol':sol,'rhs':rhs,'diag':diag,'indices':indices,'weights':weights})

	self.Solve(kernelName)
	return sol


def SolveAD(self):
	"""
	Forward and reverse differentiation of the HFM.
	"""
	if not (self.forward_ad or self.reverse_ad): return
	eikonal = self.kernel_data['eikonal']
	
	if eikonal.policy.bound_active_blocks:
		dist = eikonal.args['values']
		if self.multiprecision: 
			dist += self.float_t(self.multip_step) * eikonal.args['valuesq']
	else: dist=0.

	diag = eikonal.args['flow_weightsum'].copy() # diagonal preconditioner
	self.boundary = diag==0. #seeds, or walls, or out of domain
	diag[self.boundary]=1.
	
	indices = eikonal.args['flow_indices_twolevel'] 
	weights = eikonal.args['flow_weights']

	if self.forward_ad:
		rhs = misc.block_expand(self.rhs.gradient(),self.shape_i,
			mode='constant',constant_values=np.nan)
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
# Failed attempt using a generic sparse linear solver. (Fails to converge or too slow.)
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