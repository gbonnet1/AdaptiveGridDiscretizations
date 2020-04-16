# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import time
import collections
from . import misc
from ...AutomaticDifferentiation.cupy_support import flat
from .cupy_module_helper import SetModuleConstant
"""
The solvers defined below are member functions of the "interface" class devoted to 
running the gpu eikonal solver.
"""

def Solve(self,name):

	verb = self.verbosity>=2 or (self.verbosity>=1 and name=='eikonal')
	if verb: print(f"Running the {name} GPU kernel")
	data = self.kernel_data[name]
	data.kernel = data.module.get_function("Update")
	solver = data.policy.solver

	SetModuleConstant(data.module,'atol',data.policy.atol,self.float_t)
	SetModuleConstant(data.module,'rtol',data.policy.rtol,self.float_t)

	#Check args
	assert isinstance(data.args,collections.OrderedDict)
	for key,value in data.args.items(): data.args[key] = cp.ascontiguousarray(value)
	for key,value in data.args.items():
		if value.dtype.type not in (self.float_t,self.int_t,np.uint8):
			raise ValueError(f"Inconsistent type {value.dtype.type} for key {key}")

	# Run
	kernel_start = time.time()
	if solver=='global_iteration':
		niter_o = self.global_iteration(data)
	elif solver in ('AGSI','adaptive_gauss_siedel_iteration'):
		niter_o = self.adaptive_gauss_siedel_iteration(data)
	else: raise ValueError(f"Unrecognized solver : {solver}")
	kernel_time = time.time() - kernel_start # TODO : use cuda event ...

	# Report
	if verb: print(f"GPU kernel {name} ran for {kernel_time} seconds, "
		f" and {niter_o} iterations.")

	data.stats.update({
		'niter_o':niter_o,
		'time':kernel_time,})

	if niter_o>=data.policy.nitermax_o:
		nonconv_msg = (f"Solver {solver} for kernel {name} did not "
			f"reach convergence after maximum allowed number {niter_o} of iterations")
		if self.raiseOnNonConvergence: raise ValueError(nonconv_msg)
		else: self.Warn(nonconv_msg)


def KernelArgs(data):
	"""
	Return the arguments to the kernel, applying some pre and post processing steps.
	"""
	policy = data.policy
	args = data.args

	kernel_args = tuple(args.values())

	# Only used for eikonal
	if policy.strict_iter_o:
		args['values'],args['valuesNext'] = args['valuesNext'],args['values']
		if policy.multiprecision:
			args['valuesq'],args['valuesqNext'] = args['valuesqNext'],args['valuesq']

	return kernel_args		

def global_iteration(self,data):
	"""
	Solves the eikonal equation by applying repeatedly the updates on the whole domain.
	"""	
	updateNow_o  = cp.ones(	self.shape_o,   dtype='uint8')
	updateNext_o = cp.zeros(self.shape_o,   dtype='uint8')
	updateList_o = cp.ascontiguousarray(cp.flatnonzero(updateNow_o),dtype=self.int_t)
	nitermax_o = data.policy.nitermax_o

	for niter_o in range(nitermax_o):
		val_old = data.args['values'].copy()
		data.kernel((updateList_o.size,),(self.size_i,), 
			KernelArgs(data) + (updateList_o,updateNext_o))
		if cp.any(updateNext_o): updateNext_o.fill(0)
		else: return niter_o
	return nitermax_o

def adaptive_gauss_siedel_iteration(self,data):
	"""
	Solves the eikonal equation by propagating updates, ignoring causality. 
	"""
	
	trigger = data.trigger
	if trigger.shape==self.shape: 
		trigger = misc.block_expand(data.trigger,self.shape_i,
		mode='constant',constant_values=False)
	trigger = np.any(trigger.reshape(self.shape_o+(-1,)),axis=-1)
	update_o = cp.ascontiguousarray(trigger.astype(np.uint8))
	policy = data.policy
	nitermax_o = policy.nitermax_o

	"""Pruning drops the complexity from N+eps*N^(1+1/d) to N, where N is the number 
	of points and eps is a small but positive constant related with the block size. 
	However it usually has no effect on performance, or a slight negative effect, due
	to the smallness of eps. Nevertheless, pruning allows the bound_active_blocks method,
	which does, sometimes, have a significant positive effect on performance."""
	if data.traits['pruning_macro']:
		updatePrev_o = update_o * (2*self.ndim+1) # Seeds cause their own initial update
		updateNext_o = np.full_like(update_o,0)
		updateList_o = cp.ascontiguousarray(cp.flatnonzero(updatePrev_o), dtype=self.int_t)

		if policy.bound_active_blocks:
			policy.minChgPrev_thres = np.inf
			policy.minChgNext_thres = np.inf
			SetModuleConstant(data.module,'minChgPrev_thres',policy.minChgPrev_thres,self.float_t)
			SetModuleConstant(data.module,'minChgNext_thres',policy.minChgNext_thres,self.float_t)
			minChgPrev_o = cp.full(self.shape_o,np.inf,dtype=self.float_t)
			minChgNext_o = minChgPrev_o.copy()
			def minChg(): return (minChgPrev_o,minChgNext_o)
		else:
			def minChg(): return tuple()

		for niter_o in range(nitermax_o):
									
			#print(updatePrev_o)

			updateList_o = np.repeat(updateList_o,2*self.ndim+1)
			if updateList_o.size==0: return niter_o
			data.kernel((updateList_o.size,),(self.size_i,), 
				KernelArgs(data) + minChg() + (updateList_o,updatePrev_o,updateNext_o))

			"""
			print("--------------- Called kernel ---------------")
			show = np.zeros_like(updateNext_o)
			l=updateList_o
			flat(show)[ l[l<self.size_o] ]=1 # Active
			flat(show)[ l[l>=self.size_o]-self.size_o ]=2 # Frozen
			print(show); #print(np.max(self.block['valuesq']))
			"""

#			print("after kernel: \n",updateNext_o,"\n")
			updatePrev_o,updateNext_o = updateNext_o,updatePrev_o
			updateList_o = updateList_o[updateList_o!=-1]
			if policy.bound_active_blocks: 
				self.set_minChg_thres(data,updateList_o,minChgNext_o)
				minChgPrev_o,minChgNext_o = minChgNext_o,minChgPrev_o

	else: # No pruning
		for niter_o in range(nitermax_o):
			updateList_o = cp.ascontiguousarray(cp.flatnonzero(update_o), dtype=self.int_t)
#			print(update_o.astype(int)); print()
			update_o.fill(0)
			if updateList_o.size==0: return niter_o
#			for key,value in self.block.items(): print(key,type(value))
			data.kernel((updateList_o.size,),(self.size_i,), 
				KernelArgs(data) + (updateList_o,update_o))
#			print(self.block['values'])
#			print(self.block['values'],self.block['valuesNext'],self.block['values'] is self.block['valuesNext'])


	return nitermax_o

def set_minChg_thres(self,data,updateList_o,minChgNext_o):
	"""
	Set the threshold for the AGSI variant limiting the number of active blocks, based
	on causality.
	"""
#	print(f"Entering set_minChg_thres. prev : {self.minChgPrev_thres}, next {self.minChgNext_thres}")
	policy = data.policy
	nConsideredBlocks = len(updateList_o)
	minChgPrev_thres,policy.minChgPrev_thres \
		= policy.minChgPrev_thres,policy.minChgNext_thres

	if nConsideredBlocks<policy.bound_active_blocks:
		policy.minChgNext_thres=np.inf
	else:
		activePos = updateList_o<self.size_o
		nActiveBlocks = int(np.sum(activePos))
		minChgPrev_delta = policy.minChgNext_thres - minChgPrev_thres
		if not np.isfinite(minChgPrev_delta): 
			activeList = updateList_o[activePos]
			activeMinChg = flat(minChgNext_o)[activeList]

			minChgPrev_thres = float(np.min(activeMinChg))
			policy.minChgNext_thres = float(np.max(activeMinChg))
			minChgPrev_delta = policy.minChgNext_thres - minChgPrev_thres
		mult = max(min(policy.bound_active_blocks/max(1,nActiveBlocks),2.),0.7)
		minChgNext_delta = max(minChgPrev_delta * mult, policy.minChg_delta_min)
		policy.minChgNext_thres += minChgNext_delta
	
	SetModuleConstant(data.module,'minChgPrev_thres',policy.minChgPrev_thres,self.float_t)
	SetModuleConstant(data.module,'minChgNext_thres',policy.minChgNext_thres,self.float_t)

