import numpy as np
import time
from . import misc
from ...AutomaticDifferentiation.numpy_like import flat
from .cupy_module_helper import SetModuleConstant
"""
The solvers defined below are member functions of the "interface" class devoted to 
running the gpu eikonal solver.
"""

def global_iteration(self):
	"""
	Solves the eikonal equation by applying repeatedly the updates on the whole domain.
	"""	
	xp=self.xp
	updateNow_o  = xp.ones(	self.shape_o,   dtype='uint8')
	updateNext_o = xp.zeros(self.shape_o,   dtype='uint8')
	updateList_o = xp.ascontiguousarray(xp.flatnonzero(updateNow_o),dtype=self.int_t)
	kernel = self.module.get_function("Update")

	for niter_o in range(self.nitermax_o):
		kernel((updateList_o.size,),self.shape_i, self.KernelArgs() + (updateList_o,updateNext_o))
		if xp.any(updateNext_o): updateNext_o.fill(0)
		else: return niter_o
	return self.nitermax_o

def adaptive_gauss_siedel_iteration(self):
	"""
	Solves the eikonal equation by propagating updates, ignoring causality. 
	"""
	xp = self.xp
	block_values,block_seedTags = (self.block[key] for key in ('values','seedTags'))

	finite_o = xp.any(xp.isfinite(block_values).reshape( self.shape_o + (-1,) ),axis=-1)
	seed_o = xp.any(block_seedTags,axis=-1)

	update_o  = xp.logical_and(finite_o,seed_o)
	for k in range(self.ndim): # Take care of a rare bug where the seed is along in its block
		for eps in (-1,1): 
			update_o = np.logical_or(update_o,np.roll(update_o,axis=k,shift=eps))
	update_o = xp.ascontiguousarray(update_o, dtype='uint8')

	kernel = self.module.get_function("Update")

	"""Pruning is intellectually satisfying, because the expected complexity drops from 
	N+eps*N^(1+1/d) to N, where N is the number of points and eps is a small but positive 
	constant related with the block size. However it has no effect on performance, or a slight
	negative effect, due to the smallness of eps."""
	if self.traits['pruning_macro']: 
		updateList_o = xp.ascontiguousarray(xp.flatnonzero(update_o), dtype=self.int_t)
		updatePrev_o = np.full_like(update_o,0)
		flat(updatePrev_o)[updateList_o] = 2*self.ndim+1 # Seeds cause their own initial update
		for niter_o in range(self.nitermax_o):
						
			"""
			show = np.zeros_like(update_o)
			l=updateList_o
			flat(show)[ l[l<self.size_o] ]=1 # Active
			flat(show)[ l[l>=self.size_o]-self.size_o ]=2 # Frozen
			print(show); #print(np.max(self.block['valuesq']))
			"""
			#print(updatePrev_o)

			updateList_o = np.repeat(updateList_o,2*self.ndim+1)
			if updateList_o.size==0: return niter_o
			kernel((updateList_o.size,),self.shape_i, 
				self.KernelArgs() + (updateList_o,updatePrev_o,update_o))
			updatePrev_o,update_o = update_o,updatePrev_o
			updateList_o = updateList_o[updateList_o!=-1]
			if self.bound_active_blocks: set_minChg_thres(self,updateList_o)
	else:
		for niter_o in range(self.nitermax_o):
			updateList_o = xp.ascontiguousarray(xp.flatnonzero(update_o), dtype=self.int_t)
#			print(update_o.astype(int)); print()
			update_o.fill(0)
			if updateList_o.size==0: return niter_o
#			for key,value in self.block.items(): print(key,type(value))
			kernel((updateList_o.size,),self.shape_i, 
				self.KernelArgs() + (updateList_o,update_o))
#			print(self.block['values'])
#			print(self.block['values'],self.block['valuesNext'],self.block['values'] is self.block['valuesNext'])


	return self.nitermax_o

def set_minChg_thres(self,updateList_o):
	"""
	Set the threshold for the AGSI variant limiting the number of active blocks, based
	on causality.
	"""
#	print(f"Entering set_minChg_thres. prev : {self.minChgPrev_thres}, next {self.minChgNext_thres}")
	nConsideredBlocks = len(updateList_o)
	minChgPrev_thres,self.minChgPrev_thres = self.minChgPrev_thres,self.minChgNext_thres
	if nConsideredBlocks<self.bound_active_blocks:
		self.minChgNext_thres=np.inf
	else:
		activePos = updateList_o<self.size_o
		nActiveBlocks = max(1,int(np.sum(activePos)))
		minChgPrev_delta = self.minChgNext_thres - minChgPrev_thres
		if not np.isfinite(minChgPrev_delta): #nActiveBlocks==nConsideredBlocks: #:
			activeList = updateList_o[activePos]
			activeMinChg = flat(self.block['minChgNext_o'])[activeList]
#			print(f"{np.min(activeMinChg)},{type(np.min(activeMinChg))}")
			minChgPrev_thres = float(np.min(activeMinChg))
			self.minChgNext_thres = float(np.max(activeMinChg))
#			print("recomputed")
			minChgPrev_delta = self.minChgNext_thres - minChgPrev_thres
#		print("hi, attempting to bound active blocs")
#		print(f"prev : {minChgPrev_thres}, next : {self.minChgNext_thres}, delta {minChgPrev_delta}")
		minChgNext_delta = minChgPrev_delta * self.bound_active_blocks/nActiveBlocks
#		print(f"active {nActiveBlocks}, bound {self.bound_active_blocks}, next delta {minChgNext_delta}")
		self.minChgNext_thres += minChgNext_delta
	
#	print(f"Leaving set_minChg_thres. prev : {self.minChgPrev_thres}, next {self.minChgNext_thres}")
	mod = self.module
	SetModuleConstant(mod,'minChgPrev_thres',self.minChgPrev_thres,self.float_t)
	SetModuleConstant(mod,'minChgNext_thres',self.minChgNext_thres,self.float_t)

