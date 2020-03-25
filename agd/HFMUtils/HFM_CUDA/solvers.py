import numpy as np
import time
from . import misc

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
	updateList_o = xp.array(xp.flatnonzero(updateNow_o),dtype=self.int_t)
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
	update_o = xp.array(update_o, dtype='uint8')

	kernel = self.module.get_function("Update")

	"""Pruning is intellectually satisfying, because the expected complexity drops from 
	N+eps*N^(1+1/d) to N, where N is the number of points and eps is a small but positive 
	constant related with the block size. However it has no effect on performance, or a slight
	negative effect, due to the smallness of eps."""
	if self.traits['pruning_macro']: 
		updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
		updatePrev_o = np.full_like(update_o,2*self.ndim+1)
		for niter_o in range(self.nitermax_o):
			"""
			l = updateList_o[updateList_o!=-1]
			show = np.zeros_like(update_o)
			show.reshape(-1)[l]=1
			print(show); #print(np.max(self.block['valuesq']))
			"""
			updateList_o = np.repeat(updateList_o[updateList_o!=-1],2*self.ndim+1)
			if updateList_o.size==0: return niter_o
			kernel((updateList_o.size,),self.shape_i, 
				self.KernelArgs() + (updateList_o,updatePrev_o,update_o))
			updatePrev_o,update_o = update_o,updatePrev_o
	else:
		for niter_o in range(self.nitermax_o):
			updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
#			print(update_o.astype(int)); print()
			update_o.fill(0)
			if updateList_o.size==0: return niter_o
			for key,value in self.block.items(): print(key,type(value))
			kernel((updateList_o.size,),self.shape_i, 
				self.KernelArgs() + (updateList_o,update_o))
#			print(self.block['values'],self.block['valuesNext'],self.block['values'] is self.block['valuesNext'])


	return self.nitermax_o