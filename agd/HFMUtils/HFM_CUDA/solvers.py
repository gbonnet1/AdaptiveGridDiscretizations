import numpy as np
import time
from . import misc
from . import nonzero_untidy

"""
The solvers defined below are member functions of the "interface" class devoted to 
running the gpu eikonal solver.
"""

def global_iteration(self,kernel_args):
	"""
	Solves the eikonal equation by applying repeatedly the updates on the whole domain.
	"""	
	xp=self.xp
	updateNow_o  = xp.ones(	self.shape_o,   dtype='uint8')
	updateNext_o = xp.zeros(self.shape_o,   dtype='uint8')
	updateList_o = xp.array(xp.flatnonzero(updateNow_o),dtype=self.int_t)
	kernel = self.module.get_function("Update")

	for niter_o in range(self.nitermax_o):
		kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,updateNext_o))
		if xp.any(updateNext_o): updateNext_o.fill(0)
		else: return niter_o
	return self.nitermax_o

def adaptive_gauss_siedel_iteration(self,kernel_args):
	"""
	Solves the eikonal equation by propagating updates, ignoring causality. 
	"""
	xp = self.xp
	block_values,block_seedTags = (self.block[key] for key in ('values','seedTags'))

	finite_o = xp.any(xp.isfinite(block_values).reshape( self.shape_o + (-1,) ),axis=-1)
	seed_o = xp.any(block_seedTags,axis=-1)

	update_o  = xp.array( xp.logical_and(finite_o,seed_o), dtype='uint8')
	kernel = self.module.get_function("Update")

	"""#flatnonzero version
	for niter_o in range(self.nitermax_o):
		updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
		update_o.fill(0)
		if len(updateList_o)==0: return niter_o
		kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
	return self.nitermax_o
	"""

	# profiled version
	time_flat = 0.
	time_solve = 0.
	s=0
	time2 = time.time()
	for niter_o in range(self.nitermax_o):
		s+=update_o.flatten()[0]
		time1 = time.time(); time_solve+=time1-time2
		updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
		update_o.fill(0)
		if len(updateList_o)==0: print(f"flat {time_flat}, solve {time_solve}"); return niter_o
		time2 = time.time(); time_flat+=time2-time1
		kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
	return self.nitermax_o

	# untidy nonzeros version
	find_nonzeros = nonzero_untidy.nonzero(update_o,
		interface.GetValue('find_nonzeros_kwargs',default={},
			help="Keyword arguments for untidy nonzero lookup"))

	for niter_o in range(self.nitermax_o):
		updateList_o,updateList_o_size = find_nonzeros()
		update_o.fill(0)
		if len(updateList_o)==0: return niter_o
		kernel((updateList_o_size,),self.shape_i, kernel_args + (updateList_o,update_o))
	return self.nitermax_o


