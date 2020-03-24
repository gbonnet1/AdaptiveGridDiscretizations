import numpy as np
import time
from . import misc
from . import nonzero_untidy
from . import propagate

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

	updated_tot = 0

	# profiling version
	time_flat = 0.
	time_solve = 0.
	s=0

	variant = self.hfmIn['AGSI_variant']
	#flatnonzero version
	if variant=='flatnonzero':
		for niter_o in range(self.nitermax_o):
			updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
			update_o.fill(0)
			updated_tot+=updateList_o.size
			if updateList_o.size==0: print("up tot(flat) ",updated_tot); return niter_o
			kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
		return self.nitermax_o
	

	
	
	if variant=="flatnonzero_profiled":
		time2 = time.time()
		for niter_o in range(self.nitermax_o):
			s+=update_o.reshape(-1)[0]; time1 = time.time(); time_solve+=time1-time2
			updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
			update_o.fill(0)
			updated_tot+=updateList_o.size
			if len(updateList_o)==0: print(f"updated tot {updated_tot}, flat {time_flat}, solve {time_solve}"); return niter_o
			time2 = time.time(); time_flat+=time2-time1
			kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
		return self.nitermax_o
	

	
	# untidy nonzeros version
	find_nonzeros = nonzero_untidy.nonzero(update_o,
		**self.GetValue('nonzero_untidy_kwargs',default={},
			help="Keyword arguments for untidy nonzero lookup"))
	if variant=='untidy':
		for niter_o in range(self.nitermax_o):
			updateList_o,updateList_o_size = find_nonzeros()
			update_o.fill(0)
			updated_tot+=updateList_o_size
			if updateList_o_size==0: print("up tot", updated_tot); return niter_o
			kernel((updateList_o_size,),self.shape_i, kernel_args + (updateList_o,update_o))
		return self.nitermax_o
	

	
	# untidy nonzeros version, profiled
	
	if variant=='untidy_profiled':
		time2 = time.time()
		for niter_o in range(self.nitermax_o):
			s+=update_o.reshape(-1)[0]; time1 = time.time(); time_solve+=time1-time2
			updateList_o,updateList_o_size = find_nonzeros()
			update_o.fill(0)
			updated_tot+=updateList_o_size
			if updateList_o_size==0: print(f"flat {time_flat}, solve {time_solve}"); print("up tot", updated_tot); return niter_o
			time2 = time.time(); time_flat+=time2-time1
			kernel((updateList_o_size,),self.shape_i, kernel_args + (updateList_o,update_o))
		return self.nitermax_o
	
	if variant=='saving':
		updates = [] 
		self.hfmOut['updates'] = updates
		for niter_o in range(self.nitermax_o):
			updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
			update_o.fill(0)
			updates.append(updateList_o)
			updated_tot+=updateList_o.size
			if updateList_o.size==0: print("up tot(flat) ",updated_tot); return niter_o
			kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
		return self.nitermax_o


	if variant=='using':
		start = time.time()
		updates = self.hfmIn['updates'][:-1]
		for updateList_o in updates:
			kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
			np.cumsum(updateList_o.reshape(-1).astype(np.int32))
		print(update_o.reshape(-1)[0])
		elapsed = time.time()-start; print(f"elapsed : {elapsed}")
		return len(updates)

	if variant=='cpu':
		for niter_o in range(self.nitermax_o):

			updateList_o = xp.array(np.flatnonzero(update_o.get()), dtype=self.int_t)
			update_o.fill(0)
			updated_tot+=updateList_o.size
			if updateList_o.size==0: print("up tot(flat) ",updated_tot); return niter_o
			kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
		return self.nitermax_o


	if variant=='propagate':
		find_next = propagate.propagate(update_o)
		updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
		for niter_o in range(self.nitermax_o):
			updated_tot+=updateList_o.size
			if updateList_o.size==0: print("up tot(prop) ",updated_tot);  print(find_next.time); return niter_o
			kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
#			print(f"----\nupdate_o : {update_o}")
			updateList_o = find_next(updateList_o)
		return self.nitermax_o




	#oracle version


