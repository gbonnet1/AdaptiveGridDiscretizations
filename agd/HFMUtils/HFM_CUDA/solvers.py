import numpy as np
import time
from . import misc

"""
The solvers defined below are member functions of the "interface" class devoted to 
running the gpu eikonal solver.
"""

def global_iteration(self,kernel_args):
	"""
	Solves the eikonal equation by applying repeatedly the updates on the whole domain.
	"""

#	updateNow_o  = xp.ones( shape_o,dtype='uint8')
#	updateNext_o = xp.zeros(shape_o,dtype='uint8') 
	
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

def neighbors(x,shape):
	"""
	Returns the immediate neighbors of x, including x itself, 
	on a cartesian grid of given shape. (Geometry axes last.)
	- shape : bounds of the cartesian grid box
	"""
	xp = misc.get_array_module(x)
	ndim = len(shape)
	x = x.reshape((1,1)+x.shape)
	x = xp.tile(x,(ndim+1,2,1,1))
	for i in range(ndim): x[i,0,:,i] = xp.maximum(x[i,0,:,i]-1,0)
	for i in range(ndim): x[i,1,:,i] = xp.minimum(x[i,1,:,i]+1,shape[i]-1)
	x = x.reshape(-1,ndim)
	x = xp.ravel_multi_index(xp.moveaxis(x,-1,0),shape)
	x = xp.unique(x)
	x = xp.unravel_index(x,shape)
	return xp.stack(x,axis=-1)



def adaptive_gauss_siedel_iteration(self,kernel_args):
	"""
	Solves the eikonal equation by propagating updates, ignoring causality. 
	"""
	xp = self.xp
	block_values,block_seedTags = (self.block[key] for key in ('values','seedTags'))

	finite_o = xp.any(xp.isfinite(block_values).reshape( self.shape_o + (-1,) ),axis=-1)
	seed_o = xp.any(block_seedTags,axis=-1)

	update_o  = xp.array( xp.logical_and(finite_o,seed_o), dtype='uint8')
#	updateNext_o = xp.zeros(self.shape_o, dtype='uint8')
	kernel = self.module.get_function("Update")

	for niter_o in range(self.nitermax_o):
		updateList_o = xp.array(xp.flatnonzero(update_o), dtype=self.int_t)
		update_o.fill(0)
		if len(updateList_o)==0: return niter_o
		kernel((updateList_o.size,),self.shape_i, kernel_args + (updateList_o,update_o))
	return self.nitermax_o





