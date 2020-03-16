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
	updateNow_o  = xp.arange(self.size_o, dtype=self.int_t)
	updateNext_o = xp.zeros( self.shape_o,    dtype='uint8')
	updateList_o = xp.nonzero(updateNow_o.flatten(), dtype=self.int_t)

	for niter_o in range(nitermax_o):
		kernel(updateList_o.size,self.shape_i, kernel_args + (updateList_o,updateNext_o))
		if xp.any(updateNext_o): updateNext_o.fill(0)
		else: return niter_o
	return nitermax_o

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

	finite_o = xp.any(xp.isfinite(block_values).reshape( shape_o + (-1,) ),axis=-1)
	seed_o = xp.any(block_seedTags,axis=-1)

	updateNow_o  = xp.array( xp.logical_and(finite_o,seed_o), dtype='uint8')
	updateNext_o = xp.zeros(shape_o, dtype='uint8') 
	
	for niter_o in range(nitermax_o):
		updateList_o = xp.nonzero(updateNow_o.flatten(), dtype=self.int_t)
		kernel(updateList_o.size,shape_i, kernel_args + (updateList_o,updateNext_o))
		if not xp.any(updateNext_o): return niter_o
		updateNow_o,updateNext_o = updateNext_o,updateNow_o

	return nitermax_o





