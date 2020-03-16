import numpy as np
import time
from . import misc

def global_iteration(tol,nitermax_o,data_t,shapes_io,
	kernel_args,kernel,report):
	"""
	Solves the eikonal equation by applying repeatedly the updates on the whole domain.
	Inputs : 
	 - tol (float) : convergence tolerance
	 - nitermax_o (int) : maximum number of iterations 
	 - data_t (tuple) : GPU float_t and int_t data types
	 - block (tuple) : block_values,block_metric,block_seedTags
	"""
	float_t,int_t = data_t
	shape_i,shape_o = shapes_io
	xp = misc.get_array_module(kernel_args[0])

	updateNow_o  = xp.ones( shape_o,dtype='uint8')
	updateNext_o = xp.zeros(shape_o,dtype='uint8') 
	
	for niter_o in range(nitermax_o):
		kernel(shape_o,shape_i, kernel_args + (updateNow_o,updateNext_o))

		if not xp.any(updateNext_o): return niter_o
		updateNow_o.fill(1)
		updateNext_o.fill(0)

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



def adaptive_gauss_siedel_iteration(tol,nitermax_o,data_t,shapes_io,
	kernel_args,kernel,report):
	"""
	Solves the eikonal equation by applying propagating updates, ignoring causality. 
	"""
	float_t,int_t = data_t
	shape_i,shape_o = shapes_io
	block_values,_,block_seedTags,_ = kernel_args
	xp = misc.get_array_module(block_values)

	finite_o = xp.any(xp.isfinite(block_values).reshape( shape_o + (-1,) ),axis=-1)
	seed_o = xp.any(block_seedTags,axis=-1)

	updateNow_o  = xp.array( xp.logical_and(finite_o,seed_o), dtype='uint8')
	updateNext_o = xp.zeros(shape_o,dtype='uint8') 
	
	for niter_o in range(nitermax_o):
		kernel(shape_o,shape_i, kernel_args + (updateNow_o,updateNext_o))
		if not xp.any(updateNext_o): return niter_o
		updateNow_o,updateNext_o = updateNext_o,updateNow_o

	return nitermax_o





