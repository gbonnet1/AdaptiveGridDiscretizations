import numpy as np
from . import misc

def global_iteration(tol,nitermax_o,data_t,shapes_io,kernel_args,kernel):
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

	ax_o = tuple(xp.arange(s,dtype=int_t) for s in shape_o)
	x_o = xp.meshgrid(*ax_o, indexing='ij')
	x_o = xp.stack(x_o,axis=-1)
	min_chg = xp.full(x_o.shape[:-1],np.inf,dtype=float_t)
#	print(f"{x_o.flatten()=},{min_chg=}")

	for niter_o in range(nitermax_o):
		kernel((min_chg.size,),shape_i, kernel_args + (x_o,min_chg))
#		print(f"min_chg={min_chg}")
		if xp.all(xp.isinf(min_chg)): return niter_o

	return nitermax_o

def neighbors(x,shape):
	"""
	Returns the immediate neighbors of x, including x itself, 
	on a cartesian grid bounded by shape. (Geometry axes last.)
	- shape : bounds of the cartesian grid box
	"""
	xp = misc.get_array_module(x)
	ndim = len(shape)
	x = x.reshape((x.shape[0],1,1,x.shape[1]))
	print(x.shape)
	x = xp.tile(x,(1,ndim+1,2,1))
	print(x.shape)
	for i in range(ndim): x[:,i,i,0] = xp.minimum(x[i,0,:,i]+1,shape[i])
	for i in range(ndim): x[i,1,:,i] = xp.maximum(x[i,1,:,i]-1,0)
	x = x.reshape(-1,ndim)
	x = xp.ravel_multi_index(x,shape)
	x = xp.unique(x)
	x = xp.unravel_index(x,shape)
	return x



def adaptive_gauss_sidel_iteration(tol,nitermax_o,data_t,shapes_io,kernel_args,kernel):
	"""
	Solves the eikonal equation by applying propagating updates, ignoring causality. 
	"""
	float_t,int_t = data_t
	shape_i,shape_o = shapes_io
	xp = misc.get_array_module(kernel_args[0])

	block_seedTags = kernel_args[3]
	block_seeds = np.any(block_seedTags!=0,axis=-1)
	x_o = xp.stack(xp.nonzero(block_seeds,dtype=int_t),axis=-1)

	for niter_o in range(nitermax_o):
		min_chg = xp.empty(len(x_o),dtype=float_t)
		kernel((min_chg.size,),shape_i, kernel_args + (x_o,min_chg))
		if xp.all(xp.isinf(min_chg)): return niter_o

		# Handle neighbors structure on the CPU, since there are few.
		x_o,min_chg = x_o.get(),min_chg.get(); yp = np
		x_o = x_o[yp.isfinite(min_chg),:]
		x_o = neighbors(x_o,shape_o)
		x_o = xp.array(x_o)

	return nitermax_o





