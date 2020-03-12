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

	ax_o = tuple(xp.arange(s,dtype=int_t) for s in shape_o)
	x_o = xp.meshgrid(*ax_o, indexing='ij')
	x_o = xp.stack(x_o,axis=-1)
	min_chg = xp.full(shape_o,np.inf,dtype=float_t)
	report['kernel_time']=[]

#	min_chg = xp.full(x_o.shape[:-1],np.inf,dtype=float_t)
#	print(f"{x_o.flatten()=},{min_chg=}")

	for niter_o in range(nitermax_o):
		time_start = time.time()
		kernel((min_chg.size,),shape_i, kernel_args + (x_o,min_chg))
		report['kernel_time'].append(time.time()-time_start)

#		print(f"min_chg={min_chg}")
#		if xp.all(xp.isinf(min_chg)): return niter_o

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
	xp = misc.get_array_module(kernel_args[0])

	block_seedTags = kernel_args[2]
	block_seeds = np.any(block_seedTags!=0,axis=-1)
	x_o = xp.nonzero(block_seeds)
	x_o = tuple(xp.array(xi_o,dtype=int_t) for xi_o in x_o)
	x_o = xp.stack(x_o,axis=-1)
	min_chg = xp.full(shape_o,np.inf,dtype=float_t)

	report['kernel_time']=[]
	report['block_updates']=[]
	
	for niter_o in range(nitermax_o):
		time_start = time.time()
		kernel((min_chg.size,),shape_i, kernel_args + (x_o,min_chg))
		report['kernel_time'].append(time.time()-time_start)

#		if xp.all(xp.isinf(min_chg)): return niter_o

		"""
		# Handle neighbors structure on the CPU, since there are few.
		x_o,min_chg = x_o.get(),min_chg.get(); yp = np
#		yp=xp
		x_o = x_o[yp.isfinite(min_chg),:]
		x_o = neighbors(x_o,shape_o)
		x_o = xp.array(x_o,dtype=int_t)
"""

	return nitermax_o





