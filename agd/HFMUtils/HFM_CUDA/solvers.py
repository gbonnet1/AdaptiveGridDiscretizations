import numpy as np
from . import misc

def global_iteration(tol,nitermax_o,data_t,shapes_io,kernel_args,kernel):
	"""
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
		if xp.all(xp.isinf(min_chg)):
			return niter_o
	return nitermax_o


def adaptive_gauss_sidel_iteration(tol,nitermax_o,data_t,shapes_io,kernel_args,kernel):
	float_t,int_t = data_t
	shape_i,shape_o = shapes_io
	xp = misc.get_array_module(kernel_args[0])


