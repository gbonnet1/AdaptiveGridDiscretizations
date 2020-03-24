import numpy as np

"""
def dtype(arg,data_t):
	"
	For a numeric array, returns dtype.
	Otherwise, returns one of the provided floating point 
	or integer data type, depending on the argument data type.
	Inputs:
	 - data_t (tuple) : (float_t,int_t)
	"
	float_t,int_t = data_t
	if isinstance(arg,numbers.Real): 
		return float_t
	elif isinstance(arg,numbers.Integral):
		return int_t
	elif isinstance(arg,(tuple,list)):
		return dtype(arg[0],data_t)
	else:
		return arg.dtype
"""

def default_traits(interface):
	"""
	Default traits of the GPU implementation of an HFM model.
	"""
	traits = {
	'Scalar':np.float32,
	'Int':   np.int32,
	'multiprecision_macro':0,
	}

	ndim = interface.ndim

	if ndim==2:
		traits.update({
		'shape_i':(24,24),
		'niter_i':48,
		})
	elif ndim:
		traits.update({
		'shape_i':(4,4,4),
		'niter_i':12,
		})
	else:
		raise ValueError("Unsupported model")

	return traits