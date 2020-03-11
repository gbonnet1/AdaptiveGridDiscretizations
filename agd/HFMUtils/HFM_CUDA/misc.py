import numpy as np

def get_array_module(*args):
	"""Returns the array module (numpy or cupy)"""
	assert len(args)>0
	for arg in args:
		module_name = type(arg).__module__
		if module_name=='cupy': 
			return sys.modules[module_name]
		elif module_name!='numpy':
			raise ValueError(f"Non array object {arg}")
	return sys.modules['numpy']

#	import cupy # Alternative implementation requiring cupy import
#	return cupy.get_array_module(*args)


def packbits(arr,bitorder='big'):
	"""Implements bitorder option in cupy"""
	xp = get_array_module(arr)
	if bitorder=='little':
		shape = arr.shape
		arr = arr.reshape(-1,8)
		arr = arr[:,::-1]
		arr = arr.reshape(shape)
	return xp.packbits(arr)


def round_up(num,den):
	"""
	Returns the least multiple of den after num.
	num and den must be integers.
	"""
	return (num+den-1)//den

def block_expand(arr,shape_i,**kwargs):
	"""
	Reshape an array so as to factor  shape_i (the inner shape),
	and move its axes last.
	Inputs : 
	 - **kwargs : passed to xp.pad
	Output : 
	 - padded and reshaped array
	 - original shape
	"""
	xp = get_array_module(arr)
	assert(arr.ndim==len(shape_i))
	shape=np.array(arr.shape)
	shape_i = np.array(shape_i)

	# Extend data
	shape_ext = round_up(shape,shape_i)
	shape_pad = shape-shape_i
	arr = xp.pad(arr, tuple( (0,s) for s in shape_pad), **kwargs) 

	# Reshape
	shape_factor = shape_pad/shape_i
	shape_interleaved = np.stack( (shape_factor,shape_i), axis=1).T.flatten()
	arr = arr.reshape(shape_interleaved)

	# Move axes
	rg = np.xrange(arr.ndim)
	axes_interleaved = 1+2*rg
	axes_split = arr.ndim+rg
	arr = xp.moveaxis(arr,axes_interleaved,axes_split)

	return arr,shape

def block_squeeze(arr,shape):
	xp = get_array_module(arr)
	ndim = len(shape)
	shape_o = arr.shape[:ndim]
	shape_i = arr.shape[ndim:]

	# Move axes
	rg = np.xrange(ndim)
	axes_interleaved = 1+2*rg
	axes_split = ndim+rg
	arr = xp.moveaxis(arr,axes_split,axes_interleaved)

	# Reshape
	arr = arr.reshape(shape_o*shape_i)

	# Extract subdomain
	region = tuple(slice(0,s) for s in shape)
	arr = arr.__getitem__(region)
	return arr