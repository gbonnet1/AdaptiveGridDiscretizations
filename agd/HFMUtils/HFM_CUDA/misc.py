import numpy as np
from ... import get_array_module


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
	ndim = len(shape_i)
	shape=np.array(arr.shape)
	shape_i = np.array(shape_i)

	# Extend data
	shape_o = round_up(shape,shape_i)
	shape_pad = shape_o*shape_i - shape
	arr = xp.pad(arr, tuple( (0,s) for s in shape_pad), **kwargs) 

	# Reshape
	shape_interleaved = np.stack( (shape_o,shape_i), axis=1).flatten()
#	print(f"{shape=},{shape_i=},{shape_pad=},{shape_o=},{shape_interleaved=}")
	arr = arr.reshape(shape_interleaved)

	# Move axes
	rg = np.arange(ndim)
	axes_interleaved = 1+2*rg
	axes_split = ndim+rg
#	print(f"{axes_interleaved=},{axes_split=}")
	arr = xp.moveaxis(arr,axes_interleaved,axes_split)

	return xp.ascontiguousarray(arr)

def block_squeeze(arr,shape):
	xp = get_array_module(arr)
	ndim = len(shape)
	shape_o = np.array(arr.shape[:ndim])
	shape_i = np.array(arr.shape[ndim:])

	# Move axes
	rg = np.arange(ndim)
	axes_interleaved = 1+2*rg
	axes_split = ndim+rg
	arr = xp.moveaxis(arr,axes_split,axes_interleaved)

	# Reshape
	arr = arr.reshape(shape_o*shape_i)

	# Extract subdomain
	region = tuple(slice(0,s) for s in shape)
	arr = arr.__getitem__(region)
	return arr

# ----- Access an array, maintaining a report of the oprations -------

"""
def HasValue(dico,key,report):
	report['key visited'].append(key)
	return key in dico

def GetValue(dico,key,report,default=None,verbosity=2,help=None):
	
	#Get a value from a dictionnary, printing some requested help.
	
	verb = report['verbosity']

	if key in report['help'] and key not in report['help content']:
		report['help content'][key] = help
		if verb>=1:
			if help is None: 
				print(f"Sorry : no help for key {key}")
			else:
				print(f"---- Help for key {key} ----")
				print(help)
				print("-----------------------------")

	if key in dico:
		report['key used'].append(key)
		return dico[key]
	elif default is not None:
		report['key defaulted'].append((key,default))
		if verb>=verbosity:
			print(f"key {key} defaults to {default}")
		return default
	else:
		raise ValueError(f"Missing value for key {key}")
"""