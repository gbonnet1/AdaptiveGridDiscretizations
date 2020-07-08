import numpy as np
from .Base import implements_cupy_alt,expand_dims,cp

"""
This file implements a few numpy functions that not well supported by the
cupy version (6.0, thus outdated) that is available on windows by conda at the 
time of writing.
"""

@implements_cupy_alt(np.max,TypeError)
def max(a,*args,**kwargs):
	initial=kwargs.pop('initial') # cupy (old version ?) does not accept initial argument
	return np.maximum(initial,np.max(a,*args,**kwargs))

def flat(a):
	try: return a.flat # cupy.ndarray (old version ?) does not have flat
	except AttributeError: return a.reshape(-1) 

@implements_cupy_alt(np.full_like,TypeError)
def full_like(arr,*args,**kwargs): # cupy (old version ?) lacks the shape argument
	shape = kwargs.pop('shape')
	if arr.size>0:
		arr = np.broadcast_to(arr.reshape(-1)[0], shape)
		return np.full_like(arr,*args,**kwargs)
	else:
		kwargs.setdefault('dtype',arr.dtype)
		return cp.full(shape,*args,**kwargs)

def zeros_like(a,*args,**kwargs): return full_like(a,0.,*args,**kwargs)
def ones_like(a,*args,**kwargs):  return full_like(a,1.,*args,**kwargs)

def _along_axis(arr,indices,axis):
	axis%=arr.ndim
	def indices_(ax):
		if ax==axis: return indices
		sax = arr.shape[ax]
		ind = np.arange(sax).reshape((1,)*ax + (sax,)+(1,)*(arr.ndim-ax-1))
		return np.broadcast_to(ind,indices.shape)
	return tuple(indices_(ax) for ax in range(arr.ndim))

@implements_cupy_alt(np.take_along_axis,TypeError)
def take_along_axis(arr,indices,axis):
	return arr[_along_axis(arr,indices,axis)]

@implements_cupy_alt(np.put_along_axis,TypeError)
def put_along_axis(arr,indices,values,axis):
	arr[_along_axis(arr,indices,axis)]=values

@implements_cupy_alt(np.ravel_multi_index,TypeError)
def ravel_multi_index(multi_index,dims,mode='raise',order='C'):
	shape = multi_index.shape[1:]
	bound = cp.array(dims,dtype=np.int32).reshape( (len(dims),)+(1,)*len(shape) )
	if mode=='raise':
		if np.any(multi_index<0) or np.any(multi_index>=bound): 
			raise ValueError('Index out of bounds')
	elif mode=='clip':
		multi_index = np.maximum(0,np.minimum(bound,multi_index))
	elif mode=='wrap':
		multi_index = np.mod(multi_index,bound)

	if  order=='F': multi_index,dims = map(reversed,(multi_index,dims))
	elif order!='C': raise ValueError(f"Unrecognized order {order}")

	result = np.zeros_like(multi_index[0])
	for i,d in zip(multi_index,dims):
		result*=d
		result+=i
	return result

@implements_cupy_alt(np.packbits,TypeError)
def packbits(arr,bitorder='big'):
	"""Implements bitorder option in cupy""" 
	if bitorder=='little':
		shape = arr.shape
		arr = arr.reshape(-1,8)
		arr = arr[:,::-1]
		arr = arr.reshape(shape)
	return cp.packbits(arr)

@implements_cupy_alt(np.nanmean,TypeError)
def nanmean(arr):
	pos = np.logical_not(np.isnan(arr))
	return np.mean(arr[pos])



