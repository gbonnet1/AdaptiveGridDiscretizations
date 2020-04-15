import numpy as np
from . import cupy_generic
from .ad_generic import is_ad
from .numpy_like import implements_cupy_alt

"""
This file implements a few numpy functions that not well supported by the
cupy version (6.0, thus outdated) that is available on windows by conda at the 
time of writing.
"""

@implements_cupy_alt(np.max,TypeError)
def max(a,*args,**kwargs):
	initial=kwargs.pop('initial') # cupy (old version ?) does not accept initial argument
	return np.maximum(initial,np.max(a,*args,**kwargs))

"""
@implements(np.max,cupy_only=True)
def max(a,*args,**kwargs):
	try: return np.max(a,*args,**kwargs)
	except TypeError: # cupy (old version ?) does not accept initial argument
		initial = kwargs.pop('initial')
		return np.maximum(initial,np.max(a,*args,**kwargs))
"""

def flat(a):
	try: return a.flat # cupy.ndarray (old version ?) does not have flat
	except AttributeError: return a.reshape(-1) 

@implements_cupy_alt(np.expand_dims,ValueError)
def expand_dims(a,axis): # numpy will not accept cupy arrays
	if axis<0: axis=1+a.ndim+axis
	newshape = a.shape[:axis]+(1,)+a.shape[axis:]
	return a.reshape(newshape)
"""
@implements(np.expand_dims,cupy_only=True)
def expand_dims(a,axis):
	try: return np.expand_dims(a,axis)
	except TypeError: 
		if axis<0: axis=1+a.ndim+axis
		newshape = a.shape[:axis]+(1,)+a.shape[axis:]
		return a.reshape(newshape)
"""

@implements_cupy_alt(np.full_like,TypeError)
def full_like(arr,*args,**kwargs): # cupy (old version ?) lacks the shape argument
	arr = np.broadcast_to(arr.flatten()[0], kwargs.pop('shape'))
	return np.full_like(arr,*args,**kwargs)

"""
def _full_like(a,*args,**kwargs): # needed for cupy variant below
	if is_ad(a): return type(a)(np.full_like(a.value,*args,**kwargs))
	else: return np.full_like(a,*args,**kwargs) 

@implements(np.full_like,cupy_only=True)
def full_like(arr,*args,**kwargs):
	try: return _full_like(arr,*args,**kwargs)
	except TypeError: # Some old versions of cupy lack the shape argument
		arr = np.broadcast_to(arr.flatten()[0], kwargs.pop('shape'))
		return _full_like(arr,*args,**kwargs)
"""
def zeros_like(a,*args,**kwargs): return full_like(a,0.,*args,**kwargs)
def ones_like(a,*args,**kwargs):  return full_like(a,1.,*args,**kwargs)

