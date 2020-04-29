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

def flat(a):
	try: return a.flat # cupy.ndarray (old version ?) does not have flat
	except AttributeError: return a.reshape(-1) 

@implements_cupy_alt(np.full_like,TypeError)
def full_like(arr,*args,**kwargs): # cupy (old version ?) lacks the shape argument
	arr = np.broadcast_to(arr.reshape(-1)[0], kwargs.pop('shape'))
	return np.full_like(arr,*args,**kwargs)

def zeros_like(a,*args,**kwargs): return full_like(a,0.,*args,**kwargs)
def ones_like(a,*args,**kwargs):  return full_like(a,1.,*args,**kwargs)

