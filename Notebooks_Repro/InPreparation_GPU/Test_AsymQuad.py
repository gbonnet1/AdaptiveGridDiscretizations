import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory 

from agd import Eikonal
import cupy as cp
import numpy as np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from agd import AutomaticDifferentiation as ad
from agd import Metrics

def caster(arr): return cp.asarray(arr,dtype=np.float32) # array_float_caster
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))


n=5; ndim=2
hfmIn = Eikonal.dictIn({
	'mode':'gpu',
    'model':f'AsymQuad{ndim}',
#    'verbosity':1,
    'seeds':[[0]*ndim],
#    'solver':'AGSI', 
#    'solver':'global_iteration',
    'raiseOnNonConvergence':False,
    'nitermax_o':1,
#    'tol':5*1e-7,
#    'multiprecision':True,
#    'values_float64':True,
	'exportValues':True,
#	'factoringRadius':10,
#    'help':['nitermax_o','traits'],
	'dims':[n]*ndim,
	'origin':[-0.5]*ndim,
	'gridScale':1.,
#	'order':2,
#	'order2_threshold':0.3,
#	'factoringRadius':10000,
#	'seedRadius':2,
#	'returns':'in_raw',

	'traits':{
	'debug_print':1,
#	'shape_i':(2,)*ndim,'niter_i':10,
#	'niter_i':8,'shape_i':(4,)*ndim,
#	'niter_i':1,'shape_i':(8,8),
#	'niter_i':16,'shape_i':(8,8),
#	'niter_i':32,'shape_i':(16,16),
#	'niter_i':48,'shape_i':(24,24),
#	'niter_i':64,'shape_i':(32,32),
#   'debug_print':1,
#    'niter_i':1,
#    'strict_iter_i_macro':1,
	'pruning_macro':0,
	'strict_iter_o_macro':1,
    },
#    'nonzero_untidy_kwargs':{'log2_size_i':8,'size2_i':256},
})

norm = Metrics.AsymQuad(cp.eye(ndim,dtype=np.float32), cp.array([-1,0],dtype=np.float32))

#norm = Metrics.AsymQuad(cp.eye(ndim,dtype=np.float32), cp.zeros(ndim,dtype=np.float32))
hfmIn['metric'] = norm

hfmOut = hfmIn.Run()
if n<=5: print(hfmOut['values'])

exact = norm.norm(hfmIn.Grid())
print("difference to exact",norm_infinity(exact-hfmOut['values']))
