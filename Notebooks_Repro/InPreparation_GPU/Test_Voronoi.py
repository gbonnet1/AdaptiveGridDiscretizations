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


n=2; ndim=5
hfmIn = Eikonal.dictIn({
	'mode':'gpu',
    'model':f'Riemann{ndim}',
#    'verbosity':1,
    'seeds':[[0]*ndim],
#    'solver':'AGSI', 
#    'solver':'global_iteration',
    'raiseOnNonConvergence':False,
#    'nitermax_o':1,
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
	'shape_i':(2,)*ndim,'niter_i':10,
#	'niter_i':8,'shape_i':(4,)*ndim,
#	'niter_i':1,'shape_i':(8,8),
#	'niter_i':16,'shape_i':(8,8),
#	'niter_i':32,'shape_i':(16,16),
#	'niter_i':48,'shape_i':(24,24),
#	'niter_i':64,'shape_i':(32,32),
#   'debug_print':1,
#    'niter_i':1,
    'strict_iter_i_macro':1,
	'pruning_macro':0,
	'strict_iter_o_macro':1,
    },
#    'nonzero_untidy_kwargs':{'log2_size_i':8,'size2_i':256},
})

#hfmIn['metric'] = Metrics.Riemann(cp.eye(ndim,dtype=np.float32))
#hfmIn['metric'] = Metrics.Riemann.from_diagonal(cp.array((1,2,3,4,5),dtype=np.float32))
#x=np.array((1,2,3,4,5),dtype=np.float32)
#print(x.flags,"WRITEABLE")

#hfmIn['metric'] = Metrics.Riemann.needle(1+cp.arange(ndim,dtype=np.float32),caster(0.1),caster(1.))
hfmIn['metric'] = Metrics.Riemann.needle(cp.array((0.1,7.3,2.4,5.8,1.6)[:ndim],dtype=np.float32),caster(1.),caster(0.1))

print(hfmIn['metric'].m)

if False:
	hfmIn['model'] = f'Isotropic{ndim}'
	hfmIn.pop('metric')

hfmOut = hfmIn.Run()
#if n<=5: print(hfmOut['values'])

exact = ad.Optimization.norm(hfmIn.Grid(),axis=0,ord=2)
print("difference to exact",norm_infinity(exact-hfmOut['values']))

hfmIn.mode = 'cpu_transfer'

#cpuIn = hfmIn.copy()
#for key in ('traits','array_float_caster'): cpuIn.pop(key)
#cpuIn['metric'] = np.array(cpuIn['metric'].to_HFM().get(),dtype=np.float64)
cpuOut = cpuIn.Run()

diff = cpuOut['values']-hfmOut['values'].get()
print("LInf error: ",np.max(np.abs(diff)))
if n<=20: print(diff)
