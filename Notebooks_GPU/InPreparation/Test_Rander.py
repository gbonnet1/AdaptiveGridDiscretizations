import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory 

from agd import HFMUtils
from agd.HFMUtils import HFM_CUDA
import cupy as cp
import numpy as np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from agd import AutomaticDifferentiation as ad
from agd import Metrics

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))


n=6
hfmIn = HFMUtils.dictIn({
    'model':'Rander2',
#    'verbosity':1,
    'arrayOrdering':'RowMajor',
    'seeds':[[6,6]],
#    'solver':'AGSI', 
#    'solver':'global_iteration',
    'raiseOnNonConvergence':False,
    'nitermax_o':3,
#    'tol':5*1e-7,
#    'multiprecision':True,
#    'values_float64':True,
	'exportValues':True,

#    'help':['nitermax_o','traits'],
	'dims':np.array((n,n)),
	'origin':[-0.5,-0.5],
	'gridScale':1.,
#	'order':2,
#	'order2_threshold':0.3,
#	'factoringRadius':10000,
#	'seedRadius':2,
#	'returns':'in_raw',
	'traits':{
	'niter_i':8,'shape_i':(4,4),
#	'niter_i':1,'shape_i':(8,8),
#	'niter_i':16,'shape_i':(8,8),
#	'niter_i':32,'shape_i':(16,16),
#	'niter_i':48,'shape_i':(24,24),
#	'niter_i':64,'shape_i':(32,32),
   'debug_print':1,
#    'niter_i':1,
#    'strict_iter_i_macro':1,
	'pruning_macro':0,
#	'strict_iter_o_macro':1,
    },
#    'nonzero_untidy_kwargs':{'log2_size_i':8,'size2_i':256},
})

hfmIn['metric'] = Metrics.Rander(cp.eye(2,dtype=np.float32),cp.array((0.0,0),dtype=np.float32))
if False:
    hfmIn['model']='Riemann2'
    hfmIn['metric']=Metrics.Riemann(cp.eye(2,dtype=np.float32))

hfmOut = hfmIn.RunGPU()
if n<=20: print(hfmOut['values'])

cpuIn = hfmIn.copy()
cpuIn.pop('traits')

"""
cpuIn['metric'] = np.array(cpuIn['metric'].get(),dtype=np.float64)
cpuOut = cpuIn.Run()

diff = cpuOut['values']-hfmOut['values'].get()
print("LInf error: ",np.max(np.abs(diff)))
if n<=20: print(diff)
"""