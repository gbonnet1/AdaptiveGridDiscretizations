import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory 

from agd import Eikonal
import numpy as np; xp=np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from agd import AutomaticDifferentiation as ad
from agd import Metrics
from agd import FiniteDifferences as fd

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))

xp,Eikonal = [ad.cupy_friendly(e) for e in (xp,Eikonal)]

n=5
hfmIn = Eikonal.dictIn({
    'model':'Riemann2',
#    'verbosity':1,
    'arrayOrdering':'RowMajor',
    'seeds':[[0,0]],
#    'solver':'AGSI', 
#    'solver':'global_iteration',
    'raiseOnNonConvergence':False,
    'nitermax_o':200,
    'tol':5*1e-7,
    'multiprecision':True,
#    'values_float64':True,
	'exportValues':True,

#    'help':['nitermax_o','traits'],
	'dims':(n,n+1),
	'origin':[-0.5,-0.5],
	'gridScale':1.,
#	'order':2,
#	'order2_threshold':0.3,
#	'factoringRadius':10000,
#	'seedRadius':2,
#	'returns':'in_raw',
    'precompute_scheme':1,
	'traits':{
	'niter_i':8,'shape_i':(4,4),
#	'niter_i':1,'shape_i':(8,8),
#	'niter_i':16,'shape_i':(8,8),
#	'niter_i':32,'shape_i':(16,16),
#	'niter_i':48,'shape_i':(24,24),
#	'niter_i':64,'shape_i':(32,32),
   'debug_print':1,
#    'niter_i':1,
    'strict_iter_i_macro':1,
	'pruning_macro':0,
	'strict_iter_o_macro':1,
    },
#    'nonzero_untidy_kwargs':{'log2_size_i':8,'size2_i':256},
})

m = xp.array([[1,0.5],[0.5,1]])
gpuM = fd.as_field(m,shape=hfmIn.shape[0:])
cpuM = m
hfmIn['metric'] = Metrics.Riemann(cpuM)

#hfmIn['metric'] = xp.array([1.,0.5,1.],dtype=np.float32) #Metrics.Riemann()
#hfmIn['metric'] = fd.as_field(hfmIn['metric'],shape=hfmIn.shape)
hfmOut = hfmIn.Run()
if n<=20: print(hfmOut['values'])

cpuIn = hfmIn.copy()
cpuIn.pop('traits',None)
#cpuIn['metric'] = Metrics.Riemann(cpuM)
cpuIn['mode'] = 'cpu_transfer'
cpuOut = cpuIn.Run()

diff = cpuOut['values']-hfmOut['values'].get()
print("LInf error: ",np.max(np.abs(diff)))
if n<=20: print(diff)
