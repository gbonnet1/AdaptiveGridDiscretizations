import sys; sys.path.insert(0,"../../..") # Allow import of agd from parent directory 

from agd import HFMUtils
from agd.HFMUtils import HFM_CUDA
import cupy as xp
import numpy as np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from packaging import version
from agd import AutomaticDifferentiation as ad

"""
shape=(4000,4000)
print('making grid')
grid = np.meshgrid(*(xp.arange(s) for s in shape), 
				indexing='ij')
print('done grid')
grid = ad.array(grid)
print('put together')
grid = xp.array(grid,dtype='float32')
print('converted')
raise
"""


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))

n=21
hfmIn = HFMUtils.dictIn({
    'model':'Isotropic2',
#    'verbosity':1,
    'arrayOrdering':'RowMajor',
    'seeds':[[n-1,n-1]],
#    'kernel':"dummy",
#    'solver':'AGSI', 
#    'solver':'global_iteration',
#    'raiseOnNonConvergence':False,
#    'nitermax_o':10,
#    'tol':1e-8,
    'multiprecision':False,
#    'values_float64':True,

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
#   'debug_print':1,
#    'niter_i':1,
#    'strict_iter_i':1,
    },
#    'nonzero_untidy_kwargs':{'log2_size_i':8,'size2_i':256},
})

if False:
	hfmIn.update({
		'model':'Isotropic3',
		'dims':np.array((200,200,200)),
		'seeds':[[0,0,0]],
		'origin':[-0.5,-0.5,-0.5]
		})
	hfmIn['traits'].update({
		'niter_i':12,
		'shape_i':(4,4,4),
		})


#print(f"Corners {hfmIn.Corners}")
#print(help(hfmIn.SetRect))

#hfmIn.SetRect([[-1,1],[-1,1]],dimx=8)
hfmIn['cost'] = xp.ones(hfmIn['dims'].astype(int),dtype='float32')


#in_raw = hfmIn.RunGPU(returns='in_raw'); print(in_raw['in_raw']['source'])

#out_raw = hfmIn.RunGPU(returns='out_raw'); print(out_raw); hfmOut = out_raw['hfmOut']
hfmOut = hfmIn.RunGPU()

#print(hfmOut['values'].shape)
#print(hfmOut)



if len(hfmOut['values'])<22: print(hfmOut['values'])
print(f"niter_o : {hfmOut['niter_o']}")

#Comparison with CPU.

hfmInCPU = hfmIn.copy()
for key in ('traits','niter_o','solver','raiseOnNonConvergence','nitermax_o'): 
		hfmInCPU.pop(key,None)

hfmInCPU.update({
	'exportValues':1,
#	'factoringMethod':'Static',
#	'factoringPointChoice':'Key',
})

if False: #Isotopic code
	hfmInCPU['cost']=hfmIn['cost'].get()
	hfmOutCPU = hfmInCPU.Run()
	print("Infinity norm of error : ",norm_infinity(hfmOut['values'].get()-hfmOutCPU['values']))
	print(f"GPU(s) : {hfmOut['solverGPUTime']}, CPU(s) : {hfmOutCPU['FMCPUTime']},"
		f"Acceleration : {hfmOutCPU['FMCPUTime']/hfmOut['solverGPUTime']}")

if False: # Riemannian code
	from agd import Metrics
	ndim = len(hfmInCPU['dims'])
	hfmInCPU['model'] = f"Riemann{ndim}"
	hfmInCPU.pop('cost')
	hfmInCPU['metric'] = Metrics.Riemann(np.eye(ndim))

	hfmOutCPU = hfmInCPU.RunSmart()
	print("Infinity norm of error : ",norm_infinity(hfmOut['values'].get()-hfmOutCPU['values']))
	print(f"GPU(s) : {hfmOut['solverGPUTime']}, (Riemann) CPU(s) : {hfmOutCPU['FMCPUTime']}")

#print("kernel_time : ",sum(hfmOut["kernel_time"][1:]))

#import agd.HFMUtils.HFM_CUDA.solvers as solvers
#x = np.array([[1,1],[0,0]])
#print(solvers.neighbors(x,(3,3)))
if len(hfmOut['values'])<20: print(hfmOut['values'].get() - hfmOutCPU['values'])
