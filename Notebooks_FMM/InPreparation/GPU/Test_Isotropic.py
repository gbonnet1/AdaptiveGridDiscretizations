import sys; sys.path.insert(0,"../../..") # Allow import of agd from parent directory 

from agd import HFMUtils
from agd.HFMUtils import HFM_CUDA
import cupy as xp
import numpy as np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity

"""
import os
print(os.listdir("folder"))
print(max(os.path.getmtime(os.path.join("folder",file)) for file in os.listdir("folder")))
print(os.path.getmtime("./folder"))
"""


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))

hfmIn = HFMUtils.dictIn({
    'model':'Isotropic2',
    'arrayOrdering':'RowMajor',
    'seeds':[[0,0]],
#    'kernel':"dummy",
    'solver':'AGSI', #'global_iteration',
#    'solver':'global_iteration',
    'raiseOnNonConvergence':False,
    'nitermax_o':4000,
    'tol':1e-8,

    'verbosity':1,
#    'help':['nitermax_o','traits'],
	'dims':np.array((4000,4000)),
	'gridScale':1,

	'traits':{
#    'debug_print':1,
#    'niter_i':1,
#    'strict_iter_i':1,
    },

})
#hfmIn.SetRect([[-1,1],[-1,1]],dimx=8)
hfmIn['cost'] = xp.ones(hfmIn['dims'].astype(int),dtype='float32')


#in_raw = hfmIn.RunGPU(returns='in_raw'); print(in_raw['in_raw']['source'])

#out_raw = hfmIn.RunGPU(returns='out_raw'); print(out_raw); hfmOut = out_raw['hfmOut']
hfmOut = hfmIn.RunGPU()
#print(hfmOut['values'])
#print(hfmOut)

hfmInCPU = hfmIn.copy()
for key in ('traits','niter_o','solver','raiseOnNonConvergence','nitermax_o'): 
	hfmInCPU.pop(key,None)
hfmInCPU.update({
	'exportValues':1,
	'cost':hfmIn['cost'].get()
})
hfmOutCPU = hfmInCPU.Run()

print(norm_infinity(hfmOut['values'].get()-hfmOutCPU['values']))
#print(hfmOut['values'])

print(f"GPU(s) : {hfmOut['solverGPUTime']}, CPU(s) : {hfmOutCPU['FMCPUTime']}")
print(f"niter_o : {hfmOut['niter_o']}")
print("kernel_time : ",sum(hfmOut["kernel_time"][1:]))

#import agd.HFMUtils.HFM_CUDA.solvers as solvers
#x = np.array([[1,1],[0,0]])
#print(solvers.neighbors(x,(3,3)))