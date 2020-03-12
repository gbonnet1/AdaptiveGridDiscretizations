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
    'solver':'globalIteration',
    'raiseOnNonConvergence':False,
    'niter_o':1,
    'traits':{
    'debug_print':0,
    'niter_i':1
    },
    'verbosity':1,
#    'help':['niter_o','traits'],
	'dims':np.array((10,10)),
	'gridScale':1,
})
#hfmIn.SetRect([[-1,1],[-1,1]],dimx=8)
hfmIn['cost'] = xp.ones(hfmIn['dims'].astype(int),dtype='float32')

out_raw = hfmIn.RunGPU(returns='out_raw'); print(out_raw); hfmOut = out_raw['hfmOut']
hfmOut = hfmIn.RunGPU()
#print(hfmOut['values'])
#print(hfmOut)

hfmInCPU = hfmIn.copy()
for key in ('traits','niter_o','solver'): hfmInCPU.pop(key)
hfmInCPU.update({
	'exportValues':1,
	'cost':hfmIn['cost'].get()
})
hfmOutCPU = hfmInCPU.Run()

print(norm_infinity(hfmOut['values'].get()-hfmOutCPU['values']))
print(hfmOut['values'])
