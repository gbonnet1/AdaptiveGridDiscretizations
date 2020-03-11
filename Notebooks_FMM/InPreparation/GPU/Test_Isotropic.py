import sys; sys.path.insert(0,"../../..") # Allow import of agd from parent directory 

from agd import HFMUtils
from agd.HFMUtils import HFM_CUDA
import cupy as xp
import numpy as np
import time


import os
print(os.listdir("folder"))
print(max(os.path.getmtime(os.path.join("folder",file)) for file in os.listdir("folder")))
print(os.path.getmtime("./folder"))

"""

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

hfmIn = HFMUtils.dictIn({
    'model':'Isotropic2',
    'arrayOrdering':'RowMajor',
    'seeds':[[0,0]],
#    'kernel':"dummy",
    'solver':'globalIteration',
    'niter_o':2,
    'traits':{
    'debug_print':1,
    'niter_i':1
    },
    'verbosity':1,
#    'help':['niter_o','traits'],
	'dims':np.array((8,8)),
	'gridScale':1,
})
#hfmIn.SetRect([[-1,1],[-1,1]],dimx=8)
hfmIn['cost'] = xp.ones(hfmIn['dims'].astype(int),dtype='float32')

hfmOut = hfmIn.RunGPU(returns='out_raw')
print(hfmOut)

#print(hfmOut['values'])
#print(hfmOut)

"""