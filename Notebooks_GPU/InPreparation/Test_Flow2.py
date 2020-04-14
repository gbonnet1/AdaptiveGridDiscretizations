import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory 

from agd import HFMUtils
from agd.HFMUtils import HFM_CUDA
import cupy as cp
import numpy as np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from packaging import version
from agd import AutomaticDifferentiation as ad

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))

n=200; 
hfmIn=HFMUtils.dictIn({
    'model':'Isotropic2',
    'seeds':[[0.,0.]],
    'verbosity':0,
    'exportGeodesicFlow':1,
    'geodesic_hlen':20,
    'array_float_caster':cp.asarray,
    'geodesic_traits':{'debug_print':1,},
})
hfmIn.SetRect([[-1,1],[-1,1]],dimx=n+1,sampleBoundary=True)
X = hfmIn.Grid()
hfmIn.update({
    'cost':np.prod(np.sin(2*np.pi*X),axis=0) +1.1, # Non-constant cost
    'tips':hfmIn.Grid(dims=(5,4)).reshape(2,-1).T
})
hfmIn['tips'] = hfmIn['tips'][:1,:]

gpuOut = hfmIn.RunGPU()

#print("flow",gpuOut['flow'])
print("geodesics : ",gpuOut['geodesics'])
print("stopping criteria : ",gpuOut['geodesic_stopping_criteria'])