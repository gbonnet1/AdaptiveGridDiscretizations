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

n=7
hfmIn = HFMUtils.dictIn({
    'model':'Isotropic2',
    'seeds':[[0.,0.]],
    'exportValues':1,
    'cost':cp.array(1.,dtype=np.float32),
    'exportGeodesicFlow':1,
#    'tips':[[n,n]],
    'traits':{
    	'niter_i':16,'shape_i':(8,8),
    	'debug_print':1,
    },
    'geodesic_traits':{
        'debug_print':1,
    },
    'geodesic_typical_len':20,
    'geodesic_max_len':20,
})
hfmIn.SetRect([[0,n],[0,n]],dimx=n+1,sampleBoundary=True)

gpuOut = hfmIn.RunGPU()

print("flow",gpuOut['geodesicFlow'])
#print("geodesics : ",gpuOut['geodesics'])
#print("stopping criteria : ",gpuOut['geodesic_stopping_criteria'])