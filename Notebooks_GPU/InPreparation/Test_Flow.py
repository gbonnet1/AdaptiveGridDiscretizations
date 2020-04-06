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

n=201
hfmIn = HFMUtils.dictIn({
    'model':'Isotropic2',
    'seeds':[[0.,0.]],
    'exportValues':1,
    'cost':cp.array(1.,dtype=np.float32),
    'exportGeodesicFlow':1,
    'tips':[[1,1]],
    'traits':{
#    	'niter_i':16,'shape_i':(8,8),
 #   	'debug_print':1,
    },
    'geodesic_traits':{
#        'debug_print':1,
    },
#    'geodesic_typical_len':25,
#    'geodesic_max_len':40,
})
hfmIn.SetRect([[-1,1],[-1,1]],dimx=n+1,sampleBoundary=True)

gpuOut = hfmIn.RunGPU()

#print("flow",gpuOut['flow'])
print("geodesics : ",gpuOut['geodesics'])
print("stopping criteria : ",gpuOut['geodesic_stopping_criteria'])