import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory 

from agd import HFMUtils
from agd.HFMUtils import HFM_CUDA
import cupy as cp
import numpy as np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from agd import AutomaticDifferentiation as ad

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))

n=20; nTheta=60
hfmIn = HFMUtils.dictIn({
    'model':'Dubins2',
    'seeds':cp.array([[0.,0.,np.pi]],dtype=np.float32),
    'exportValues':1,
    'cost':1,
    'xi':0.2,
})
hfmIn.SetRect([[-1,1],[-1,1]],dimx=n+1,sampleBoundary=True)
hfmIn['dims'] = np.append(hfmIn['dims'],nTheta)

hfmIn.RunGPU()