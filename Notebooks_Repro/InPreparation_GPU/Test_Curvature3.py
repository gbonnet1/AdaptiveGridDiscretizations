import sys; sys.path.insert(0,"../..")
#from Miscellaneous import TocTools; print(TocTools.displayTOC('Flow_GPU','Repro'))

from agd import Eikonal
from agd import AutomaticDifferentiation as ad
from agd import FiniteDifferences as fd
from agd import Metrics
from agd import LinearParallel as lp
norm = ad.Optimization.norm

import numpy as np; xp=np; allclose = np.allclose
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

xp,Eikonal,plt,allclose = [ad.cupy_friendly(e) for e in (xp,Eikonal,plt,allclose)]

n=50 
hfmIn = Eikonal.dictIn({
    'model':'ReedsSheppGPU3',
    'cost':1.,
    'seed':[0,0,0,0,0],
    
    # Define euclidean geometry on R^3 x S^2, for illustration
    'eps':1,
    'xi':1,
    'traits':{'decomp_v_align_macro':False},
    
    # Recompute flow at geodesic extraction
#    'geodesic_online_flow':True,

#    'dual':True,
#    'forward':True,    
})
hfmIn.SetRect([[-1,1],[-1,1],[-1,1]],dimx=n)
transformations = hfmIn.SetSphere(dimsp=32,separation=False) # projective model
#hfmIn.SetUniformTips((3,3,3,3,3))

hfmOut = hfmIn.Run()