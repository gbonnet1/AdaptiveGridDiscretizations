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
#    'dual':True,
})
hfmIn.SetRect([[-1,1],[-1,1],[-1,1]],dimx=n)
hfmIn.SetSphere(dimsp=n,separation=False)

