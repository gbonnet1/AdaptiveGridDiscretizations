import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory (useless if conda package installed)
#from Miscellaneous import TocTools; print(TocTools.displayTOC('DeviationHorizontality','FMM'))

from agd import Eikonal
from agd.Metrics import Riemann # Riemannian metric
from agd import AutomaticDifferentiation as ad
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
from agd.Plotting import savefig, SetTitle3D; #savefig.dirName = 'Figures/DeviationHorizontality/'

from agd.LinearParallel import outer_self as Outer # outer product v v^T of a vector with itself
norm = ad.Optimization.norm

import numpy as np; xp=np
import matplotlib.pyplot as plt

xp,plt,Eikonal = map(ad.cupy_friendly,(xp,plt,Eikonal))

hfmIn = Eikonal.dictIn({
    'model':'Elastica2',
    'eps':0.1, # Relaxation parameter
    
    'seed':[-0.84147096, -0.29699776, -0.5       ],
    'tip' :[ 0.04158076, -0.2976676 , -3.0299523 ],
    'precompute_scheme':True,
#    'solver':'AGSI',
})
hfmIn.SetRect(sides=[[-1,1],[-0.4,0.4]],dimx=101)
hfmIn.nTheta = 101

X = hfmIn.Grid()
hfmIn['theta'] = X[2]
hfmIn['xi'] = 1.5
hfmIn['kappa'] = xp.zeros(hfmIn.shape)
hfmIn['cost'] = xp.ones(hfmIn.shape)

hfmOut = hfmIn.Run()