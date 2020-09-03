import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory (useless if conda package installed)
#from Miscellaneous import TocTools; print(TocTools.displayTOC('RadarModels','FMM'))

from agd import Eikonal
from agd import FiniteDifferences as fd
from agd import AutomaticDifferentiation as ad
from agd import Metrics
from agd import LinearParallel as lp
from agd.Interpolation import map_coordinates
norm = ad.Optimization.norm
#from agd.Plotting import savefig, SetTitle3D; #savefig.dirName = 'Figures/Curvature/'

import numpy as np; xp=np
from matplotlib import pyplot as plt

xp,plt,Eikonal = map(ad.cupy_friendly,(xp,plt,Eikonal))

hfmIn = Eikonal.dictIn({
    'model':'Isotropic2',
    'seed':[-2.5,0],
})

hfmIn.SetRect([[-3,3],[-2,2]],dimx=301)
hfmIn.SetUniformTips((4,3))

X = hfmIn.Grid()
dx = hfmIn['gridScale']

def RCSMetric(radar_pos,X,*args,**kwargs):
    X,radar_pos = fd.common_field((X,radar_pos),depths=(1,1))
    u = radar_pos - X # Radar direction, for anisotropic part
    r = norm(u,axis=0) # Radar distance, for the isotropic part
    return Metrics.AsymQuad.needle(u,*args,**kwargs).with_cost(0.5+1/(1+r**2))

hfmIn['model'] = 'AsymmetricQuadratic2'
hfmIn.pop('cost',None);

detections = (0.5,2,1) # forward, side, reverse
radar_pos = [0,1.]
hfmIn['metric'] = RCSMetric(radar_pos,X,*detections)

metric = hfmIn['metric']

print(metric.m.dtype)
print(metric.w.dtype)

"""
print('ha1')
a = xp.ascontiguousarray(np.moveaxis(metric.m,(0,1),(-2,-1)))
print('ha2')
v = xp.ascontiguousarray(np.moveaxis(metric.w,0,-1))
print('ha3')
aiv = xp.linalg.solve(a,v) # This one is intolerably long -> Try update of cupy
print('ha4')
"""

print('hey1')
dual = metric.dual()
print('hey2')
flat = dual.flatten(solve_w=True)
print('hey3')
hfmOut = hfmIn.Run() # BUG ??? Correct but too long