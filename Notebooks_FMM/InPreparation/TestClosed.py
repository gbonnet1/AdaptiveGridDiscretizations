import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory (useless if conda package installed)
#from Miscellaneous import TocTools; print(TocTools.displayTOC('Curvature','FMM'))

from agd import Eikonal
from agd import FiniteDifferences as fd
from agd import AutomaticDifferentiation as ad
from agd.Interpolation import UniformGridInterpolation
norm = ad.Optimization.norm
#from agd.Plotting import savefig, SetTitle3D; #savefig.dirName = 'Figures/Curvature/'

import numpy as np; xp=np
from matplotlib import pyplot as plt

xp,plt,Eikonal = map(ad.cupy_friendly,(xp,plt,Eikonal))

hfmIn = Eikonal.dictIn({
    'model':'Dubins2',
    'xi':0.3, # Minimal curvature radius
    'cost':1,
    'seed_Unoriented':[0.2,0.5],
    'tip_Unoriented':[1.75,0.5],
    'exportValues':True,
})
hfmIn.SetRect([[0,2],[0,1]],dimx=200)
hfmIn.nTheta = 96

aX0,aX1,aΘ = hfmIn.Axes()
X = ad.array(np.meshgrid(aX0,aX1,indexing='ij'))
nθ = len(aΘ)

hfmIn['tips'] = ad.array([ (*hfmIn['tip_Unoriented'],θ) for θ in aΘ])

def DetectionProba(q,x,r0=0.2):
    """
    Detection probability at x, from a device located at q.
    """
    q,x = fd.common_field((q,x),(1,1))
    dqx2 = ((q-x)**2).sum(axis=0) # Squared distance
    return 1./(1 + dqx2/r0**2)

def ClosedPathCost(hfmIn,hfmOut):
    """
    Total cost of a closed path through the keypoint, depending on the orientation.
    """
    valuesI = UniformGridInterpolation(hfmIn.Grid(),hfmOut['values'],periodic=(False,False,True))
    tipValues = valuesI(hfmIn['tips'].T)
    return tipValues[:nθ//2] + tipValues[nθ//2:]

qDevice = xp.array([[0.5,0.6],[1.2,0.1],[1.4,0.7]]).T
proba = sum(DetectionProba(qi,X,0.1) for qi in qDevice.T)
proba_broadcasted = np.broadcast_to(np.expand_dims(proba,axis=-1), hfmIn.shape)

qDevice_ad = ad.Dense.identity(constant=qDevice)
proba_ad = sum(DetectionProba(qi,X,0.1) for qi in qDevice_ad.T)
proba_ad_broadcasted = np.broadcast_to(np.expand_dims(proba_ad,axis=-1), hfmIn.shape)

print("max proba diff : ",np.max(np.abs(proba_broadcasted-proba_ad_broadcasted.value)))
assert(np.allclose(proba_broadcasted,proba_ad_broadcasted.value))

hfmIn['cost'] = proba_ad_broadcasted
hfmOut_ad = hfmIn.Run()

hfmIn['cost'] = proba_broadcasted
hfmOut = hfmIn.Run()


cost = ClosedPathCost(hfmIn,hfmOut)
iθ = int(np.argmin(cost))

print(hfmOut['geodesic_stopping_criteria'])
print(hfmOut_ad['geodesic_stopping_criteria'])

val_diff = hfmOut['values']-hfmOut_ad['values'].value
where = np.isfinite(hfmOut['values'])

print(np.max(np.abs(val_diff[where])))


valuesI = UniformGridInterpolation(hfmIn.Grid(),hfmOut['values'])

x = xp.array((0.1,0.1,0.1))
print(valuesI(x ).dtype,x.dtype,valuesI.scale.dtype)