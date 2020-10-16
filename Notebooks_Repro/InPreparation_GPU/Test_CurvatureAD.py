import sys; sys.path.insert(0,"../..")
#from Miscellaneous import TocTools; print(TocTools.displayTOC('EikonalAD_GPU','Repro'))

from agd import AutomaticDifferentiation as ad
if ad.cupy_generic.cp is None: raise ad.DeliberateNotebookError('Cupy module required')
from agd import Eikonal
from agd import Metrics
from agd import FiniteDifferences as fd
import agd.AutomaticDifferentiation.cupy_generic as cugen
norm_infinity = ad.Optimization.norm_infinity
#Eikonal.dictIn.default_mode = 'gpu'

import numpy as np; xp=np
import itertools
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

xp,plt,Eikonal = [ad.cupy_friendly(e) for e in (xp,plt,Eikonal)]

hfmIn = Eikonal.dictIn({
    'model':'ReedsSheppForward2',
    'exportValues':True,
    'seed':[0,0,0],
    'xi':0.3,
#    'precompute_scheme':False,
#    'geodesic_online_flow':True,
#    'solver':'AGSI',
#    'nitermax_o':1,
	#'traits':{'offset_t':np.int8},

})
hfmIn.SetRect([[-1,1],[-1,1]],dimx=101)
hfmIn.nTheta = 96
X = hfmIn.Grid()
#hfmIn.SetUniformTips((3,3,3))


delta = ad.Dense.identity(constant=xp.zeros(2))
costValue = (1.+(X[1]<-0.3)).astype(hfmIn.float_t) # (float32 + bool) -> float64 -> float32
costVariation = 1+delta[0]*(X[0]>0.) + delta[1]*(X[0]<=0) # Relative variation
hfmIn['cost'] = costValue*costVariation

hfmOut = hfmIn.Run()

#for geo in hfmOut['geodesics']: plt.plot(*geo[:2])
#plt.show()

val = hfmOut['values']
diff = np.abs(np.where(np.isfinite(val), val.value - val.gradient().sum(axis=0), 0))

print("Big ones : ", np.sum(diff>1e-4) / diff.size)
print("Max rel error : ",np.max(diff/(1+val.value)))