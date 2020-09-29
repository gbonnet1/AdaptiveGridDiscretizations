import sys; sys.path.insert(0,"../..")
from agd import Eikonal
from agd import AutomaticDifferentiation as ad
from agd.Metrics import Riemann
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
norm_infinity = ad.Optimization.norm_infinity

import numpy as np; xp=np

xp = ad.cupy_friendly(np)

hfmIn = Eikonal.dictIn({
	'model':'SubRiemann2',
	'seed':[0,0],
	'exportValues':True,
})

n=5
hfmIn.SetRect([[0,1],[0,1]],dimx=n,sample_boundary=True)

V0 = np.zeros((2,1,n))
V1 = V0.copy()
V0[0]=1
V1[1]=1
hfmIn['controls'] = [V0,V1]

hfmOut=hfmIn.Run()
print(hfmOut['values'])
