import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory 

from agd import Eikonal
from agd import Metrics
import numpy as np; xp=np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from agd import AutomaticDifferentiation as ad

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))

xp,Eikonal = [ad.cupy_friendly(e) for e in (xp,Eikonal)]

n=8
hfmIn = Eikonal.dictIn({
	'model':'Diagonal2',
	'metric':Metrics.Diagonal((1.,1.)),
	'seed':(0,0),
	'factoringRadius':10,
	'exportValues':True,
	})

hfmIn.SetRect([[0,n],[0,n]],dimx=n+1,sampleBoundary=True)
hfmOut = hfmIn.Run()
print(hfmOut['values'])