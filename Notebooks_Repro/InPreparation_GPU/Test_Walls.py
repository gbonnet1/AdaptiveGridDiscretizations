import sys; sys.path.insert(0,"../..")
#from Miscellaneous import TocTools; print(TocTools.displayTOC('Flow_Repro','GPU'))

import cupy as cp
import numpy as np
import itertools
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

from agd import HFMUtils
from agd import AutomaticDifferentiation as ad
from agd import Metrics
from agd import FiniteDifferences as fd
from agd import LinearParallel as lp
import agd.AutomaticDifferentiation.cupy_generic as cugen
norm_infinity = ad.Optimization.norm_infinity

cp = ad.functional.decorate_module_functions(cp,cugen.set_output_dtype32) # Use float32 and int32 types in place of float64 and int64
plt = ad.functional.decorate_module_functions(plt,cugen.cupy_get_args)
HFMUtils.dictIn.RunSmart = cugen.cupy_get_args(HFMUtils.RunSmart,dtype64=True,iterables=(dict,Metrics.Base))

n=7
hfmIn = HFMUtils.dictIn({
    'model':'Isotropic2',
    'seeds':[[0.,0.]],
    'exportValues':1,
    'cost':cp.array(1.),
    'exportGeodesicFlow':1,
    'traits':{
    'debug_print':1,
    'shape_i':(8,8),'niter_i':16,
    },
})
hfmIn.SetRect([[0,1],[0,1]],dimx=n+1,sampleBoundary=True)
X = hfmIn.Grid()
hfmIn['tips']=hfmIn.Grid(dims=(4,4)).reshape(2,-1).T

walls = X[0]>=0.5;
hfmIn['walls']=walls
hfmOut = hfmIn.RunGPU()

print(hfmOut['values'])
print(walls)
