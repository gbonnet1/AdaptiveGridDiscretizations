import sys; sys.path.insert(0,"../..")

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
from agd.Metrics.Seismic import Reduced
import agd.AutomaticDifferentiation.cupy_generic as cugen

norm_infinity = ad.Optimization.norm_infinity
from Notebooks_GPU.ExportedCode.Isotropic_Repro import RunCompare

cp = ad.functional.decorate_module_functions(cp,cugen.set_output_dtype32) # Use float32 and int32 types in place of float64 and int64
plt = ad.functional.decorate_module_functions(plt,cugen.cupy_get_args)
HFMUtils.dictIn.RunSmart = cugen.cupy_get_args(HFMUtils.RunSmart,dtype64=True,iterables=(dict,Metrics.Base))

xp=cp
n=7
hfmIn_Constant = HFMUtils.dictIn({
    'model':'TTI2',
    'arrayOrdering':'RowMajor',
    'exportValues':1,
    'seeds':xp.array([[0.,0.]]),
    'factoringMethod':'Static',
    'nitermax_o':1,
    'factoringRadius':10,
#    'seedRadius':2,
    'order':2,
    'traits':{'debug_print':1},
    'raiseOnNonConvergence':False
#    'tips':[[x,y] for y in HFMUtils.CenteredLinspace(-1,1,6) 
#                    for x in HFMUtils.CenteredLinspace(-1,1,6)],
#    'exportGeodesicFlow':1,
})

hfmIn_Constant.SetRect(sides=[[0,1],[0,1]],dimx=n+1,sampleBoundary=True) # Define the domain
X = hfmIn_Constant.Grid() # Horizontal and vertical axis

metric = Reduced(xp.array([1.,1]),0.*xp.array([[0,0.1],[0.1,0.]])) #.rotate_by(xp.array(0.5)) #Linear and quadratic part
hfmIn_Constant['metric'] = metric

metric.cost_bound()

hfmOut = hfmIn_Constant.RunGPU()

print(hfmOut['values'][0,:])