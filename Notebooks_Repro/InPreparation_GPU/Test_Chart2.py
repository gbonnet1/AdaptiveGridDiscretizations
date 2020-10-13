import sys; sys.path.insert(0,"../..")

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

hfmIn = Eikonal.dictIn({
    'model':'Isotropic2',
    'seed':[0.5,0.7],
    'exportValues':True,
    'factoringRadius':10,
})

hfmIn.SetSphere(201)

print(hfmIn['chart_mapping'].shape)