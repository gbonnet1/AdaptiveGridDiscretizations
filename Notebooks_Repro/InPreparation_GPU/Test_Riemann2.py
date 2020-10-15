import sys; sys.path.insert(0,"../..")
import numpy as np; xp=np

from agd import AutomaticDifferentiation as ad
from agd import Eikonal
from agd import Metrics
from agd import FiniteDifferences as fd
from agd import LinearParallel as lp

xp,Eikonal = [ad.cupy_friendly(e) for e in (xp,Eikonal)]

def surface_metric(x,z,mu):
    ndim,shape = x.ndim-1,x.shape[1:]
    x_ad = ad.Dense.identity(constant=x,shape_free=(ndim,))
    tensors = lp.outer_self( z(x_ad).gradient() ) + mu**-2 * fd.as_field(xp.eye(ndim),shape)
    return Metrics.Riemann(tensors)

def height3(x): 
    r = fd.as_field(lp.rotation(xp.asarray(np.pi)/3,xp.asarray((1.,2,3))),x.shape[1:])
    y = 2*lp.dot_AV(r,x)
    return np.sin(y[0])*np.sin(y[1])*np.sin(y[2])

hfmIn = Eikonal.dictIn({
    'model':'Riemann3',
    'seed':[0.,0.,0.],
    'exportValues':1,
    'clear_hfmIn':True,
#    'traits':{'geom_first_macro':False,},
})
hfmIn.SetRect([[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi]],dimx=301,sampleBoundary=True)
hfmIn.SetUniformTips((4,4,4))
hfmIn['metric'] = surface_metric(hfmIn.Grid(),height3,mu=10)

hfmOut = hfmIn.Run()