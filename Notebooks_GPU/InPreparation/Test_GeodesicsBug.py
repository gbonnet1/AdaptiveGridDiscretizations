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
from mpl_toolkits.mplot3d import Axes3D # Plots 3D paths

xp,plt,Eikonal = map(ad.cupy_friendly,(xp,plt,Eikonal))

eps = 0.1 # Small parameter, for penalizing the sub-Riemannian constraint.
xi = 0.5 # Typical radius of curvature.

hfmIn = Eikonal.dictIn({
    'model':'Riemann3_Periodic', # The third dimension is periodic (and only this one), in this model.
    'seed':[0.,0.,0.],
    'arrayOrdering':'RowMajor',
    'tips':[[x,y,t] for x in Eikonal.CenteredLinspace(-1,1,4) 
            for y in Eikonal.CenteredLinspace(-0.5,0.5,2)
           for t in [0,np.pi/3,2*np.pi/3]],
})
hfmIn.SetRect(sides=[[-1,1],[-0.4,0.4],[-np.pi,np.pi]],dims=[101,40,101])
X,Y,Theta = hfmIn.Grid()
zero = np.zeros_like(X)

if hfmIn.mode=='gpu': hfmIn.update({'model':'Riemann3','periodic':(False,False,True)})

if False:
	# This is fine
	o1 = Outer([np.cos(Theta), np.sin(Theta),zero])
	o2 = eps**(-2)*Outer([-np.sin(Theta),np.cos(Theta),zero])
	o3 = xi**2*Outer([zero,zero,1+zero])
	mymetric = o1+o2+o3


if True: 
	# This causes a bug (IllegalMemoryAccess), later when solving geodesics
 	mymetric = Outer([np.cos(Theta), np.sin(Theta),zero]) \
  		+ eps**(-2)*Outer([-np.sin(Theta),np.cos(Theta),zero]) \
 	  	+ xi**2*Outer([zero,zero,1+zero])

#	ReedsSheppMetric = Riemann( # Riemannian metric defined by a positive definite tensor field
#    	Outer([np.cos(Theta), np.sin(Theta),zero])
#    	+ eps**(-2)*Outer([-np.sin(Theta),np.cos(Theta),zero])
#    	+ xi**2*Outer([zero,zero,1+zero])
#	)

#	hfmIn['metric'] = ReedsSheppMetric


def γ(t, height=0.3): return ad.array([np.sin(t), height*np.cos(3*t)])

def tangent(γ,t):
    """Returns the normalized tangent to a curve gamma"""
    t_ad = ad.Dense.identity(constant=t, shape_free=tuple())
    γ_p = γ(t_ad).gradient(0) # gamma'
    return γ_p / norm(γ_p,axis=0) 

def curvature(γ,t):
    """Returns the curvature of a planar curve gamma"""
    t_ad = ad.Dense2.identity(constant=t, shape_free=tuple())
    γ_ad = γ(t_ad)
    γ_p  = γ_ad.gradient(0)
    γ_pp = γ_ad.hessian(0,0)
    return lp.det([γ_p,γ_pp]) / norm(γ_p,axis=0)**3

def lift(γ,t):
    """Orientation in [0,2*pi] of the tangent vector."""
    t_ad = ad.Dense.identity(constant=t, shape_free=tuple())
    γ_ad = γ(t_ad)
    γ_p = γ_ad.gradient(0) # gamma'
    return ad.array([*γ_ad.value, np.arctan2(γ_p[1],γ_p[0])])

IsotropicMetric = Riemann([
    [1.,0.,0.],
    [0.,1.,0.],
    [0.,0.,0.1**2] 
])

T = np.linspace(0,2*np.pi,100)

hfmIn.update({
    'metric' :IsotropicMetric,
    'seeds' : lift(γ,T).T,
    'seedValueVariation' : [*tangent(γ,T),curvature(γ,T)],
    'exportValues' : 1,
})
#hfmIn.pop('tips',None);

hfmOut = hfmIn.Run()
hfmOut = hfmIn.Run()

#print(hfmOut['geodesics'])
