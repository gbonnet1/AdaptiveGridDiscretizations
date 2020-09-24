# Code automatically exported from notebook HighAccuracy.ipynb in directory Notebooks_FMM
# Do not modify
from ... import Eikonal
from ... import Metrics
from agd.Metrics.Seismic import Hooke
from ... import AutomaticDifferentiation as ad
from agd.Plotting import savefig; #savefig.dirName = 'Figures/HighAccuracy'

import numpy as np
import matplotlib.pyplot as plt

def PoincareCost(q):
    """
    Cost function defining the Poincare half plane model of the hyperbolic plane.
    """
    return 1/q[1]

def PoincareDistance(p,q):
    """
    Distance between two points of the half plane model of the hyperbolic plane.
    """
    a = p[0]-q[0]
    b = p[1]-q[1]
    c = p[1]+q[1]
    d = np.sqrt(a**2+b**2)
    e = np.sqrt(a**2+c**2)
    return np.log((e+d)/(e-d))

diagCoef = (0.5**2,1) # Diagonal coefficients of M

def diff(x,y,α=0.5): return ad.array([x,y+α*np.sin(np.pi*x)]) # Diffeomorphism f

def RiemannMetric(diag,diff,x,y): 
    X_ad = ad.Dense.identity(constant=(x,y),shape_free=(2,))
    Jac = np.moveaxis(diff(*X_ad).gradient(),0,1)
    return Metrics.Riemann.from_diagonal(diag).inv_transform(Jac)

def RiemannExact(diag,diff,x,y):
    a,b = diag
    fx,fy = diff(x,y)
    return np.sqrt(a*fx**2+b*fy**2)

M=((1.25,0.5),(0.5,2.))

def v(x,y,γ=0.8): return γ*np.sin(np.pi*x)*np.sin(np.pi*y)/np.pi

def RanderMetric(x,y):
    X_ad = ad.Dense.identity(constant=(x,y),shape_free=(2,))
    omega = v(*X_ad).gradient()
    return Metrics.Rander(M,omega)

