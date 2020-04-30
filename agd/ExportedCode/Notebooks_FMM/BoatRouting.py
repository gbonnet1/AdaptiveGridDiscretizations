# Code automatically exported from notebook Notebooks_FMM/BoatRouting.ipynb
# Do not modify
import sys; sys.path.insert(0,"..") # Allow import of agd from parent directory (useless if conda package installed)
#from Miscellaneous import TocTools; print(TocTools.displayTOC('BoatRouting','FMM'))

from ... import HFMUtils
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from agd.Metrics import Rander,Riemann
from ... import AutomaticDifferentiation as ad
from agd.Plotting import savefig; #savefig.dirName = 'Images/BoatRouting'

import numpy as np
import matplotlib.pyplot as plt

def route_min(v,params):
    v,α,ω,M = fd.common_field((v,)+params,depths=(1,0,1,2))
    v_norm = np.sqrt(lp.dot_VAV(v,M,v))
    ωα_norm = np.sqrt( lp.dot_VAV(ω,M,ω) + α)
    cost = 2*(v_norm*ωα_norm - lp.dot_VAV(v,M,ω))
    time = v_norm / ωα_norm
    fuel = cost/time - α
    head = (v/time - ω)/np.sqrt(fuel)
    return {
        'cost':cost,
        'time':time,
        'fuel':fuel, # ρ, instantaneous fuel consumption
        'head':head, # u, where to head for
    }

def metric(params):
    α,ω,M = fd.common_field(params,depths=(0,1,2))
    return Rander( M*(lp.dot_VAV(ω,M,ω) + α), -lp.dot_AV(M,ω)).with_cost(2.)

def Spherical(θ,ϕ): 
    """Spherical embedding: θ is longitude, ϕ is latitude from equator toward pole"""
    return (np.cos(θ)*np.cos(ϕ), np.sin(θ)*np.cos(ϕ), np.sin(ϕ))

def IntrinsicMetric(Embedding,*X):
    X_ad = ad.Dense.identity(constant=X,shape_free=(2,)) # First order dense AD variable
    Embed_ad = ad.asarray(Embedding(*X_ad)) # Differentiate the spherical embedding
    Embed_grad = Embed_ad.gradient()
    Embed_M = lp.dot_AA(Embed_grad,lp.transpose(Embed_grad)) # Riemannian metric
    return Embed_M

def bump(x,y): 
    """Gaussian-like bump (not normalized)"""
    return np.exp(-(x**2+y**2)/2)

def Currents(θ,ϕ):
    """Some arbitrary vector field (water currents)"""
    bump0 = bump(θ+1,(ϕ+0.3)*2); ω0=(0,1) # intensity and direction of the currents
    bump1 = 2*bump(2*(θ-0.7),ϕ-0.2); ω1=(1,-1)
    bump0,ω0,bump1,ω1 = fd.common_field( (bump0,ω0,bump1,ω1), depths=(0,1,0,1))
    return bump0*ω0+bump1*ω1

