import sys; sys.path.insert(0,"../..")
from agd import Eikonal
from agd import AutomaticDifferentiation as ad
from agd import Metrics
from agd import LinearParallel as lp
norm_infinity = ad.Optimization.norm_infinity

import numpy as np; xp=np
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

xp,Eikonal,plt = [ad.cupy_friendly(e) for e in (xp,Eikonal,plt)]

def Hamiltonian(controls,complement,ϵ=0.1):
    """
    The quadratic Hamiltonian associated to a family of control vector fields, 
    and relaxed with a family of complementary vector fields.
    """
    return lp.outer_self(controls).sum(axis=2) + lp.outer_self(ϵ*complement).sum(axis=2) 

def ω(θ,ϕ,κ0=1,κ1=1):
    z,u = np.zeros_like(θ), np.ones_like(θ)
    return np.stack(ad.asarray([
        [ np.cos(θ),np.sin(θ),z,κ1*np.sin(θ-ϕ)],
        [-np.sin(θ),np.cos(θ),κ0*u,z]
    ]),axis=1)

def ω_(θ,ϕ,ξ=1,ζ=1):
    z,u = np.zeros_like(θ), np.ones_like(θ)
    ρ = -ζ*np.sin(θ-ϕ)
    return np.stack(ad.asarray([
        [ρ*np.cos(θ), ρ*np.sin(θ),z,u],
        [ξ*np.sin(θ),-ξ*np.cos(θ),u,z]
    ]),axis=1)

hfmIn = Eikonal.dictIn({
    'model':'Riemann4',
    'seed':[0,0,0,0],
    'periodic':[True,True,True,True],
    'traits':{'niter_i':3},
    'raiseOnNonConvergence':False,
    'exportValues':True,
    'nitermax_o':100,
#    'precompute_scheme':False,
})

#hfmIn.SetRect([[-0.5,0.5],[-0.5,0.5],[-np.pi,np.pi],[-np.pi,np.pi]],dims=[51,51,64,64])
#hfmIn.SetRect([[-0.5,0.5],[-0.5,0.5],[-np.pi,np.pi],[-np.pi,np.pi]],dims=[5,5,8,8])
hfmIn.SetRect([[-0.5,0.5],[-0.5,0.5],[-np.pi,np.pi],[-np.pi,np.pi]],dims=[10,10,10,10])

hfmIn['origin'][2:] -= hfmIn['gridScales'][2:]/2 # Angular grid starts at -pi
hfmIn.SetUniformTips([4,4,4,3])


_,_,aθ,aϕ = hfmIn.Axes()
assert np.allclose(aθ[0],-np.pi) and np.allclose(aϕ[0],-np.pi)
θs,ϕs = xp.meshgrid(aθ,aϕ,indexing='ij')


r0=0.2; κ0=1/r0
r1=0.2; κ1=1/r1
κs = [κ0,κ1]
hfmIn['dualMetric'] = Metrics.Riemann(Hamiltonian(ω(θs,ϕs,*κs),ω_(θs,ϕs,*κs),ϵ=0.1))

hfmOut = hfmIn.Run()

vals = hfmOut['values']
finite = np.isfinite(vals).sum()/vals.size

print(f"All finite ? {finite}")
print("nan, -inf, inf",np.isnan(vals).sum(),np.sum(vals==-np.inf),np.sum(vals==np.inf))

diff = hfmIn['dualMetric'].m
coefs,offsets = Eikonal.VoronoiDecomposition(diff)

print("Voronoi coefs",np.isfinite(coefs).sum()/coefs.size)

print(Eikonal.VoronoiDecomposition(diff[:,:,5,8]))
print(coefs.shape,coefs[:,5,8])
print("minimal coefficient : ",np.min(coefs))
print("mean coefficient : ",np.mean(coefs))
small_coefs = coefs<1e-5
print("Proportion of small coefficients : ", small_coefs.sum()/small_coefs.size)
print("Min number of small coefficients : ", small_coefs.sum(axis=0).min())


#print(coefs)

reconstruct = (coefs*lp.outer_self(offsets)).sum(axis=2)
coefs = np.maximum(coefs,0.)
reconstruct_pos = (coefs*lp.outer_self(offsets)).sum(axis=2)
#error = 
print("Reconstruction error : ", norm_infinity(reconstruct - diff), 
    "positive parts of coefficients : ", norm_infinity(reconstruct_pos - diff))


# Error : expected 12 coefficients, not 10, for the four dimensional reduction, in the eikonal solver