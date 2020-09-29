import sys; sys.path.insert(0,"../..")
import numpy as np; xp=np; allclose = np.allclose

from agd import AutomaticDifferentiation as ad
from agd.Metrics.Seismic import Hooke
mica,_ = Hooke.mica # Hooke tensor associated to this crystal
from agd import Domain
from agd import FiniteDifferences as fd


xp,mica,allclose = map(ad.cupy_friendly,(xp,mica,allclose))

x=ad.Sparse.identity(constant=[1.,2])
x=x*x
for i in range(8): x=x+2*x
print(x.size_ad)
print(x)
x.simplify_ad()
print(x)



def ElasticEnergy(v,hooke,dom,order=1):
    """
    Finite differences approximation of c(ϵ,ϵ), where c is the Hooke tensor and ϵ the stress tensor,
    which is twice the (linearized) elastic energy density.
    """
    assert len(v)==2
    coefs,moffsets = hooke.Selling()
    dvp = tuple( dom.DiffUpwind(v[i], moffsets[i], order=order) for i in range(2))
    dvm = tuple(-dom.DiffUpwind(v[i],-moffsets[i], order=order) for i in range(2))
    
    # Consistent approximations of Tr(moffset*grad(v))
    dv  = ad.array((dvp[0]+dvp[1], dvp[0]+dvm[1], dvm[0]+dvp[1], dvm[0]+dvm[1]))
    dv2 = 0.25* (dv**2).sum(axis=0)
    
    coefs = fd.as_field(coefs,v.shape[1:])
    return (coefs*dv2).sum(axis=0) 

def v(X):
    x0,x1 = X*(2.*np.pi)
    return ad.array((np.cos(x0) - 2.*np.sin(x1),np.cos(x0+2*x1)))

def hooke(X):
    x0,x1 = X*(2.*np.pi)
    angle = 0.3*np.sin(x0)+0.5*np.cos(x1)    
    return mica.extract_xz().rotate_by(angle)

n=20
aX,h = xp.linspace(0,1,20,endpoint=False,retstep=True)
X=ad.array(np.meshgrid(aX,aX,indexing='ij'))
dom = Domain.MockDirichlet(X.shape,h,padding=None) #Periodic domain (wrap instead of pad)

v_ad = ad.Sparse2.identity(constant=np.zeros_like(X))
energy_density_ad = ElasticEnergy(v_ad,hooke(X),dom)
print(f"Stencil cardinality: {energy_density_ad.size_ad2}")

#energy_density_ad[0,0].simplify_ad(atol=0.)

energy_density_ad.simplify_ad(atol=0.)
print(f"Stencil cardinality: {energy_density_ad.size_ad2}")

energy_density_fd = ElasticEnergy(   v(X),hooke(X),dom) # Uses samples of v
energy_fd = 0.5 * energy_density_fd.sum() * h**2

energy_ad = 0.5 * energy_density_ad.sum() * h**2
energy_hess = energy_ad.hessian_operator()
v_fl=v(X).flatten()
energy_fl = 0.5*np.dot(v_fl,energy_hess*v_fl)

assert allclose(energy_fl,energy_fd)

print(type(energy_fd),energy_fd)
print(type(energy_fl),energy_fl)

print(np.max(np.abs(energy_fl-energy_fd)))

print(energy_density_ad[10,5].triplets())
print(v_fl[:10])