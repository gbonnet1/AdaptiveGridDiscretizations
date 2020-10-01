# Code automatically exported from notebook RollingBall_Models.ipynb in directory Notebooks_Algo
# Do not modify
from ... import AutomaticDifferentiation as ad
from ... import LinearParallel as lp
from ... import Metrics
from ... import FiniteDifferences as fd
norm = ad.Optimization.norm

def rotation_from_quaternion(q):
    """Produces the rotation associated with a unit quaternion."""
    qr,qi,qj,qk = q
    identity = fd.as_field(xp.eye(3),qr.shape,depth=2)
    return identity + 2*ad.array([
        [-(qj**2+qk**2), qi*qj-qk*qr, qi*qk+qj*qr],
        [qi*qj+qk*qr, -(qi**2+qk**2), qj*qk-qi*qr],
        [qi*qk-qj*qr, qj*qk+qi*qr, -(qi**2+qj**2)]])

def quaternion_from_rotation(r):
    """Produces the unit quaternion, with positive real part, associated with a rotation."""
    qr = np.sqrt(lp.trace(r)+1.)/2.
    qi = (r[2,1]-r[1,2])/(4*qr)
    qj = (r[0,2]-r[2,0])/(4*qr)
    qk = (r[1,0]-r[0,1])/(4*qr)
    return ad.array([qr,qi,qj,qk])

def quaternion_from_euclidean(e):
    """Produces a point in the unit sphere by projecting a point in the equator plane."""
    e = ad.asarray(e)
    e2 = lp.dot_VV(e,e)
    return ad.array([1.-e2,*(2*e)])/(1.+e2)
def euclidean_from_quaternion(q):
    """Produces a point in the equator plane from a point in the unit sphere."""
    e2 = (1-q[0])/(1+q[0])
    return q[1:]*((1+e2)/2.)

def euclidean_from_rotation(r,qRef):
    """Produces an euclidean point from a rotation, 
    selecting in the intermediate step the quaternion 
    in the same hemisphere as qRef"""
    q = quaternion_from_rotation(r)
    q[:,lp.dot_VV(q,qRef)<0] *= -1
    return euclidean_from_quaternion(q)

def rotation_from_euclidean(e): 
    """Produces a rotation from an euclidean point. 
    Also returns the intermediate quaternion."""
    q = quaternion_from_euclidean(e)
    return rotation_from_quaternion(q),q

def antisym(a,b,c):
    z=np.zeros_like(a)
    return ad.array([[z, -c, b], [c, z, -a], [-b, a, z]])

def exp_antisym(a,b,c):
    """Matrix exponential of antisym(a,b,c).
    Note : (a,b,c) is the axis of rotation."""
    s = ad.asarray(a**2+b**2+c**2)
    s[s==0]=1e-20 # Same trick as in numpy's sinc function ...
    sq = np.sqrt(s)
    co,si = np.cos(sq),np.sin(sq)
    cosc,sinc = (1-co)/s,si/sq    
    return ad.array([
        [co+cosc*a**2, cosc*a*b-sinc*c, cosc*a*c+sinc*b],
        [cosc*a*b+sinc*c, co+cosc*b**2, cosc*b*c-sinc*a],
        [cosc*a*c-sinc*b, cosc*b*c+sinc*a, co+cosc*c**2]])

def advance(state,control):
    """Move from a state to another by applying a control during a unit time"""
    state,control = map(ad.asarray,(state,control))
    state_physical = state[:-3]
    state_physical = state_physical + 0.25*control[:len(state_physical)] # Additive action on the physical state
    
    state_angular,qRef = rotation_from_euclidean(state[-3:])
    state_angular = lp.dot_AA(state_angular,exp_antisym(*control)) # Left invariant action
    
    return np.concatenate([state_physical,euclidean_from_rotation(state_angular,qRef)],axis=0)

def make_hamiltonian(controls,advance=advance):
    """Produces the hamiltonian function associated to a sub-Riemannian model, 
    defined by its controls and the advance function"""
    def hamiltonian(state):
        """The hamiltonian, a quadratic form on the co-tangent space"""
        # Array formatting to apply to several states simultanously
        state=ad.asarray(state); controls_ = fd.as_field(controls,state.shape[1:],depth=1) 
        
        grad = advance(state,controls_).gradient()
        return lp.dot_AA(lp.transpose(grad),grad)
    return hamiltonian

