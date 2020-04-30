# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

# This file implements Selling's algorithm in dimension two and three,
# which is used to construct tensor decompositions
# It performs this on parallel on a family of matrices

import numpy as np
from itertools import cycle
from .LinearParallel import dot_VV, dot_AV, perp, cross

iterMax2 = 100 
iterMax3 = 100 

# -------- Dimension based dispatch -----

def _GetDimBounds(m):
    """Returns dim,bounds where m.shape = (dim,dim)+bounds. Purposedly fails if dim not in [1,2,3]."""
    shape = m.shape
    if len(shape)<2:
        raise ValueError("Selling error : insufficient depth of input array. Shape : " + str(shape))
    dim = shape[0]
    if dim != shape[1]:
        raise ValueError("Selling error : non square matrix. Shape : " +str(shape))
    if not dim in (1,2,3):
        raise ValueError("Selling error : unsupported matrix dimension. Shape : "+ str(shape))
    return dim, shape[2:]

def ObtuseSuperbase(m,sb=None):
    """
    In : symmetric positive definite matrix m. Initial superbase sb (optional).
    Out : obtuse superbase is stored in input argument sb if it is not None. Else it is returned.
    """
    dim,bounds = _GetDimBounds(m)
    osb = CanonicalSuperbase(dim,bounds) if sb is None else sb

    if   dim==1:            success = _ObtuseSuperbase1(m,osb)    
    elif dim==2:            success = _ObtuseSuperbase2(m,osb)
    else: assert dim==3;    success = _ObtuseSuperbase3(m,osb) 

    if sb is None:
        if not success: raise ValueError('Selling.Decomposition2 error: Selling algorithm unterminated')
        else: return osb
    else:
        sb = osb
        return success

def Decomposition(m,sb=None):
    """
         Use Selling's algorithm to decompose a tensor

        input : symmetric positive definite tensor, d<=3. Superbase (optional).
        output : coefficients, offsets
    """
    dim,bounds=_GetDimBounds(m)
    if sb is None:
        sb = ObtuseSuperbase(m)

    if   dim==1:            return _Decomposition1(m,sb)    
    elif dim==2:            return _Decomposition2(m,sb)
    else: assert dim==3;    return _Decomposition3(m,sb)


def GatherByOffset(T,Coefs,Offsets):
    """
        Get the coefficient of a each offset
    """
    TimeCoef = {};
    for (i,t) in enumerate(T):
        coefs = Coefs[:,i]
        offsets = Offsets[:,:,i]
        for (j,c) in enumerate(coefs):
            offset = tuple(offsets[:,j].astype(int))
            offset_m = tuple(-offsets[:,j].astype(int))
            if offset<offset_m:
                offset=offset_m
            if offset in TimeCoef:
                TimeCoef[offset][0].append(t)
                TimeCoef[offset][1].append(c)
            else:
                TimeCoef[offset] = ([t],[c])
    return TimeCoef


def CanonicalSuperbase(d, bounds = tuple()):
    sb=np.zeros((d,d+1,)+bounds)
    sb[:,0]=-1
    for i in range(d):
        sb[i,i+1]=1
    return sb

def SuperbasesForConditioning(cond,dim=2):
    """
    Returns a family of superbases. 
    One of them is M-obtuse, for any positive definite matrix M with condition number below the given bound.
    (Condition number is the ratio of the largest to the smallest eigenvalue.)
    """
    if dim==1:          return _SuperbasesForConditioning1(cond)
    elif dim==2:        return _SuperbasesForConditioning2(cond)
    else: assert dim==3;return _SuperbasesForConditioning3(cond)

# ------- One dimensional variant (trivial) ------

def _ObtuseSuperbase1(m,sb=None):
    osb = CanonicalSuperbase(*_GetDimBounds(m))
    if sb is None:  return osb
    else: sb=osb; return True

def _Decomposition1(m,sb):
    _,bounds = _GetDimBounds(m)
    offsets = sb[:,0].reshape((1,1,)+bounds)
    coefs = m.reshape((1,)+bounds)    
    return coefs, offsets.astype(int)

def _SuperbasesForConditioning1(cond):
    sb = CanonicalSuperbase(1)
    return sb.reshape(sb.shape+(1,))

# ------- Two dimensional variant ------

# We do everyone in parallel, without selection or early abort
def _ObtuseSuperbase2(m,sb):
    """
        Use Selling's algorithm to compute an obtuse superbase.

        input : symmetric positive definite matrix m, dim=2
        input/output : superbase b (must be valid at startup)
        
        module variable : iterMax2, max number of iterations

        output : wether the algorithm succeeded
    """
    iterReducedMax = 3
    iter=0
    iterReduced = 0
    sigma = cycle([(0,1,2),(1,2,0),(2,0,1)])
    while iterReduced<iterReducedMax and iter<iterMax2:
        # Increment iterators
        iter += 1
        iterReduced += 1
        (i,j,k) = next(sigma)
        
        # Test for a positive angle, and modify superbase if necessary
        acute = dot_VV(sb[:,i],dot_AV(m,sb[:,j])) > 0
        if np.any(acute):
            sb[:,k,acute] = sb[:,i,acute]-sb[:,j,acute]
            sb[:,i,acute] = -sb[:,i,acute]
            iterReduced=0
    
    return iterReduced==iterReducedMax
    
# Produce the matrix decomposition
def _Decomposition2(m,sb):
    """
        Use Selling's algorithm to decompose a tensor

        input : symmetric positive definite tensor 
        output : coefficients, offsets
    """
    _,bounds = _GetDimBounds(m)
    coef=np.zeros((3,)+bounds)
    for (i,j,k) in [(0,1,2),(1,2,0),(2,0,1)]:
        coef[i] = -dot_VV(sb[:,j], dot_AV(m, sb[:,k]) )
    
    return coef,perp(sb).astype(int)

def _SuperbasesForConditioning2(cond):
    """
    Implementation is based on exploring the Stern-Brocot tree, 
    with a stopping criterion based on the angle between consecutive vectors.
    """

    mu = np.sqrt(cond)
    theta = np.pi/2. - np.arccos( 2/(mu+1./mu))

    u=np.array( (1,0) )
    l = [np.array( (-1,0) ),np.array( (0,1) )]
    m = []
    superbases =[]

    def angle(u,v): return np.arctan2(u[0]*v[1]-u[1]*v[0], u[0]*v[0]+u[1]*v[1])

    while l:
        v=l[-1]
        if angle(u,v)<=theta:
            m.append(u)
            u=v
            l.pop()
        else:
            l.append(u+v)
            superbases.append((u,v,-u-v))


    return np.array(superbases).transpose((2,1,0))

# ------- Three dimensional variant -------

# We do everyone in parallel, without selection or early abort
def _ObtuseSuperbase3(m,sb):
    """
        Use Selling's algorithm to compute an obtuse superbase.

        input : symmetric positive definite matrix m, dim=3
        input/output : superbase b (must be valid at startup)
        
        module variable : iterMax3, max number of iterations

        output : wether the algorithm succeeded
    """
    iterReducedMax = 6
    iter=0
    iterReduced = 0
    sigma = cycle([(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)])
    while iterReduced<iterReducedMax and iter<iterMax3:
        # Increment iterators
        iter += 1
        iterReduced += 1
        (i,j,k,l) = next(sigma)
        
        # Test for a positive angle, and modify superbase if necessary
        acute = dot_VV(sb[:,i],dot_AV(m,sb[:,j])) > 0
        if np.any(acute):
            sb[:,k,acute] += sb[:,i,acute]
            sb[:,l,acute] += sb[:,i,acute]
            sb[:,i,acute] = -sb[:,i,acute]
            iterReduced=0
    
    return iterReduced==iterReducedMax
    
def _Decomposition3(m,sb):
    """
        Use Selling's algorithm to decompose a tensor

        input : symmetric positive definite tensor, d=3
        output : coefficients, offsets
    """
    _,bounds = _GetDimBounds(m)
    
    coef=np.zeros((6,)+bounds)
    offset=np.zeros((3,6,)+bounds)
    for iter,(i,j,k,l) in zip(range(6),
    [(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)]):
        coef[iter] = -dot_VV(sb[:,i], dot_AV(m, sb[:,j]) )
        offset[:,iter] = cross(sb[:,k], sb[:,l])
        
    return coef,offset.astype(int)

def _SuperbasesForConditioning3(cond):
    raise ValueError("Sorry, _SuperbasesForConditioning3 is not implemented yet")