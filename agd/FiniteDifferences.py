# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import itertools
from . import AutomaticDifferentiation as ad
import functools
import operator

def as_field(u,shape,conditional=True,depth=None):
	"""
	Checks if the last dimensions of u match the given shape. 
	If not, u is extended with these additional dimensions.
	conditional : if False, reshaping is always done
	depth (optional) : the depth of the geometrical tensor field (1: vectors, 2: matrices)
	"""
	u=ad.asarray(u)
	ndim = len(shape)
	def as_is():
		if not conditional: return False
		elif depth is None: return u.ndim>=ndim and u.shape[-ndim:]==shape
		else: assert u.shape[depth:] in (tuple(),shape); return u.shape[depth:]==shape
	if as_is(): return u
	else: return np.broadcast_to(u.reshape(u.shape+(1,)*ndim), u.shape+shape)

def common_field(arrays,depths,shape=None):
	if shape is None:
		assert len(arrays)==len(depths)
		for arr,depth in zip(arrays,depths):
			if arr is None: continue
			arr=ad.asarray(arr)
			shape = arr.shape[depth:]
			if shape!=tuple(): break
	return tuple(None if arr is None else as_field(arr,shape,depth=depth) 
		for (arr,depth) in zip(arrays,depths))

# ----- Utilities for finite differences ------

def BoundedSlices(slices,shape):
	"""
	Returns the input slices with None replace with the upper bound
	from the given shape
	"""
	if slices[-1]==Ellipsis:
		slices=slices[:-1]+(slice(None,None,None),)*(len(shape)-len(slices)+1)
	def BoundedSlice(s,n):
		if not isinstance(s,slice):
			return slice(s,s+1)
		else:
			return slice(s.start, n if s.stop is None else s.stop, s.step)
	return tuple(BoundedSlice(s,n) for s,n in zip(slices,shape))

def OffsetToIndex(shape,offset, mode='clip', uniform=None, where=(Ellipsis,)):
	"""
	Returns the index corresponding to position + offset, 
	and a boolean for wether it falls inside the domain.
	Set padding=None for periodic boundary conditions
	"""
	ndim = len(shape) # Domain dimension
	assert(offset.shape[0]==ndim)
	if uniform is None: # Uniform = True iff offsets are independent of the position in the domain
		uniform = not ((offset.ndim > ndim) and (offset.shape[-ndim:]==shape))

	odim = (offset.ndim-1) if uniform else (offset.ndim - ndim-1) # Dimensions for distinct offsets 
	everywhere = where==(Ellipsis,)

	grid = (np.mgrid[tuple(slice(n) for n in shape)]
		if everywhere else
		np.squeeze(np.mgrid[BoundedSlices(where,shape)],
		tuple(1+i for i,s in enumerate(where) if not isinstance(s,slice)) ) )
	grid = grid.reshape( (ndim,) + (1,)*odim+grid.shape[1:])

	if not everywhere and not uniform:
		offset = offset[(slice(None),)*(1+odim)+where]

	neigh = grid + (offset.reshape(offset.shape + (1,)*ndim) if uniform else offset)
	inside = np.full(neigh.shape[1:],True) # Weither neigh falls out the domain

	if mode=='wrap': # Apply periodic bc
		for coord,bound in zip(neigh,shape):
			coord %= bound
	else: #identify bad indices
		for coord,bound in zip(neigh,shape):
			inside = np.logical_and.reduce( (inside, coord>=0, coord<bound) )

	neighIndex = np.ravel_multi_index(neigh, shape, mode=mode)
	return neighIndex, inside

def TakeAtOffset(u,offset, padding=np.nan, **kwargs):
	mode = 'wrap' if padding is None else 'clip'
	neighIndex, inside = OffsetToIndex(u.shape,offset,mode=mode, **kwargs)

	values = u.flatten()[neighIndex]
	if padding is not None:
		if ad.isndarray(values):
			values[np.logical_not(inside)] = padding
		elif not inside:
			values = padding
	return values

def AlignedSum(u,offset,multiples,weights,**kwargs):
	"""Returns sum along the direction offset, with specified multiples and weights"""
	return sum(TakeAtOffset(u,mult*np.asarray(offset),**kwargs)*weight 
		for mult,weight in zip(multiples,weights))


# --------- Degenerate elliptic finite differences -------

def Diff2(u,offset,gridScale=1.,**kwargs):
	"""
	Approximates <offset, (d^2 u) offset> with second order accuracy.
	Second order finite difference in the specidied direction.
	"""
	return AlignedSum(u,offset,(1,0,-1),np.array((1,-2,1))/gridScale**2,**kwargs)


def DiffUpwind(u,offset,gridScale=1.,**kwargs):
	"""
	Approximates <d u, offset> with first order accuracy.
	Upwind first order finite difference in the specified direction.
	"""
	return AlignedSum(u,offset,(1,0),np.array((1,-1))/gridScale,**kwargs)


# --------- Non-Degenerate elliptic finite differences ---------

def DiffCentered(u,offset,gridScale=1.,**kwargs):
	"""
	Approximates <d u, offset> with second order accuracy.
	Centered first order finite difference in the specified direction.
	"""
	return AlignedSum(u,offset,(1,-1),np.array((1,-1))/(2*gridScale),**kwargs)

def DiffUpwind2(u,offset,gridScale=1.,**kwargs):
	"""
	Approximates <d u, offset> with second order accuracy.
	Upwind finite difference scheme, but lacking the degenerate ellipticity property.
	"""
	return AlignedSum(u,offset,(2,1,0),np.array((-0.5,2.,-1.5))/gridScale,**kwargs)

def DiffUpwind3(u,offset,gridScale=1.,**kwargs):
	"""
	Approximates <d u, offset> with third order accuracy.
	Upwind finite difference scheme, but lacking the degenerate ellipticity property.
	"""
	return AlignedSum(u,offset,(3,2,1,0),
		np.array((1./3.,-1.5,3.,-11./6.))/gridScale,**kwargs)

def DiffCross(u,offset0,offset1,gridScale=1.,**kwargs):
	"""
	Approximates <offsets0, (d^2 u) offset1> with second order accuracy.
	Centered finite differences scheme, but lacking the degenerate ellipticity property.
	"""
	weights = np.array((1,1))/(4*gridScale**2)
	return (AlignedSum(u,offset0+offset1,(1,-1),weights,**kwargs) 
		- AlignedSum(u,offset0-offset1,(1,-1),weights,**kwargs) )

# ------------ Composite finite differences ----------

def AxesOffsets(u=None,offsets=None,dimension=None):
	"""
	Returns the offsets corresponding to the axes.
		Inputs : 
	 - offsets (optional). Defaults to np.eye(dimension)
	 - dimension (optional). Defaults to u.ndim
	"""
	if offsets is None:
		if dimension is None:
			dimension = u.ndim
		offsets = np.eye(dimension).astype(int)
	return offsets



def DiffHessian(u,offsets=None,dimension=None,**kwargs):
	"""
	Approximates the matrix (<offsets[i], (d^2 u) offsets[j]> )_{ij}, using AxesOffsets as offsets.
	Centered and cross finite differences are used, thus lacking the degenerate ellipticity property.
	"""
	from . import Metrics
	offsets=AxesOffsets(u,offsets,dimension)
	return Metrics.misc.expand_symmetric_matrix([
		Diff2(u,offsets[i],**kwargs) if i==j else DiffCross(u,offsets[i],offsets[j],**kwargs) 
		for i in range(len(offsets)) for j in range(i+1)])

def DiffGradient(u,offsets=None,dimension=None,**kwargs):
	"""
	Approximates the vector (<d u, offsets[i]>)_i, using AxesOffsets as offsets
	Centered finite differences are used, thus lacking the degerate ellipticity property.
	"""
	return DiffCentered(u,AxesOffsets(u,offsets,dimension),**kwargs)

# ----------- Interpolation ---------

def UniformGridInterpolator1D(bounds,values,mode='clip',axis=-1):
	"""Interpolation on a uniform grid. mode is in ('clip','wrap', ('fill',fill_value) )"""
	
	val = values.swapaxes(axis,0)
	fill_value = None
	if isinstance(mode,tuple):
		mode,fill_value = mode		
	def interp(position):
		endpoint=not (mode=='wrap')
		size = val.size
		index_continuous = (size-int(endpoint))*(position-bounds[0])/(bounds[-1]-bounds[0])
		index0 = np.floor(index_continuous).astype(int)
		index1 = np.ceil(index_continuous).astype(int)
		index_rem = index_continuous-index0
		
		fill_indices=False
		if mode=='wrap':
			index0=index0%size
			index1=index1%size
		else: 
			if mode=='fill':
				 fill_indices = np.logical_or(index0<0, index1>=size)
			index0 = np.clip(index0,0,size-1) 
			index1 = np.clip(index1,0,size-1)
		
		index_rem = index_rem.reshape(index_rem.shape+(1,)*(val.ndim-1))
		result = val[index0]*(1.-index_rem) + val[index1]*index_rem
		if mode=='fill': result[fill_indices] = fill_value
		result = np.moveaxis(result,range(position.ndim),range(-position.ndim,0))
		return result
	return interp

def AxesOrderingBounds(grid):
	dim = len(grid)
	lbounds = grid.__getitem__((slice(None),)+(0,)*dim)
	ubounds = grid.__getitem__((slice(None),)+(-1,)*dim)

	def active(i):
		di = grid.__getitem__((slice(None),)+(0,)*i+(1,)+(0,)*(dim-1-i))
		return np.argmax(np.abs(di-lbounds))
	axes = tuple(active(i) for i in range(dim))

	return axes,lbounds,ubounds 