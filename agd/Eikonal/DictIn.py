import numpy as np
from collections.abc import MutableMapping
from .. import AutomaticDifferentiation as ad
from . import run_detail
from .. import Metrics

_array_float_fields = {
	'origin','dims','gridScale','gridScales','values',
	'seeds','seeds_Unoriented','tips','tips_Unoriented',
	'seedValues','seedValues_Unoriented','seedValueVariation',
	'cost','speed','costVariation',
	'inspectSensitivity','inspectSensitivityWeights','inspectSensitivityLengths',
	'exportVoronoiFlags'
}

# Alternative key for setting or getting a single element
_singleIn = {
	'seed':'seeds','seedValue':'seedValues',
	'seed_Unoriented':'seeds_Unoriented','seedValue_Unoriented':'seedValues_Unoriented',
	'tip':'tips','tip_Unoriented':'tips_Unoriented',
	'seedFlag':'seedFlags','seedFlag_Unoriented':'seedFlags_Unoriented',
}

_singleOut = {
	'geodesic':'geodesics','geodesic_Unoriented':'geodesics_Unoriented',
	'geodesic_euclideanLength':'geodesics_euclideanLength',
}

SEModels = {'ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2',
'ReedsSheppExt2','ReedsSheppForwardExt2','ElasticaExt2','DubinsExt2',
'ReedsShepp3','ReedsSheppForward3'}

class dictOut(MutableMapping):
	"""
	A dictionnary like structure used as output of the Eikonal solvers. 
	"""

	def __init__(self,store=None):
		self.store=store

	def __copy__(self): return dictOut(self.store.copy())
	def copy(self): return self.__copy__()

	def __repr__(self):
		return f"dictOut({self.store})"

	def __setitem__(self, key, value):
		if key in _singleOut: key = _singleIn[key]; value = [value]
		self.store[key] = value

	def __getitem__(self, key): 
		if key in _singleOut:
			values = self.store[_singleOut[key]]
			if len(values)!=1: 
				raise ValueError(f"Found {len(values)} values for key {key}")
			return values[0]
		return self.store[key]

	def __delitem__(self, key): 
		key = _singleOut.get(key,key)
		del self.store[key]

	def __iter__(self): return iter(self.store)
	def __len__(self):  return len(self.store)


def CenteredLinspace(a,b,n):
	"""
	Returns a linspace shifted by half a node length.
	Inputs : 
	 - a,b : interval endpoints
	 - n : number of points
	"""
	n_=int(n); assert(n==n_) #Allow floats for convenience
	r,dr=np.linspace(a,b,n_,endpoint=False,retstep=True)
	return r+dr/2

class dictIn(MutableMapping):
	"""
	A dictionary like structure used as input of the Eikonal solvers.
	"""

	def __init__(self, store=None, mode='cpu', float_t=None):
		
		assert mode in ('cpu','cpu_raw','gpu','gpu_transfer')
		self.mode = mode
		if self.mode in ('gpu','gpu_transfer'):
			import cupy as cp
			self.xp = cp
			if float_t is None: float_t = np.float32
		else: 
			self.xp = np

		self.float_t = np.float64 if float_t is None else float_t
		self.array_float_caster = lambda x : self.xp.array(x,dtype=float_t)
		self.store = {'arrayOrdering':'RowMajor'}
		if store: self.update(store)

	def __copy__(self): return dictIn(self.store.copy(),self.mode,self.float_t)
	def copy(self): return self.__copy__()

	def __repr__(self): 
		return f"dictIn({self.store},mode={self.mode},float_t={self.float_t})"

	def __setitem__(self, key, value):
		if key in _singleIn: 
			key = _singleIn[key]; value = [value]
		if key in _array_float_fields and not ad.isndarray(value):
			value = self.array_float_caster(value)
		self.store[key] = value

	def __getitem__(self, key): 
		if key in _singleIn:
			values = self.store[_singleIn[key]]
			if len(values)!=1: 
				raise ValueError(f"Found {len(values)} values for key {key}")
			return values[0]
		return self.store[key]

	def __delitem__(self, key): 
		key = _singleIn.get(key,key)
		del self.store[key]

	def __iter__(self): return iter(self.store)
	def __len__(self):  return len(self.store)

	def Run(self,**kwargs):
		if self['arrayOrdering']!='RowMajor': 
			raise ValueError("Unsupported array ordering")

		if   self.mode=='cpu':     return dictOut(run_detail.RunSmart(self,**kwargs))
		elif self.mode=='cpu_raw': return dictOut(run_detail.RunRaw(self,**kwargs))
		
		from . import HFM_CUDA
		if   self.mode=='gpu':       return dictOut(HFM_CUDA.RunGPU(self,**kwargs))
		elif self.mode=='gpu_transfer':
			gpuIn = ad.cupy_generic.cupy_set(self, # host->device
				dtype32=self.float_t==np.float32, iterables=(dictIn,Metrics.Base))
			gpuOut = HFM_CUDA.RunGPU(gpuIn)
			cpuOut = ad.cupy_generic.cupy_get(gpuOut,iterables=(dict,list))
			return dictOut(cpuOut) # device->host

	# ------- Grid related functions ------

	@property
	def shape(self):
		return tuple(self['dims'].astype(int))

	@property
	def SE(self):
		"""Wether the model is based on the Special Euclidean group"""
		return self['model'] in SEModels	
	
	@property
	def vdim(self):
		"""Dimension of the ambient vector space"""
		dim = int(self['model'][-1])
		return (2*dim-1) if self.SE else dim

	@property
	def nTheta(self):
		"""
		Number of points for discretizing the interval [0,2 pi] for SE2 and SE3 models
		"""
		if not self.SE: raise ValueError("Not an SE model")
		shape = self.shape
		if self.vdim!=len(self.shape): raise ValueError("Angular resolution not set")
		n = shape[-1]
		return (2*n) if self.get('projective',False) else n

	@nTheta.setter
	def nTheta(self,value):
		if not self.SE: raise ValueError("Not an SE model")
		dims = self['dims']
		vdim = self.vdim
		projective = self.get('projective',False)
		if vdim==len(dims): dims=dims[:int((vdim+1)/2)] #raise ValueError("Angular resolution already set")
		if   vdim==3: self['dims'] = np.append(dims,value/2 if projective else value)
		elif vdim==5: self['dims'] = np.concatenate((dims,
			[value/4 if projective  else value/2, value]))

	@property
	def gridScales(self):
		if self.SE:
			h = self['gridScale']
			hTheta = 2.*np.pi / self.nTheta
			if self.vdim==3: return self.array_float_caster( (h,h,hTheta) )
			else: return self.array_float_caster( (h,h,h,hTheta,hTheta) )
		elif 'gridScales' in self: return self['gridScales']
		else: return self.array_float_caster((self['gridScale'],)*self.vdim)	
		
	@property
	def corners(self):
		"""Base point Grid()[0,..,0] for non-SE models, base = origin+gridScales/2"""
		dims = self['dims']
		origin = self.get('origin',self.xp.zeros_like(dims))
		gridScales = self.gridScales
		if self.SE: 
			tail = (-gridScales[-1]/2,)*(len(dims)-len(origin))
			origin = np.concatenate((origin,tail))
		return (origin,origin+gridScales*dims)

	def Axes(self,dims=None):
		"""Axes of the cartesian discretization grid of the domain"""
		bottom,top = self.corners
		if dims is None: dims=self['dims']
		return [self.array_float_caster(CenteredLinspace(b,t,d)) 
			for b,t,d in zip(bottom,top,dims)]

	def Grid(self,dims=None):
		"""
		Returns a grid of coordinates, containing all the discretization points of the domain.
		- dims(optional) : use a different sampling of the domain
		"""
		axes = self.Axes(dims);
		ordering = self['arrayOrdering']
		if ordering=='RowMajor': return ad.array(np.meshgrid(*axes,indexing='ij',copy=False))
		elif ordering=='YXZ_RowMajor': return ad.array(np.meshgrid(*axes,copy=False))
		else: raise ValueError('Unsupported arrayOrdering : '+ordering)

	def SetUniformTips(self,dims):
		"""Regularly spaced sample points in the domain"""
		self['tips'] = self.Grid(dims).reshape(self.vdim,-1).T

	def SetRect(self,sides,sampleBoundary=False,gridScale=None,gridScales=None,dimx=None,dims=None):
		"""
		Defines a box domain, for the HFM library.
		Inputs.
		- sides, e.g. ((a,b),(c,d),(e,f)) for the domain [a,b]x[c,d]x[e,f]
		- sampleBoundary : switch between sampling at the pixel centers, and sampling including the boundary
		- gridScale, gridScales : side h>0 of each pixel (alt : axis dependent)
		- dimx, dims : number of points along the first axis (alt : along all axes)
		"""
		if any(e in self for e in ('gridScale','gridScales','dims','origin')):
			raise ValueError("Domain already partially set")
		corner0,corner1 = np.asarray(sides,dtype=float).T
		dim = len(corner0)
		sb=float(sampleBoundary)
		width = np.asarray(corner1)-np.asarray(corner0)
		if gridScale is not None: 
			gridScales=[gridScale]*dim; self['gridScale']=gridScale
		elif gridScales is not None:
			self['gridScales']=gridScales
		elif dimx is not None:
			gridScale=width[0]/(dimx-sb); gridScales=[gridScale]*dim; self['gridScale']=gridScale
		elif dims is not None:
			gridScales=width/(np.asarray(dims)-sb); self['gridScales']=gridScales
		else: 
			raise ValueError('Missing argument gridScale, gridScales, dimx, or dims')

		h=gridScales
		ratios = [(M-m)/delta+sb for delta,m,M in zip(h,corner0,corner1)]
		dims = [round(r) for r in ratios]
		assert(np.min(dims)>0)
		origin = [c+(r-d-sb)*delta/2 for c,r,d,delta in zip(corner0,ratios,dims,h)]
		self['dims']   = np.asarray(dims)
		self['origin'] = np.asarray(origin)

	def PointFromIndex(self,index,to=False):
		"""
		Turns an index into a point.
		Optional argument to: if true, inverse transformation, turning a point into a continuous index
		"""
		bottom,_ = self.corners
		scale = self.gridScales
		start = bottom +0.5*scale
		index = self.array_float_caster(index)
		if not to: return start+scale*index
		else: return (index-start)/scale

	def IndexFromPoint(self,point):
		"""
		Returns the index that yields the position closest to a point, and the error.
		"""
		point = self.array_float_caster(point)
		continuousIndex = self.PointFromIndex(point,to=True)
		index = np.round(continuousIndex)
		return index.astype(int),(continuousIndex-index)

	def VectorFromOffset(self,offset,to=False):
		offset = self.array_float_caster(offset)
		if not to: return offset*self.gridScales
		else: return offset/self.gridScales  

	def GridNeighbors(self,point,gridRadius):
		"""
		Returns the neighbors around a point on the grid. 
		Geometry last convention
		Inputs: 
		- point (array): geometry last
		- gridRadius (scalar): given in pixels
		"""
		xp = self.xp
		point = self.array_float_caster(point)
		point_cindex = self.PointFromIndex(point,to=True)
		aX = [xp.arange(int(np.floor(ci-gridRadius)),int(np.ceil(ci+gridRadius)+1)) for ci in point_cindex]
		neigh_index =  xp.stack(xp.meshgrid( *aX, indexing='ij'),axis=-1)
		neigh_index = neigh_index.reshape(-1,neigh_index.shape[-1])

		# Check which neighbors are close enough
		offset = neigh_index-point_cindex
		close = np.sum(offset**2,axis=-1) < gridRadius**2

		# Check which neighbors are in the domain (periodicity omitted)
		neigh = self.PointFromIndex(neigh_index)
		bottom,top = self.corners
		inRange = np.all(np.logical_and(bottom<neigh,neigh<top),axis=-1)

		return neigh[np.logical_and(close,inRange)]
