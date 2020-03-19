import numpy as np
from collections import namedtuple

SEModels = {'ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2',
'ReedsSheppExt2','ReedsSheppForwardExt2','ElasticaExt2','DubinsExt2',
'ReedsShepp3','ReedsSheppForward3'}


def GetCorners(params):
	dims = params['dims']
	dim = len(dims)
	h = params['gridScales'] if 'gridScales' in params.keys() else [params['gridScale']]*dim
	origin = params['origin'] if 'origin' in params.keys() else [0.]*dim
	if params['model'] in SEModels:
		origin = np.append(origin,[0]*(dim-len(origin)))		
		hTheta = 2*np.pi/dims[-1]
		h[-1]=hTheta; origin[-1]=-hTheta/2;
		if dim==5: h[-2]=hTheta; origin[-2]=-hTheta/2;
	return [origin,origin+h*dims]

def CenteredLinspace(a,b,n):
	n_=int(n); assert(n==n_) #Allow floats for convenience
	r,dr=np.linspace(a,b,n_,endpoint=False,retstep=True)
	return r+dr/2

def GetAxes(params,dims=None):
	bottom,top = GetCorners(params)
	if dims is None: dims=params['dims']
	return [CenteredLinspace(b,t,d) for b,t,d in zip(bottom,top,dims)]


def GetGrid(params,dims=None):
	"""Returns a grid of coordinates for the domain.
	Calls np.meshgrid, converts to ndarray"""
	axes = GetAxes(params,dims);
	ordering = params['arrayOrdering']
	if ordering=='RowMajor':
		return np.array(np.meshgrid(*axes,indexing='ij'))
	elif ordering=='YXZ_RowMajor':
		return np.array(np.meshgrid(*axes))
	else: 
		raise ValueError('Unsupported arrayOrdering : '+ordering)

def Rect(sides,sampleBoundary=False,gridScale=None,gridScales=None,dimx=None,dims=None):
	"""
	Defines a box domain, for the HFM library.
	Inputs.
	- sides, e.g. ((a,b),(c,d),(e,f)) for the domain [a,b]x[c,d]x[e,f]
	- sampleBoundary : switch between sampling at the pixel centers, and sampling including the boundary
	- gridScale, gridScales : side h>0 of each pixel (alt : axis dependent)
	- dimx, dims : number of points along the first axis (alt : along all axes)
	"""
	corner0,corner1 = np.array(sides,dtype=float).T
	dim = len(corner0)
	sb=float(sampleBoundary)
	result=dict()
	width = np.array(corner1)-np.array(corner0)
	if gridScale is not None:
		gridScales=[gridScale]*dim; result['gridScale']=gridScale
	elif gridScales is not None:
		result['gridScales']=gridScales
	elif dimx is not None:
		gridScale=width[0]/(dimx-sb); gridScales=[gridScale]*dim; result['gridScale']=gridScale
	elif dims is not None:
		gridScales=width/(np.array(dims)-sb); result['gridScales']=gridScales
	else: 
		raise ValueError('Missing argument gridScale, gridScales, dimx, or dims')

	h=gridScales
	ratios = [(M-m)/delta+sb for delta,m,M in zip(h,corner0,corner1)]
	dims = [round(r) for r in ratios]
	assert(np.min(dims)>0)
	origin = [c+(r-d-sb)*delta/2 for c,r,d,delta in zip(corner0,ratios,dims,h)]
	result.update({'dims':np.array(dims),'origin':np.array(origin)});
	return result


# -------------- Point to and from index --------------

GridSpec = namedtuple('GridSpec',['bottom','scale','dims'])

def to_YXZ(params,index):
	assert params['arrayOrdering'] in ('RowMajor','YXZ_RowMajor')
	if params['arrayOrdering']=='RowMajor':
		return index
	else:
		return np.stack((index[:,1],index[:,0],*index[:,2:].T),axis=1)

def GetGridSpec(params):
	"""
	Returns the bottom point, scale and dimensions of the grid.
	"""
	bottom,top = GetCorners(params)
	dims=np.array(params['dims'])
	scale = (top-bottom)/dims
	return GridSpec(bottom,scale,dims)


def PointFromIndex(params,index,to=False):
	"""
	Turns an index into a point.
	Optional argument to: if true, inverse transformation, turning a point into a continuous index
	"""
	bottom,scale,_ = GetGridSpec(params)
	start = bottom +0.5*scale
	if not to: return start+scale*to_YXZ(params,index)
	else: return to_YXZ(params,(index-start)/scale)

def IndexFromPoint(params,point):
	"""
	Returns the index that yields the position closest to a point, and the error.
	"""
	continuousIndex = PointFromIndex(params,point,to=True)
	index = np.round(continuousIndex)
	return index.astype(int),(continuousIndex-index)

def VectorFromOffset(params,offset,to=False):
	assert params['arrayOrdering']=='RowMajor'
	scale = Scale(params)
	if not to: return scale*offset
	else: return offset/scale  

def GridNeighbors(params,point,gridRadius):
	"""
	Returns the neighbors around a point on the grid. 
	Geometry last convention
	Inputs: 
	- params (dict): hfmIn
	- point (array): geometry last
	- gridRadius (scalar): given in pixels
	"""
	assert params['arrayOrdering']=='RowMajor'
	cindex = PointFromIndex(params,point)
	neigh =  np.meshgrid( (np.arange(-np.floor(ci+seedRadius),np.ceil(ci+seedRadius)+1)
				for ci in cindex) , indexing='ij')
	neigh = np.stack(neigh,axis=-1).reshape(-1,neigh.shape[-1])
	diff = neigh-point
	close = np.sum(diff**2,axis=-1) < gridRadius**2
	return neigh[close]
