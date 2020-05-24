#from ..Sparse import spAD
import cupy as cp

def simplify_ad(x,atol=0.,blockSize=256):
	"""Calls the GPU implementation of the simplify_ad method"""
	
	# Get the data
	coef,index = map(cp.ascontiguousarray,(x.coef,x.index))
	size_ad = x.size_ad
	if size_ad==0: return
	bound_ad = int(2**np.ceil(np.log2(size_ad)))

	# Set the traits
	int_t = np.int32
	size_t = int_t
	index_t = index.dtype.type
	scalar_t = coef.dtype.type
	traits = {
		'IndexT':index_t,
		'SizeT':size_t,
		'Scalar':scalar_t,
		'bound_ad':bound_ad,
	}

	# Setup the cupy kernel

	TODO

	SetModuleConstant('size_ad',x.size_ad,int_t)
	SetModuleConstant('atol',atol,scalar_t)
	SetModuleConstant('size_tot',x.size,size_t)

	# Call the kernel
	gridSize = int(np.ceil(x.size/blockSize))
	new_size_ad = cp.zeros(x.shape,dtype=np.int32)
	kernel((gridSize,),(blockSize,),(index,coef,new_size_ad))
	new_size_ad = np.max(new_size_ad)
	
	x.coef  = coef[...,:new_size_ad]
	x.index = index[...,:new_size_ad]

