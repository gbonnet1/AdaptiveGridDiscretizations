from . import misc
import numpy as np
import os
from .. import Grid


def default_traits(model):
	"""
	Default traits of the GPU implementation of an HFM model.
	"""
	traits = {
	'Scalar':'float32',
	'Int':   'int32',
	}

	if model=='Isotropic2':
		traits.update({
		'shape_i':(8,8),
		})
	elif model=='Isotropic3':
		traits.update({
		'shape_i':(4,4,4),
		})
	else:
		raise ValueError("Unsupported model")

	return traits

def kernel_source(model,traits):
	"""
	Returns the source (mostly a preamble) for the gpu kernel code 
	for the given traits and model.
	"""
	source = "".join(f"#define {key}_macro\n" for key in traits)
	traits = traits.copy()

	if 'shape_i' in traits:
		shape_i = traits.pop('shape_i')
		size_i = np.prod(shape_i)
		log2_size_i = int(np.ceil(np.log2(size_i)))
		source += (f"const int shape_i[{len(shape_i)}] = " 
			+ "{" +",".join(str(s) for s in shape_i)+ "};\n"
			+ f"const int size_i = {size_i};\n"
			+ f"const int log2_size_i = {log2_size_i};\n")

	if 'Scalar' in traits:
		Scalar = traits.pop('Scalar')
		if   'float32' in str(Scalar): ctype = 'float'
		elif 'float64' in str(Scalar): ctype = 'double'
		else: raise ValueError(f"Unrecognized scalar type {Scalar}")
		source += f"typedef Scalar {ctype};\n"

	if 'Int' in traits:
		Int = traits.pop('Int')
		if   'int32' in str(Int): ctype = 'int'
		elif 'int64' in str(Int): ctype = 'long long'
		else: raise ValueError(f"Unrecognized scalar type {Int}")
		source += f"typedef Int {ctype};\n"

	for key,value in traits.items():
		source += f"const int {key}={value};\n"

	source += f'#include "{model}.h"\n'
	return source

def RunGPU(hfmIn,returns='out'):
	"""
	Runs a GPU accelerated eikonal solver.
	"""
	assert returns in ('in_raw','out_raw','out')
	hfmOut = {
	'key used':[],
	'key defaulted':[],
	'key visited':[],
	'help':hfmIn.get('help',[]),
	'help content':{},
	'verbosity':hfmIn.get('verbosity',1)
	}

	model = misc.GetValue(hfmIn,'model',hfmOut,
		help='Minimal path model to be solved')
	assert hfmIn['arrayOrdering'] == 'RowMajor'

	verbosity = misc.GetValue(hfmIn,'verbosity',hfmOut,default=1,
		help="Choose the amount of detail displayed on the run")

	# ---- Get traits ----
	if verbosity>=1: print("Preparing the GPU kernel for (excludes compilation)")
	traits = default_traits(model)
	traits.update(misc.GetValue(hfmIn,'traits',hfmOut,default={},
		help="Optional traits parameters passed to kernel"))
	float_t = np.dtype(traits['Scalar']).type
	int_t   = np.dtype(traits['Int']   ).type

	# ------- Format the geometrical data --------
	if verbosity>=1: print("Prepating the domain data (shape,metric,...)")
	shape = misc.GetValue(hfmIn,'dims',hfmOut,
		help="dimensions (shape) of the computational domain").astype(int)
	shape_i = traits['shape_i']
	shape_o = misc.round_up(shape,shape_i)
	h = misc.GetValue(hfmIn,'gridScale',hfmOut,
		help="Scale of the computational grid")
	metric = misc.GetValue(hfmIn,'cost',hfmOut,
		help="Cost function for the minimal paths")
	assert metric.dtype == float_t
	xp = misc.get_array_module(metric)
	block_metric = misc.block_expand(metric*h,shape_i,mode='constant',constant_values=xp.inf)

	# Prepare the values array
	if verbosity>=1: print("Preparing the values array (setting seeds,...)")
	values = xp.full(shape,xp.inf,dtype=float_t)
	seeds = misc.GetValue(hfmIn,'seeds',hfmOut,
		help="Points from where the front propagation starts")
	seedValues = xp.zeros(len(seeds),dtype=float_t)
	seedValues = misc.GetValue(hfmIn,'seedValues',hfmOut,default=seedValues,
		help="Initial value for the front propagation")
	seedRadius = misc.GetValue(hfmIn,'seedRadius',hfmOut,default=0.,
		help="Spreading the seeds over a few pixels can improve accuracy")
	if seedRadius==0.:
		seedIndices,_ = Grid.IndexFromPoint(hfmIn,seeds)
		print(seedIndices)
		values[tuple(seedIndices.T)] = seedValues
	else: 
		raise ValueError("Positive seedRadius not supported yet")
	block_values = misc.block_expand(values,shape_i,mode='constant',constant_values=xp.inf)

	# Tag the seeds
	block_seedTags = block_values<float_t(np.inf)
#	print(f"{block_seedTags.shape=},{shape_o=}")
	block_seedTags = block_seedTags.reshape( tuple(shape_o) + (-1,) )
	block_seedTags = misc.packbits(block_seedTags,bitorder='little')
	
	# -------- Prepare the GPU kernel ---------
	source = kernel_source(model,traits)
	kernel = misc.GetValue(hfmIn,'kernel',hfmOut,default="None",
		help="Saved GPU Kernel from a previous run, to bypass compilation")
	if kernel == "None":
		cuoptions = (
			"-default-device",
			f"-I {os.path.realpath('./cpp')}"
			)
		import cupy
		kernel = cupy.RawKernel(source,'IsotropicUpdate',options=cuoptions)
	else:
		assert kernel=="dummy" or kernel.source == source


	# Setup and run the eikonal solver
	solver = misc.GetValue(hfmIn,'solver',hfmOut,
		help="Choice of fixed point solver")
	solverMaxIter = misc.GetValue(hfmIn,'solverMaxIter',hfmOut,default=500,
		help="Maximum number of iterations for the solver")
	tol = float_t(misc.GetValue(hfmIn,'tol',hfmOut,1e-8,
		help="Convergence tolerance for the fixed point solver"))

	if returns=='in_raw':
		return {
		'block_values':block_values,
		'block_metric':block_metric,
		'block_seedTags':block_seedTags,
		'kernel':kernel,
		'source':source,
		'hfmOut':hfmOut
		}

	if solver=='globalIteration':
		ax_o = tuple(xp.arange(s,dtype=int_t) for s in shape_o)
		x_o = xp.meshgrid(*ax_o, indexing='ij')
		x_o = xp.stack(x_o,axis=-1)
		min_chg = xp.full(x_o.shape[:-1],np.inf,dtype=float_t)

#		print(f"{x_o.flatten()=},{min_chg=}")

		for i in range(solverMaxIter):
			kernel(block_values,block_metric,block_seedTags,shape,x_o,min_chg,tol)
			if np.all(np.isinf(min_chg)):
				break
		else:
			raise ValueError(f"Solver {solver} did not reach convergence after "
				f"{solverMaxIter} iterations")


	#(u,cost,seeds,shape,x_o,min_chg,tol)

