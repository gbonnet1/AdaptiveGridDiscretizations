import numpy as np
import os
import time

from . import misc
from . import kernel_traits
from . import solvers
from .. import Grid


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
	traits = kernel_traits.default_traits(model)
	traits.update(misc.GetValue(hfmIn,'traits',hfmOut,default={},
		help="Optional traits parameters passed to kernel"))
	float_t = np.dtype(traits['Scalar']).type
	int_t   = np.dtype(traits['Int']   ).type

	# ------- Format the geometrical data --------
	if verbosity>=1: print("Prepating the domain data (shape,metric,...)")
	shape = misc.GetValue(hfmIn,'dims',hfmOut,
		help="dimensions (shape) of the computational domain").astype(int)
	shape_i = traits['shape_i']
	shape_o = tuple(misc.round_up(shape,shape_i))
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
		values[tuple(seedIndices.T)] = seedValues
	else: 
		raise ValueError("Positive seedRadius not supported yet")
	block_values = misc.block_expand(values,shape_i,mode='constant',constant_values=xp.nan)

	# Tag the seeds
	block_seedTags = xp.isfinite(block_values).reshape( shape_o + (-1,) )
	block_seedTags = misc.packbits(block_seedTags,bitorder='little')
	block_values[xp.isnan(block_values)] = xp.inf
	
	# -------- Prepare the GPU kernel ---------
	source = kernel_traits.kernel_source(model,traits)
	kernel = misc.GetValue(hfmIn,'kernel',hfmOut,default="None",
		help="Saved GPU Kernel from a previous run, to bypass compilation")
	if kernel == "None":
		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = max(os.path.getmtime(os.path.join(cuda_path,file)) 
			for file in os.listdir(cuda_path))
		source += f"// Date cuda code last modified : {date_modified}\n"
		cuoptions = ("-default-device", f"-I {cuda_path}"
			) + misc.GetValue(hfmIn,'cuoptions',hfmOut,default=tuple(),
			help="Options passed via cupy.RawKernel to the cuda compiler")
		import cupy
		kernel = cupy.RawKernel(source,'IsotropicUpdate',options=cuoptions)
	else:
		assert kernel=="dummy" or kernel.source == source


	# Setup and run the eikonal solver
	solver = misc.GetValue(hfmIn,'solver',hfmOut,
		help="Choice of fixed point solver")
	nitermax_o = misc.GetValue(hfmIn,'nitermax_o',hfmOut,default=500,
		help="Maximum number of iterations of the solver")
	tol = float_t(misc.GetValue(hfmIn,'tol',hfmOut,1e-8,
		help="Convergence tolerance for the fixed point solver"))

	in_raw = {
	'block_values':block_values,
	'block_metric':block_metric,
	'block_seedTags':block_seedTags,
	'kernel':kernel,
	'source':source
	}
	if returns=='in_raw': return {'in_raw':in_raw,'hfmOut':hfmOut}

	data_t = (float_t,int_t)
	shapes_io = (shape_i,shape_o)
	kernel_args = (block_values,block_metric,block_seedTags,tol)
	kernel_args = tuple(arg if isinstance(arg,xp.ndarray) else 
		xp.array(arg,kernel_traits.dtype(arg,data_t)) for arg in kernel_args)

	if verbosity>=1: print("Running the solver")
	solver_start_time = time.time()

	if solver=='global_iteration':
		niter_o = solvers.global_iteration(tol,nitermax_o,data_t,shapes_io,
			kernel_args,kernel,hfmOut)
	elif solver in ('AGSI','adaptive_gauss_siedel_iteration'):
		niter_o = solvers.adaptive_gauss_siedel_iteration(
			tol,nitermax_o,data_t,shapes_io,
			kernel_args,kernel,hfmOut)
	else:
		raise ValueError(f"Unrecognized solver : {solver}")

	hfmOut.update({
		'niter_o':niter_o,
		'solverGPUTime':time.time() - solver_start_time,
		})
	if verbosity>=1: print("Post-Processing")
	if niter_o>=nitermax_o:
		nonconv_msg = (f"Solver {solver} did not reach convergence after "
			f"maximum allowed number {niter_o} of iterations")
		if misc.GetValue(hfmIn,'raiseOnNonConvergence',hfmOut,default=True):
			raise ValueError(nonconv_msg)
		else:
			print("---- Warning ----\n",nonconv_msg,"\n-----------------\n")

	out_raw = {
	'block_values':block_values
	}
	if returns=='out_raw': return {'out_raw':out_raw,'in_raw':in_raw,'hfmOut':hfmOut}

	values = misc.block_squeeze(block_values,shape)
	hfmOut['values'] = values

	return hfmOut



	#(u,cost,seeds,shape,x_o,min_chg,tol)

