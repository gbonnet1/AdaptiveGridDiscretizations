from . import misc
import numpy as np
import os
from .. import Grid

def default_traits(model):
	"""
	Default traits of the GPU implementation of an HFM model.
	"""
	traits = {
	'debug_print':0,
	'float_t':cp.dtype('float32').type,
	'int_t':cp.dtype('int32').type,
	}

	if model=='Isotropic2':
		traits.update({
		'ndim':2,
		'niter':8,
		'shape_i':{8,8},
		'nsym':2,
		'nfwd':0,
		})
	elif model=='Isotropic3':
		traits.update({
		'ndim':3,
		'niter':8,
		'shape_i':{4,4,4},
		'nsym':3,
		'nfwd':0,
		})
	else:
		raise ValueError("Unsupported model")

	return traits

def format_traits(model,traits):
	"""
	Returns a suggested filename and source for the gpu code for the model.
	"""

	# TODO : support for float_t and int_t

	debug_print,ndim,niter,shape_i,nsym,nfwd = (traits[e] for e in 
		'debug_print','ndim','niter','shape_i','nsym','nfwd')

	assert ndim in {2,3}
	assert model in {'Isotropic2','Isotropic3'}
	filename = (
		f"ndim={ndim}_"
		f"niter={niter}_"
		f"shape_i={shape_i}_"
		f"nsym={nsym}_"
		f"nfwd={nfwd}_"
		".h"
		)

	size_i = np.prod(shape_i)
	log2_size_i = int(np.log2(size_i))
	source = (
		f"const Int debug_print = {debug_print};\n"
		f"const Int ndim = {ndim};\n"
		f"const Int niter = {niter};\n"
		f"const Int nsym = {nsym};\n"
		f"const Int nfwd = {nfwd};\n"
		"const Int shape_i[ndim] = {"
		",".join(str(s) for s in shape_i)
		"};\n"
		f"const Int size_i = {size_i}; // prod(shape_i)\n"
		f"const Int log2_size_i = {log2_size_i} // ceil(log2(size_i))\n"
		f"const Int nact = {nsym+nfwd}; // max active neighbors = nsym+nfwd\n"
		f"const Int ntot = {2*nsym+nfwd}; // total potential neighbors = 2*nsym+nfwd\n"
		'#include "cpp/Isotropic.h"'
		)

	return filename,source

def HasValue(dico,key,report):
	report['key visited'].append(key)
	return key in dico

def GetValue(dico,key,report,default=None,verbosity=2,help=None):
	"""
	Get a value from a dictionnary, printing some requested help.
	"""
	verb = report['verbosity']

	if key in report['help'] and key not in report['help content']:
		report['help content'][key] = help
		if verb>=1:
			if help is None: 
				print(f"Sorry : no help for key {key}")
			else:
				print(f"---- Help for key {key} ----")
				print(help)
				print("-----------------------------")

	if key in dico:
		report['key used'].append(key)
		return dico[key]
	elif default is not None:
		report['key defaulted'].append((key,value))
		if verb>=verbosity:
			print(f"key {key} defaults to {default}")
		return default
	else:
		raise ValueError("Missing value for key {key}")



def RunGPU(hfmIn,returns='out'):
	"""
	Runs a GPU accelerated eikonal solver method on the data.
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

	model = GetValue(hfmIn,'model',hfmOut,
		help='Minimal path model to be solved')
	assert hfmIn['arrayOrdering'] == 'RowMajor'

	verbosity = GetValue(hfmIn,'verbosity',hfmOut,default=1,
		"Choose the amount of detail displayed on the run")

	# Get kernel for running the method
	if verbosity>=1: print("Preparing the GPU kernel for (excludes compilation)")
	traits = default_traits(model).update( 
		GetValue(hfmIn,'traits',hfmOut,default={},
		help="Traits niter (iterations on each block) "
		"and shape_i (shape of a block) can be adjuster for performance")
	float_t = traits['float_t']
	_,source = format_traits(traits)
	kernel = hfmIn.pop('kernel',"None",hfmOut,
		help="Saved GPU Kernel from a previous run, to bypass compilation")
	if kernel == "None":
		cuoptions = (
			"-default-device",
			f"-I {os.path.realpath('./cpp')}"
			)
		kernel = cp.RawKernel(source,'IsotropicUpdate',options=cuoptions)
	else:
		assert kernel.source == source

	# ------- Format the geometrical data --------
	if verbosity>=1: print("Prepating the domain data (shape,metric,...)")
	shape = GetValue(hfmIn,'dims',hfmOut,
		help="dimensions (shape) of the computational domain")
	shape_i = traits['shape_i']
	shape_o = misc.round_up(shape,shape_i)
	h = GetValue(hfmIn,'gridScale',hfmOut,
		help="Scale of the computational grid")
	metric = GetValue(hfmIn,'cost',hfmOut,
		help="Cost function for the minimal paths")
	assert metric.dtype == float_t
	xp = misc.get_array_module(metric)
	block_metric = misc.block_expand(metric*h,shape_i)

	# Prepare the values array
	if verbosity>=1: print("Preparing the values array (seeing seeds,...)")
	values = xp.full(shape,xp.inf,dtype=float_t)
	seeds = GetValue(hfmIn,'seeds',hfmOut,
		help="Points from where the front propagation starts")
	seedValues = xp.zeros(len(seeds),dtype=float_t)
	seedValues = GetValue(hfmIn,'seedValues',hfmOut,
		help="Initial value for the front propagation")
	seedRadius = GetValue(hfmIn,'seedRadius',hfmOut,default=0.,
		help="Spreading the seeds over a few pixels can improve accuracy")
	if seedRadius==0.:
		seedIndices,_ = Grid.IndexFromPoint(_hfmIn,seeds)
		for index,value in zip(seedIndices,seedValues):
			values.__getitem__(index)=value
	else: 
		raise ValueError("Positive seedRadius not supported yet")
	block_values = misc.block_expand(values,shape_i)

	# Tag the seeds
	block_seedTags = block_values<cp.inf
	block_seedTags = block_seedTags.reshape( shape_o + (-1,) )
	block_seedTags = misc.packbits(block_seedTags,bitorder='little')
	
	# Get outer iterations policy
	solver = GetValue(hfmIn,'solver',hfmOut,
		help="Choice of fixed point solver")
	tol = float_t(GetValue(hfmIn,'tol',hfmOut,1e-8,
		help="Convergence tolerance for the fixed point solver"))
	if solver=='globalIteration':
		ax_o = tuple(xp.arange(s,dtype=int_t) for s in shape_o)
		x_o = xp.meshgrid(*ax_o, indexing='ij')
		x_o = xp.stack(x_o,axis=-1)
		min_chg = xp.full(x_o.shape[:-1],cp.inf,dtype=float_t)

		globalIterMax = GetValue(hfmIn,"GlobalIterationMax",hfmout,default=200,
			help="Maximum number of global iterations")
		for i in range(globalIterMax):
			kernel(u,cost,seeds,shape,x_o


	#(u,cost,seeds,shape,x_o,min_chg,tol)

