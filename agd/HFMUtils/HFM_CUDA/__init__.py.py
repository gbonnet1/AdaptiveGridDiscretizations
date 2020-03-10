from . import misc
from .. import Grid
import numpy as np
import os

def default_traits(model):
	"""
	Default traits of the GPU implementation of an HFM model.
	"""
	traits = {
	'verbosity':0,
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

	ndim,niter,shape_i,nsym,nfwd = (traits[e] for e in 
		'ndim','niter','shape_i','nsym','nfwd')

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
		f"const Int ndim = {ndim}\n"
		f"const Int niter = {niter}\n"
		f"const Int nsym = {nsym}\n"
		f"const Int nfwd = {nfwd}\n"
		"const Int shape_i[ndim] = {"
		(f"{shape_i[0]}, {shape_i[1]}" if ndim==2 else 
		 f"{shape_i[0]}, {shape_i[1]}, {shape_i[2]}" )
		"}\n"
		f"const Int size_i = {size_i}\n"
		f"const Int log2_size_i = {log2_size_i}\n"
		'#include "cpp/Isotropic.h"'
		)

	return filename,source


def RunGPU(hfmIn,returns='out'):
	"""
	Runs a GPU accelerated fast marching - like method on the data.
	"""
	assert returns in ('in_raw','out_raw','out')
	_hfmIn = hfmIn # We pop the contents of hfmIn to find out unused keys.
	hfmIn = hfmIn.copy()
	model = hfmIn.pop('model')
	assert hfmIn.pop('arrayOrdering') == 'RowMajor'

	# Get kernel for running the method
	traits = default_traits(model).update( hfmIn.pop('traits',{}) )
	float_t = traits['float_t']
	_,source = format_traits(traits)
	kernel = hfmIn.pop('kernel',None)
	if kernel is None:
		cuoptions = (
			"-default-device",
			f"-I {os.path.realpath('cpp')}"
			)
		kernel = cp.RawKernel(source,'IsotropicUpdate',options=cuoptions)
	else:
		assert kernel.source == source

	# Format the metric data
	shape = hfmIn.pop('dims')
	shape_i = traits['shape_i']
	shape_o = misc.round_up(shape,shape_i)
	h = hfmIn.pop('gridScale')
	metric = hfmIn.pop('cost')
	assert metric.dtype == float_t
	xp = misc.get_array_module(metric)
	block_metric = misc.block_expand(metric*h,shape_i)

	# Prepare the values array
	values = xp.full(shape,xp.inf,dtype=float_t)
	seeds = hfmIn.pop('seeds')
	seedValues = hfmIn.pop('seedValues',xp.zeros(len(seeds),dtype=float_t))
	seedRadius = hfmIn.pop('seedRadius',0.)
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
	tol = float_t(hfmIn.pop('tol',1e-8))
	policy = hfmIn.pop('policy')
	if policy=='global':
		ax_o = tuple(xp.arange(s,dtype=int_t) for s in shape_o)
		x_o = xp.array(xp.meshgrid(*ax_o, indexing='ij'))
		x_o = xp.stack( xi_o.dla

	#(u,cost,seeds,shape,x_o,min_chg,tol)

