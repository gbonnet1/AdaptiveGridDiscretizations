# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file is used to solve eikonal equations on domains defined by several local charts,
glued along a band along their boundary.

It works by gluing the appropriate values, selecting the smallest ones each time, 
and calling again the solver, a prescribed number of times.
"""
import os
import cupy as cp
import numpy as np

from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant
from ... import FiniteDifferences as fd

chart_help = """Use for a manifold defined by several local charts.
Dictionary with members : 
 - mapping : mapping from one chart to the other. [Modified]
 - paste : where to paste values using the mapping, in the eikonal solver.
 - jump : where paths should jump using the mapping, in the geodesic solver.
 - neik : number of calls to the eikonal solver.
"""
def ChartGlue(self):
	# Import the chart arguments, check their type, cast if necessary
	self.chart = self.GetValue('chart',default=None,help=chart_help)
	if self.chart is None: return
	
	neik = self.chart['niter']
#	if neik<=1: return

	# Adimensionize the mapping
	mapping = cp.ascontiguousarray(cp.asarray(self.chart['mapping'],dtype=self.float_t))
	shape_s = mapping.shape[1:]
	mapping -= fd.as_field(self.hfmIn['origin'],shape_s,depth=1)
	mapping /= fd.as_field(self.h,shape_s,depth=1)
	mapping -= 0.5

	self.chart['mapping'] = mapping # Cast useful in geodesic solver also
	paste = cp.ascontiguousarray(cp.asarray(self.chart['paste'],dtype=bool))

	ndim_s = len(shape_s)
	ndim_b = self.ndim-ndim_s
	if ndim_b<0 or self.shape[ndim_b:]!=shape_s or len(mapping)!=self.ndim:
		raise ValueError(f"Inconsistent shape of field chart['mapping'] : {mapping.shape}")
	if paste.shape!=shape_s:
		raise ValueError(f"Inconsistent shape of field chart['paste'] : {paste.shape}, expected {shape_s}")

	# Prepare the kernel
	eikonal = self.kernel_data['eikonal']
	multip = eikonal.policy.multiprecision

	traits = {
		'Int':self.int_t,
		'Scalar':self.float_t,
		'ndim':self.ndim,
		'ndim_s':ndim_s,
		'multiprecision_macro':multip,
	}
	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits)

	source += [
	'#include "Paste.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 
	source="\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)

	SetModuleConstant(module,'shape_tot',self.shape,self.int_t)
	SetModuleConstant(module,'shape_i',self.shape_i,self.int_t)
	SetModuleConstant(module,'shape_o',self.shape_o,self.int_t)
	SetModuleConstant(module,'size_i',self.size_i,self.int_t)
	SetModuleConstant(module,'size_s',np.prod(shape_s),self.int_t)
	if multip: SetModuleConstant(multip,'multip_step',self.multip_step,self.float_t)

	# Trigger not adequate : ill shaped, and must be preserved for geodesics, etc
	args = (eikonal.args['values'],eikonal.trigger,mapping,paste) 
	if multip: args = (args[:1]+(eikonal.args['valuesq'],eikonal.args['valuesNext'],
		eikonal.args['valuesqNext'])+args[1:])
	kernel = module.get_function('Paste')
	
	print("vals",args[0].shape,args[0].dtype)
	print("trigger",args[1].shape,args[1].dtype)
	print("mapping",args[2].shape,args[2].dtype)
	print("paste",args[3].shape,args[3].dtype)

	for i in range(neik-1):
		kernel((self.size_o,),(self.size_i,),args)
		self.Solve("eikonal")



