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
import functools

from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant
from ... import FiniteDifferences as fd





def InitStop(self,kernel_data):
	"""
	Returns the stopping criterion for the given kernel. 
	"""

	# Other kernels do not go through iterative solver
	assert any(kernel_data is self.kernel_data[key] 
		for key in ('eikonal', 'flow', 'forwardAD', 'reverseAD') )

	policy = kernel_data.policy

	#
	if policy.count_updates:
		nupdate_o = cp.zeros(self.shape_o,dtype=self.int_t)
		data.stats['nupdate_o']=nupdate_o

	allow_chart = kernel_data is not self.kernel_data['flow']
	chart_data = self.kernel_data['chart']
	nitermax_chart = chart_data.policy.niter_max if allow_chart else 0
	policy.niter_chart = 0

	if nitermax_chart>0:
		chart_kernel = chart_data.kernel
		chart_args = (chart_data.args['mapping'],kernel_data.args['values'])

		# Multi-precision needs a different kernel and arguments
		if policy.multiprecision: 
			assert kernel_data is self.kernel_data['eikonal']
			chart_kernel = chart_data.kernel_multip
			chart_args = (chart_data.args['mapping'],
				kernel_data.args['values'],kernel_data.args['valuesq'],
				kernel_data.args['values_Next'],kernel_data.args['valuesq_Next'])

		# TODO set some char_kernel constants, for 
		# forwardAD or reverseAD which need several values to be pasted at each point
		if self.forwardAD or self.reverseAD: raise ValueError("Not supported yet")

	# Stopping criterion depends on update_o, which is the list of blocks marked for update.
	def stop(update_o):
		# Track the number of updates if necessary
		if policy.count_updates: nupdate_o+=update_o

		# TODO : Stopping criterion based on accepted tips points ? 

		# Continue normally if any block is marked for update
		if np.any(update_o): return False

		# Apply boundary conditions if requested
		if policy.niter_chart<nitermax_chart:
			policy.niter_chart+=1
			chart_kernel((self.size_o,),(self.size_i,),chart_args+(update_o,))
			if np.any(update_o): return False

		return True

	return stop

#chart_help = """Use for a manifold defined by several local charts.
#Dictionary with members : 
# - mapping : mapping from one chart to the other. [Modified]
# - paste : where to paste values using the mapping, in the eikonal solver. (Useless)
# - jump : where paths should jump using the mapping, in the geodesic solver.
# - niter : number of calls to the eikonal solver.
#"""
def SetChart(self):
	"""
	Sets a pasting procedure for manifolds defined by several local charts.
	"""

	if self.HasValue('chart_mapping'): self.hasChart = True
	else: self.hasChart=False; return 

	# Import the chart arguments, check their type, cast if necessary
	mapping = self.GetValue('chart_mapping',default=None,
		help="Mapping from one local chart to another, "
		"for eikonal equations defined on manifolds.")

	chart_data = self.kernel_data['chart']
	policy = chart_data.policy

	policy.nitermax = self.GetValue('chart_nitermax',default=5,
		help="Number of times the boundary conditions are updated in the eikonal solver, "
		"with values from other local charts.")

	# Adimensionize the mapping
	shape_s = mapping.shape[1:]
	mapping -= fd.as_field(self.hfmIn['origin'],shape_s,depth=1)
	mapping /= fd.as_field(self.h,shape_s,depth=1)
	mapping -= 0.5
	chart_data.args['mapping'] = mapping

#	if self.HasValue(
#	paste = self.GetValue('chart_paste',default=None,
#	paste = cp.ascontiguousarray(cp.asarray(self.chart['paste'],dtype=bool))
#	chart_data.args['paste'] = paste

	ndim_s = len(shape_s)
	ndim_b = self.ndim-ndim_s
	if ndim_b<0 or self.shape[ndim_b:]!=shape_s or len(mapping)!=self.ndim:
		raise ValueError(f"Inconsistent shape of field chart['mapping'] : {mapping.shape}")
#	if paste.shape!=shape_s:
#		raise ValueError(f"Inconsistent shape of field chart['paste'] : {paste.shape}, expected {shape_s}")

	traits = {
		'Int':self.int_t,
		'Scalar':self.float_t,
		'ndim':self.ndim,
		'ndim_s':ndim_s,
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
	chart_data.kernel = module.get_function('Paste')

	modules = [module]

	if self.kernel_data['eikonal'].policy.multiprecision:
		module_multip = cupy_module_helper.GetModule(
			source+"#define multiprecision_macro 1\n",cuoptions)
		chart_data.kernel_multip = module_multip.get_function('Paste')
		SetModuleConstant(module_multip,'multip_step',self.multip_step,self.float_t)
		modules.append(module_multip)

	def SetCst(name,value,value_t):
		for mod in modules: SetModuleConstant(mod,name,value,value_t)

	SetCst('shape_tot',self.shape,self.int_t)
	SetCst('shape_i',self.shape_i,self.int_t)
	SetCst('shape_o',self.shape_o,self.int_t)
	SetCst('size_i',self.size_i,self.int_t)
	SetCst('size_s',np.prod(shape_s),self.int_t)



