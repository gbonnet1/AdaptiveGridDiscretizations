# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import cupy_module_helper

def inf_convolution(arr,kernel,niter=1,periodic=None,
	upper_saturation=None, lower_saturation=None):
	"""
	Perform an inf convolution of an input with a given kernel, on the GPU.
	- arr : the input array
	- kernel : the convolution kernel. A centered kernel will be used.
	- niter (optional) : number of iterations of the convolution.
	- periodic (optional) : on which axes to use periodic boundary conditions.
	"""
	conv_t = arr.dtype.type
	int_t = np.int32

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)

	traits = {
		'T':conv_t,
		'shape_c':kernel.shape,
		}

	if saturation: traits['saturation_macro']=1
	if periodic is not None: traits['periodic'] = periodic

	source = cupy_module_helper.traits_header(traits,
		integral_max=True,size_of_shape=True,dtype_sup=True)

	source += [
	'#include "InfConvolution.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 

	module = GetModule(source,cuoptions)
	SetModuleConstant(module,'kernel_c',kernel,conv_t)
	SetModuleConstant(module,'shape_tot',arr.shape,int_t)
	SetModuleConstant(module,'size_tot',arr.size,int_t)




