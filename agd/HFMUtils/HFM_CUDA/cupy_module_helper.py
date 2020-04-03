# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
import numbers

def _get_cupy():
	"""
	Returns the cupy python module, and wether it supports cuda RawModule.
	"""
	import cupy
	has_rawmodule = False # Version(cupy.__version__) >= Version("8") # Untested
	return cupy,has_rawmodule

def getmtime_max(directory):
	"""
	Lists all the files in the given directory, and returns the last time one of them
	was modified. Information needed when compiling cupy modules, because they are cached.
	"""
	return max(os.path.getmtime(os.path.join(directory,file)) 
		for file in os.listdir(directory))

def GetModule(source,cuoptions):
	"""Returns a cupy raw module"""
	cupy,has_rawmodule = _get_cupy()
	if has_rawmodule: return cupy.RawModule(source,options=cuoptions)
	else: return cupy.core.core.compile_with_cache(source, 
		options=cuoptions, prepend_cupy_headers=False)


def SetModuleConstant(module,key,value,dtype):
	"""
	Sets a global constant in a cupy cuda module.
	"""
	cupy,has_rawmodule = _get_cupy()
	if has_rawmodule: 
		memptr = module.get_global(key)
	else: 
		#https://github.com/cupy/cupy/issues/1703
		b = cupy.core.core.memory_module.BaseMemory()
		b.ptr = module.get_global_var(key)
		memptr = cupy.cuda.MemoryPointer(b,0)

	value=cupy.array(value,dtype=dtype)
	module_constant = cupy.ndarray(value.shape, value.dtype, memptr)
	module_constant[...] = value

# cuda does not have int8_t, int32_t, etc
np2cuda_dtype = {
	np.int8:'char',
	np.uint8:'unsigned char',
	np.int16:'short',
	np.int32:'int',
	np.int64:'long long',
	np.float32:'float',
	np.float64:'double',
	}

def traits_header(traits,
	join=False,dtype_sup=False,
	size_of_shape=False,log2_size=False):
	"""
	Returns the source (mostly a preamble) for the gpu kernel code 
	for the given traits.
	- join (optional): return a multiline string, rather than a list of strings
	- dtype_sup: insert a trait T_Sup (inf, or largest value) for each numerical type defined.
	- size_of_shape: insert traits for the size of each shape.
	- log2_size: insert a trait for the ceil of the base 2 logarithm of previous size.
	"""
	source = []
	for key,value in traits.items():
		if key.endswith('macro'):
			source.append(f"#define {key} {traits[key]}")
			continue
		else:
			source.append(f"#define {key}_macro")

		if isinstance(value,numbers.Integral):
			source.append(f"const int {key}={value};")
		elif isinstance(value,type):
			source.append(f"typedef {np2cuda_dtype[value]} {key};")
		elif all(isinstance(v,numbers.Integral) for v in value):
			source.append(f"const int {key}[{len(value)}] = "
				+"{"+",".join(str(s).lower() for s in value)+ "};")
		else: 
			raise ValueError(f"Unsupported trait {key}:{value}")

	# Special treatment for some traits
	for key,value in traits.items():
		if dtype_sup and isinstance(value,type):
			kind =  np.dtype(value).kind==
			if kind=='i':
				source.append(f"const {key} {key}_Sup = {np.iinfo(value).max};")
			elif kind=='f':
				source.append(f"const {key} {key}_Sup = 1./0.;")
		if size_of_shape and key.startswith('shape_'):
			suffix = key[len('shape_'):]
			size = np.prod(value)
			source.append(f"const int size_{suffix} = {size};")
			if log2_size:
				log2 = int(np.ceil(np.log2(size)))
				source.append(f"const int log2_size_{suffix} = {log2};")

	return "\n".join(source) if join else source



