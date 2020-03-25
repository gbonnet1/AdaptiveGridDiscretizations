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

def traits_header(traits):
	"""
	Returns the source (mostly a preamble) for the gpu kernel code 
	for the given traits.
	"""
	source = ""
	for key,value in traits.items():
		if key.endswith('macro'):
			source += f"#define {key} {traits[key]}\n"
			continue
		else:
			source += f"#define {key}_macro\n"

		if isinstance(value,numbers.Integral):
			source += f"const int {key}={value};\n"
		elif isinstance(value,type):
			source += f"typedef {np2cuda_dtype[value]} {key};\n"
		elif all(isinstance(v,numbers.Integral) for v in value):
			source += (f"const int {key}[{len(value)}] = "
				+"{"+",".join(str(s).lower() for s in value)+ "};\n")
		else: 
			raise ValueError(f"Unsupported trait {key}:{value}")

	# Special treatment for some traits
	if 'Int' in traits:
		source += f"const Int Int_MAX = {np.iinfo(traits['Int']).max};\n"
	if "shape_i" in traits:
		size_i = np.prod(traits['shape_i'])
		log2_size_i = int(np.ceil(np.log2(size_i)))
		source += (f"const int size_i = {size_i};\n"
			+ f"const int log2_size_i = {log2_size_i};\n")

	return source