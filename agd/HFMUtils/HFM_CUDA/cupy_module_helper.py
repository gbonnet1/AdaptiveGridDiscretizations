import os

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
	cp,has_rawmodule = _get_cupy()
	if has_rawmodule: return cp.RawModule(source,options=cuoptions)
	else: return cupy.core.core.compile_with_cache(self.source, 
		options=cuoptions, prepend_cupy_headers=False)


def SetModuleConstant(module,key,value,dtype):
	"""
	Sets a global constant in a cupy cuda module.
	"""
	cp,has_rawmodule = _get_cupy()
	value=cp.array(value,dtype=dtype)
	if has_rawmodule: 
		memptr = module.get_global(key)
	else: 
		#https://github.com/cupy/cupy/issues/1703
		b = xp.core.core.memory_module.BaseMemory()
		b.ptr = self.module.get_global_var(key)
		memptr = xp.cuda.MemoryPointer(b,0)

	module_constant = cp.ndarray(value.shape, value.dtype, memptr)
	module_constant[...] = value
