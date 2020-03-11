import sys

def get_array_module(*args):
	"""Returns the module (numpy or cupy) of an array"""
	assert len(args)>0
	for arg in args:
		module_name = type(arg).__module__
		if 'cupy' in module_name: 
			return sys.modules['cupy']
		elif 'numpy' not in module_name:
			raise ValueError(f"Module {module_name} not par of (numpy,cupy) for" 
				f" object with type {type(arg)} and value {arg}")
	return sys.modules['numpy']

#	import cupy # Alternative implementation requiring cupy import
#	return cupy.get_array_module(*args)

def samesize_int_t(float_t):
	"""Returns an integer type of the same size (32 or 64 bits) as a given float type"""
	float_name = str(float_t)
	if   'float32' in arg: return 'int32'
	elif 'float64' in arg: return 'float64'
	else: raise ValueError(
		f"Type {float_t} is not a float type, or has no default matching int type")