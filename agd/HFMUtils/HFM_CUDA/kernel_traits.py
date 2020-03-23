import numpy as np
import numbers

"""
def dtype(arg,data_t):
	"
	For a numeric array, returns dtype.
	Otherwise, returns one of the provided floating point 
	or integer data type, depending on the argument data type.
	Inputs:
	 - data_t (tuple) : (float_t,int_t)
	"
	float_t,int_t = data_t
	if isinstance(arg,numbers.Real): 
		return float_t
	elif isinstance(arg,numbers.Integral):
		return int_t
	elif isinstance(arg,(tuple,list)):
		return dtype(arg[0],data_t)
	else:
		return arg.dtype
"""

def default_traits(interface):
	"""
	Default traits of the GPU implementation of an HFM model.
	"""
	traits = {
	'Scalar':  'float32',
	'Int':     'int32',
	'multiprecision_macro':0,
	}

	ndim = interface.ndim

	if ndim==2:
		traits.update({
		'shape_i':(24,24),
		'niter_i':48,
		})
	elif ndim:
		traits.update({
		'shape_i':(4,4,4),
		'niter_i':12,
		})
	else:
		raise ValueError("Unsupported model")

	return traits

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

def kernel_source(traits):
	"""
	Returns the source (mostly a preamble) for the gpu kernel code 
	for the given traits.
	"""
	source = ""
	for key in list(traits.keys()):
		if 'macro' in key:
			source += f"#define {key} {traits[key]}\n"
			traits.pop(key)
		else:
			source += f"#define {key}_macro\n"

	"""
	if 'shape_i' in traits:
		shape_i = traits.pop('shape_i')
		size_i = np.prod(shape_i)
		assert size_i%8 == 0
		log2_size_i = int(np.ceil(np.log2(size_i)))
		source += (f"const int shape_i[{len(shape_i)}] = " 
			+ "{" +",".join(str(s) for s in shape_i)+ "};\n"
			+ f"const int size_i = {size_i};\n"
			+ f"const int log2_size_i = {log2_size_i};\n")


	if 'Scalar' in traits:
		Scalar = traits.pop('Scalar')
		if   'float32' in str(Scalar): ctype = 'float'
		elif 'float64' in str(Scalar): ctype = 'double'
		else: raise ValueError(f"Unrecognized scalar type {Scalar}")
		source += f"typedef {ctype} Scalar;\n"

	if 'Int' in traits:
		Int = traits.pop('Int')
		if   'int32' in str(Int): ctype = 'int'
		elif 'int64' in str(Int): ctype = 'long long'
		else: raise ValueError(f"Unrecognized scalar type {Int}")
		source += f"typedef {ctype} Int;\n"
		source += f"const Int Int_MAX = {np.iinfo(Int).max};"
"""

	for key,value in traits.items():
		if isinstance(value,numbers.Integral):
			source += f"const int {key}={value};\n"
		elif isinstance(value,type):
			source += f"typedef {np2cuda_dtype[value]} {key};\n"
		elif all(isinstance(v,numbers.Integral) for v in value):
			source += f"const int {key}[{len(value)}] = "+"{"+",".join(str(s) for s in value)+ "};\n"
		else: 
			raise ValueError(f"Unsupported trait {key}:{value}")

	# Special treatment for some traits
	if 'Int' in traits:
		source += f"const Int Int_MAX = {np.iinfo(traits['Int']).max};\n"
	if "shape_i" in traits:
		size_i = np.prod(traits['shape_i'])
		assert size_i%8 == 0
		log2_size_i = int(np.ceil(np.log2(size_i)))
		source += (f"const int size_i = {size_i};\n"
			+ f"const int log2_size_i = {log2_size_i};\n")

	return source