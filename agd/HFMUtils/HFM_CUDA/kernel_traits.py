import numpy as np
import numbers

def dtype(arg,data_t):
	"""
	For a numeric array, returns dtype.
	Otherwise, returns one of the provided floating point 
	or integer data type, depending on the argument data type.
	Inputs:
	 - data_t (tuple) : (float_t,int_t)
	"""
	float_t,int_t = data_t
	if isinstance(arg,numbers.Real): 
		return float_t
	elif isinstance(arg,numbers.Integral):
		return int_t
	elif isinstance(arg,(tuple,list)):
		return dtype(arg[0],data_t)
	else:
		return arg.dtype

def default_traits(model):
	"""
	Default traits of the GPU implementation of an HFM model.
	"""
	traits = {
	'Scalar':'float32',
	'Int':   'int32',
	}

	if model=='Isotropic2':
		traits.update({
		'shape_i':(8,8),
		})
	elif model=='Isotropic3':
		traits.update({
		'shape_i':(4,4,4),
		})
	else:
		raise ValueError("Unsupported model")

	return traits

def kernel_source(model,traits):
	"""
	Returns the source (mostly a preamble) for the gpu kernel code 
	for the given traits and model.
	"""
	source = ""
	for key in list(traits.keys()):
		if 'macro' in key:
			source += f"#define {key} {traits[key]}\n"
			traits.pop(key)
		else:
			source += f"#define {key}_macro\n"

	traits = traits.copy()

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

	for key,value in traits.items():
		source += f"const int {key}={value};\n"

	source += f'#include "{model}.h"\n'
	return source