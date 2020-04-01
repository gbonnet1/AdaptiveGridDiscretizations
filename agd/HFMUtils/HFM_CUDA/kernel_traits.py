# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

def default_traits(interface):
	"""
	Default traits of the GPU implementation of an HFM model.
	"""
	traits = {
	'Scalar':np.float32,
	'Int':   np.int32,
	'multiprecision_macro':0,
	'pruning_macro':0,
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