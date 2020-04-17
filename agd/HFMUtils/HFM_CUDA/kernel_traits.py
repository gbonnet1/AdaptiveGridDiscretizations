# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

def default_traits(self):
	"""
	Default traits of the GPU implementation of an HFM model.
	(self is an instance of the class Interface from file interface.py)
	"""
	traits = {
	'Scalar':np.float32,
	'Int':   np.int32,
	'multiprecision_macro':0,
	'pruning_macro':0,
	}

	ndim = self.ndim
	model = self.model

	if model=='Isotropic2':
		#Large shape, many iterations, to take advantage of block based causality
		traits.update({'shape_i':(24,24),'niter_i':48,})
	elif ndim==2: traits.update({'shape_i':(8,8),'niter_i':16,})
	elif model in ('ReedsShepp2','ReedsSheppForward2'): 
		traits.update({'shape_i':(4,4,4),'niter_i':6})
	elif model in ('Dubins2','Elastica2'):
		# Small shape, single iteration, since stencils are too wide anyway
		traits.update({'shape_i':(4,4,2),'niter_i':1})
	elif ndim==3:
		traits.update({'shape_i':(4,4,4),'niter_i':12,})
	else:
		raise ValueError("Unsupported model")
	return traits

def nact(self):
	"""
	Max number of active neighbors for an HFM model.
	"""
	ndim = self.ndim
	symdim = int( (ndim*(ndim+1))/2 )
	model = self.model_

	if model=='Isotropic':
		return ndim
	elif model in ('Riemann','Rander'):
		return symdim
	elif model=='ReedsShepp':
		return symdim
	elif model=='ReedsSheppForward':
		return symdim+1
	elif model=='Dubins':
		return 2*symdim
	elif model=='Elastica':
		nFejer = self.kernel_data['eikonal'].traits.get('nFejer_macro',5)
		return nFejer*symdim


