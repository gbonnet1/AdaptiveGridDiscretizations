# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

def default_traits(self):
	"""
	Default traits of the GPU implementation of an HFM model.
	(self is an instance of the class Interface from file interface.py)
	"""
	traits = {
	'Scalar': np.float32,
	'Int':    np.int32,
	'OffsetT':np.int32,
	'multiprecision_macro':0,
	'pruning_macro':0,
	}

	ndim = self.ndim
	model = self.model

	if model=='Isotropic2':
		#Large shape, many iterations, to take advantage of block based causality
		traits.update({'shape_i':(24,24),'niter_i':48,})
	elif ndim==2: traits.update({'shape_i':(8,8),'niter_i':16,})

	# Curvature penalized models get special treatment
	elif model in ('ReedsShepp2','ReedsSheppForward2'): 
		traits.update({'shape_i':(4,4,4),'niter_i':6})
	elif model in ('Dubins2','Elastica2'):
		# Small shape, single iteration, since stencils are too wide anyway
		traits.update({'shape_i':(4,4,2),'niter_i':1})


	elif ndim==3:
		traits.update({'shape_i':(4,4,4),'niter_i':12,})
	elif ndim==4:
		traits.update({'shape_i':(4,4,4,4),'niter_i':16,})
	else:
		raise ValueError("Unsupported model")

	if model=='Elastica2': traits['merge_sort_macro']=1
	if model.startswith('TTI'): traits.update({'nmix':10,'nmix_macro':1})
	return traits

def nscheme(self):
	"""
	Provides the structure of the finite difference scheme used.
	(number of symmmetric offsets, foward offsets, max or min of a number of schemes)
	"""
	ndim = self.ndim
	symdim = int( (ndim*(ndim+1))/2 )
	model = self.model_

	nsym=0 # Number of symmetric offsets
	nfwd=0 # Number of forward offsets
	nmix=1 # maximum or minimum of nmix schemes
	if model=='Isotropic':              nsym = ndim
	elif model in ('Riemann','Rander'): nsym = symdim
	elif model=='ReedsShepp':           nsym = symdim
	elif model=='ReedsSheppForward':    nsym = 1; nfwd = symdim
	elif model=='Dubins':               nfwd = symdim; nmix = 2
	elif model=='Elastica':
		nFejer = self.kernel_data['eikonal'].traits.get('nFejer_macro',5)
		nfwd = nFejer*symdim

	nact = nsym+nfwd # max number of active offsets
	ntot = 2*nsym+nfwd
	nactx = nact*nmix
	ntotx = ntot*nmix

	return {'nsym':nsym,'nfwd':nfwd,'nmix':nmix,
	'nact':nact,'ntot':ntot,'nactx':nactx,'ntotx':ntotx}


