"""
	self.cost_based = self.model.startswith('Isotropic') or self.isCurvature
	if self.cost_based:
		self.metric = self.GetValue('cost',None,verbosity=3,
			help="Cost function for the minimal paths. cost = 1/speed.")
		self.dualMetric = self.GetValue('speed',None,verbosity=3,
			help="Speed function for the minimal paths. speed = 1/cost.")
	else:

# -------
	self.costVariation = self.GetValue('costVariation',default=None,
		help = "First order variation of the cost function "
		"(defined as 1 in the case of an anisotropic metric)")
	if self.cost_based: 
		# These models internally only use the cost function, not the speed function
		if self.metric is None: self.metric = self.dualMetric.dual()
		self.dualMetric = None
		if ad.is_ad(self.metric.cost):
			assert self.costVariation is None
			self.costVariation = self.metric.cost.coef
			self.metric = Metrics.Isotropic(self.metric.cost.value)
		# Internally, we always use a relative cost variation
		if self.costVariation is not None:
			self.costVariation/=cp.expand_dims(self.metric.cost,axis=-1)

#------

		# TODO : remove. No need to create this grid for our interpolation
		grid = ad.array(np.meshgrid(*(cp.arange(s,dtype=self.float_t) 
			for s in self.shape), indexing='ij')) # Adimensionized coordinates
		self.metric.set_interpolation(grid,periodic=self.periodic) # First order interpolation
"""
#--------

import numpy as np
import cupy as cp
from .inf_convolution import inf_convolution

def GetRHS(self):
	rhs = self.cost.copy()
	seedTags = cp.full(self.shape,False,dtype=bool)

	eikonal = self.kernel_data['eikonal']
	self.seeds = self.GetValue('seeds',default=None,
		help="Points from where the front propagation starts",array_float=True)
	if seeds is None:
		if eikonal.solver=='global_iteration': return
		trigger = self.GetValue('trigger',
			help="Points which trigger the eikonal solver front propagation")
		# Fatten the trigger a little bit
		trigger = cp.asarray(trigger,dtype=np.uint8)
		conv_kernel = cp.ones( (3,)*self.ndim,dtype=np.uint8)
		trigger = inf_convolution(trigger,conv_kernel,self.periodic,mix_is_min=False)
		eikonal.trigger = trigger
		return rhs,seedTags

	# Check and adimensionize seeds
	assert self.seeds.ndim==2 and self.seeds.shape[1]==self.ndim
	self.seeds = Grid.PointFromIndex(self.hfmIn,self.seeds,to=True) 
	if len(self.seeds)==1: self.seed=self.seeds[0]

	seedValues = cp.zeros(len(self.seeds),dtype=self.float_t)
	seedValues = self.GetValue('seedValues',default=seedValues,
		help="Initial value for the front propagation",array_float=True)
	if not ad.is_ad(seedValues):
		seedValueVariation = self.GetValue('seedValueVariation',default=None,
			help="First order variation of the seed values",array_float=True)
		if seedValueVariation is not None:
			seedValues = ad.Dense.new(seedValues,seedValueVariation.T)
	seedRadius = self.GetValue('seedRadius',default=2. if self.factoringRadius>0. else 0.,
		help="Spread the seeds over a radius given in pixels, so as to improve accuracy.")

	if seedRadius==0.:
		seedIndices = np.round(self.seeds).astype(int)
		self.seedValues = seedValues
		self.seedIndices = seedIndices		
	else:
		neigh = Grid.GridNeighbors(self.hfmIn,self.seed,seedRadius) # Geometry last
		r = seedRadius 
		aX = [cp.arange(int(np.floor(ci-r)),int(np.ceil(ci+r)+1)) for ci in self.seed]
		neigh =  ad.stack(cp.meshgrid( *aX, indexing='ij'),axis=-1)
		neigh = neigh.reshape(-1,neigh.shape[-1])
		neighValues = seedValues.repeat(len(neigh)//len(self.seeds)) # corrected below

		# Select neighbors which are close enough
		neigh = neigh[ad.Optimization.norm(neigh-self.seed,axis=-1) < r]

		# Periodize, and select neighbors which are in the domain
		nper = np.logical_not(self.periodic)
		inRange = np.all(np.logical_and(-0.5<=neigh[:,nper],
			neigh[:,nper]<cp.array(self.shape)[nper]-0.5),axis=-1)
		neigh = neigh[inRange,:]
		neighValues = neighValues[inRange,:]
		
		diff = (neigh - self.seed).T # Geometry first
#			neigh[:,self.periodic] = neigh[:,self.periodic] % self.shape[self.periodic]
		metric0 = self.Metric(self.seed)
		metric1 = self.Metric(neigh.T)
		self.seedValues = neighValues+0.5*(metric0.norm(diff) + metric1.norm(diff))
		self.seedIndices = neigh
	
	rhs,seedValues = ad.common_cast(rhs,seedValues)
	pos = tuple(seedIndices.T)
	rhs[pos] = seedValues
	seedTags[pos] = True
	eikonal.trigger = seedTags
	return rhs,seedTags

def SetArgs(self):
	if self.verbosity>=1: print("Preparing the problem rhs (cost, seeds,...)")
	eikonal = self.kernel_data['eikonal']
	shape_i = self.shape_i
	
	values = ad.GetValue('values',default=None,
		help="Initial values for the eikonal solver")
	if values is None: values = cp.full(self.shape,np.inf,dtype=self.float_t)
	block_values = misc.block_expand(values,shape_i,
		mode='constant',constant_values=np.inf,contiguous=True)
	eikonal.args['values']	= block_values

	# Set the RHS and seed tags
	rhs,seedTags = self.GetRHS()
	eikonal.args['rhs'] = misc.block_expand(rhs,shape_i,
		mode='constant',constant_values=np.nan)

	if np.prod(self.shape_i)%8!=0:
		raise ValueError('Product of shape_i must be a multiple of 8')
	seedPacked = misc.block_expand(seedTags,shape_i,
		mode='constant',constant_values=True)
	seedPacked = misc.packbits(seedPacked,bitorder='little')
	seedPacked = seedPacked.reshape( self.shape_o + (-1,) )
	eikonal.args['seedTags'] = seedPacked

	# Handle multiprecision
	if self.multiprecision:
		block_valuesq = cp.zeros(block_values.shape,dtype=self.int_t)
		self.block.update({'valuesq':block_valuesq})

	if self.strict_iter_o:
		self.block['valuesNext']=block_values.copy()
		if self.multiprecision:
			self.block['valuesqNext']=block_valuesq.copy()

	if self.bound_active_blocks:
		minChg = cp.full(self.shape_o,np.inf,dtype=self.float_t)
		self.block['minChgPrev_o'] = minChg
		self.block['minChgNext_o'] = minChg.copy()