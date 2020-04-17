import numpy as np
import cupy as cp
from . import cupy_module_helper
from . import inf_convolution
from . import misc

def GetGeodesics(self):
		if not self.hasTips: return
		geodesic = self.kernel_data['geodesic'] # Not using the common solver here

		# Set the kernel traits
		geodesic_step = self.GetValue('geodesic_step',default=0.25,
			help='Step size, in pixels, for the geodesic ODE solver')

		eucl_delay = int(np.sqrt(self.ndim)/geodesic_step)
		eucl_delay = self.GetValue('geodesic_PastSeed_delay',default=eucl_delay,
			help="Delay, in iterations, for the 'PastSeed' stopping criterion of the "
			"geodesic ODE solver") # Likely in curvature penalized models
		nymin_delay = int(8.*np.sqrt(self.ndim)/geodesic_step)
		nymin_delay = self.GetValue('geodesic_Stationnary_delay',default=nymin_delay,
			help="Delay, in iterations, for the 'Stationnary' stopping criterion of the "
			"geodesic ODE solver") # Rather unlikely

		traits = { # Suggested defaults
			'eucl_delay':int(eucl_delay),
			'nymin_delay':int(nymin_delay),
			'EuclT':np.uint8,
			}
		if any(self.periodic): 
			traits['periodic'] = 1
			traits['periodic_axes'] = self.periodic
		traits.update(self.GetValue('geodesic_traits',default=traits,
			help='Traits for the geodesic backtracking kernel') )
		traits.update({ # Non-negotiable
			'ndim':self.ndim,
			'Int':self.int_t,
			'Scalar':self.float_t})
		geodesic.traits=traits

		# Get the module
		geodesic.source = cupy_module_helper.traits_header(traits,
			join=True,integral_max=True) + "\n"
		geodesic.source += '#include "GeodesicODE.h"\n'+self.cuda_date_modified
		geodesic.module = cupy_module_helper.GetModule(geodesic.source,self.cuoptions)
		geodesic.kernel = geodesic.module.get_function('GeodesicODE')

		# Set the module constants
		def SetCst(*args):
			cupy_module_helper.SetModuleConstant(geodesic.module,*args)
		# Note: geodesic solver does not use bilevel array structure
		shape_tot = self.shape
		size_tot = int(np.prod(shape_tot))  #distinct from size_tot used for solver
		SetCst('shape_tot',shape_tot,self.int_t)
		SetCst('size_tot', size_tot, self.int_t)
		typical_len = int(max(40,0.5*np.max(shape_tot)/geodesic_step))
		typical_len = self.GetValue('geodesic_typical_length',default=typical_len,
			help="Typical expected length of geodesics (number of points).")
		# Typical geodesic length is max_len for the GPU solver, which computes just a part
		SetCst('max_len', typical_len, self.int_t) 
		causalityTolerance = self.GetValue('geodesic_causalityTolerance',default=4.,
			help="Used in criterion for rejecting points in flow interpolation")
		SetCst('causalityTolerance', causalityTolerance, self.float_t)
		nGeodesics=len(self.tips)

		# Prepare the euclidean distance to seed estimate (for stopping criterion)
		eucl_bound_default = 12 if self.isCurvature else 6
		eucl_bound = self.GetValue('geodesic_targetTolerance',default=eucl_bound_default,
			help="Tolerance, in pixels, for declaring a seed as reached.")
		eucl_t = geodesic.traits['EuclT']
		eucl_integral = np.dtype(eucl_t).kind in ('i','u') # signed or unsigned integer
		eucl_max = np.iinfo(eucl_t).max if eucl_integral else np.inf
		eucl = np.full_like(self.seedTags,eucl_max,dtype=eucl_t)
		eucl[self.seedTags] = 0
		eucl_mult = 5 if eucl_integral else 1
		eucl_kernel = inf_convolution.distance_kernel(radius=1,ndim=self.ndim,
			dtype=eucl_t,mult=eucl_mult)
		eucl = inf_convolution.inf_convolution(eucl,eucl_kernel,periodic=self.periodic,
			upper_saturation=eucl_max,overwrite=True,niter=int(np.ceil(eucl_bound)))
		eucl[eucl>eucl_mult*eucl_bound] = eucl_max
		eucl=cp.ascontiguousarray(eucl)
		self.hfmOut['eucl']=eucl
		self.hfmOut['eucl_kernel']=eucl_kernel
		self.hfmOut['seedTags']=self.seedTags

		# Run the geodesic ODE solver
		stopping_criterion = list(("Stopping criterion",)*nGeodesics)
		corresp = list(range(nGeodesics))
		tips = self.hfmIn.PointFromIndex(self.tips,to=True)
		geodesics = [ [tip.reshape(1,-1)] for tip in tips]

		block_size=self.GetValue('geodesic_block_size',default=32,
			help="Block size for the GPU based geodesic solver")
		geodesic_termination_codes = [
			'Continue', 'AtSeed', 'InWall', 'Stationnary', 'PastSeed', 'VanishingFlow']

		max_len = int(max(40,20*np.max(shape_tot)/geodesic_step))
		max_len = self.GetValue("geodesic_max_length",default=max_len,
			help="Maximum allowed length of geodesics.")
		
		flow = self.kernel_data['flow']
		flow_vector    = self.flow_vector
		flow_weightsum = misc.block_squeeze(flow.args['flow_weightsum'],self.shape)
		values = self.values.astype(self.float_t)
		args = (flow_vector,flow_weightsum,values,eucl)
		args = tuple(cp.ascontiguousarray(arg) for arg in args)

		geoIt=0; geoMaxIt = int(np.ceil(max_len/typical_len))
		while len(corresp)>0:
			if geoIt>=geoMaxIt: 
				self.Warn("Geodesic solver failed to converge, or geodesic has too many points"
					" (in latter case, try setting 'geodesic_max_len':np.inf)")
				break
			geoIt+=1
			nGeo = len(corresp)
			x_s = cp.full( (nGeo,typical_len,self.ndim), np.nan, self.float_t)
			x_s[:,0,:] = np.stack([geodesics[i][-1][-1,:] for i in corresp], axis=0)
			len_s = cp.full((nGeo,),-1,self.int_t)
			stop_s = cp.full((nGeo,),-1,np.int8)

			nBlocks = int(np.ceil(nGeo/block_size))

			SetCst('nGeodesics', nGeo, self.int_t)
			geodesic.kernel( (nBlocks,),(block_size,),args + (x_s,len_s,stop_s))
			corresp_next = []
			for i,x,l,stop in zip(corresp,x_s,len_s,stop_s): 
				geodesics[i].append(x[1:int(l)])
				if stop!=0: stopping_criterion[i] = geodesic_termination_codes[int(stop)]
				else: corresp_next.append(i)
			corresp=corresp_next

		geodesics_cat = [np.concatenate(geo,axis=0) for geo in geodesics]
		self.hfmOut['geodesics']=[self.hfmIn.PointFromIndex(geo).T for geo in geodesics_cat]
		self.hfmOut['geodesic_stopping_criteria'] = stopping_criterion
