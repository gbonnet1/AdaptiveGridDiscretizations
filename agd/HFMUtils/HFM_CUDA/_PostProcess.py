import numpy as np
from . import misc
from . import kernel_traits
from . import _solvers
from ... import FiniteDifferences as fd
from ... import LinearParallel as lp


def PostProcess(self):
		if self.verbosity>=1: print("Post-Processing")
		values = misc.block_squeeze(self.block['values'],self.shape)
		if self.multiprecision:
			valuesq = misc.block_squeeze(self.block['valuesq'],self.shape)
			if self.GetValue('values_float64',default=False,
				help="Export values using the float64 data type"):
				float64_t = np.dtype('float64').type
				self.hfmOut['values'] = (values.astype(float64_t) 
					+ float64_t(self.multip_step) * valuesq)
			else:
				self.hfmOut['values'] = (values + valuesq.astype(self.float_t)*self.multip_step)
		else:
			self.hfmOut['values'] = values

		# Compute the geodesic flow, if needed, and related quantities
		shape_oi = self.shape_o+self.shape_i
		nact = kernel_traits.nact(self)
		ndim = self.ndim

		if self.flow_traits.get('flow_weights_macro',False):
			self.flow_kernel_argnames.append('flow_weights')
			self.block['flow_weights'] = self.xp.empty((nact,)+shape_oi,dtype=self.float_t)
		if self.flow_traits.get('flow_weightsum_macro',False):
			self.flow_kernel_argnames.append('flow_weightsum')
			self.block['flow_weightsum'] = self.xp.empty(shape_oi,dtype=self.float_t)
		if self.flow_traits.get('flow_offsets_macro',False):
			self.flow_kernel_argnames.append('flow_offsets')
			self.block['flow_offsets'] = self.xp.empty((ndim,nact,)+shape_oi,dtype=np.int8)
		if self.flow_traits.get('flow_indices_macro',False):
			self.flow_kernel_argnames.append('flow_indices')
			self.block['flow_indices'] = self.xp.empty((nact,)+shape_oi,dtype=self.int_t)
		if self.flow_traits.get('flow_vector_macro',False):
			self.flow_kernel_argnames.append('flow_vector')
			self.block['flow_vector'] = self.xp.ones((ndim,)+shape_oi,dtype=self.float_t)

		self.flow_needed = any(self.flow_traits.get(key+"_macro",False) for key in 
			('flow_weights','flow_weightsum','flow_offsets','flow_indices','flow_vector'))
		if self.flow_needed: _solvers.global_iteration(self,solver=False)

		self.flow = {}
		for key in self.block:
			if key.startswith('flow_'):
				self.flow[key]=misc.block_squeeze(self.block[key],self.shape,contiguous=True)

		if self.model.startswith('Rander') and 'flow_vector' in self.flow:
			if self.dualMetric is None: self.dualMetric = self.metric.dual()
			flow_orig = self.flow['flow_vector']
			m = fd.as_field(self.metric.m,self.shape,depth=2)
			w = fd.as_field(self.metric.w,self.shape,depth=1)
			eucl_gradient = lp.dot_AV(m,flow_orig)+w
			flow = self.dualMetric.gradient(eucl_gradient)
			flow[np.isnan(flow)]=0. # Vanishing flow yields nan after gradient
			flow = self.xp.ascontiguousarray(flow)
			self.flow['flow_vector_orig'],self.flow['flow_vector'] = flow_orig,flow

		if self.exportGeodesicFlow:
			self.hfmOut['flow'] = - self.flow['flow_vector'] * self.h_broadcasted

		if self.forward_ad or self.reverse_ad:
			spmod=self.xp.cupyx.scipy.sparse
			xp=self.xp
			weightsum = self.xp.expand_dims(self.flow['flow_weightsum'],axis=0)
			self.boundary = weightsum==0. #seeds, or walls, or out of domain
			coef = xp.concatenate((weightsum,-self.flow['flow_weights']),axis=0)
			rg = xp.arange(np.prod(self.shape)).reshape((1,)+self.shape)
			row = self.xp.broadcast_to(rg,coef.shape)
			col = xp.concatenate((rg,self.flow['flow_indices']),axis=0)

			coef[0,self.boundary] = 1.			
			self.triplets = (coef.flatten(),(row.flatten(),col.flatten())) 
			self.spmat = spmod.coo_matrix(self.triplets)

		if self.forward_ad:
			if self.costVariation is None:
				self.costVariation = self.xp.zeros(self.shape+self.seedValues.size_ad)
			rhs=self.cost_variation
			if ad.is_ad(self.seedValues):
				rhs[self.seedIndices.T,:] = self.seedValues.coef
			hfmOut['valueVariation'] = spmod.linalg.lsqr(self.spmat.tocsr(),rhs)

		if self.reverse_ad:
			rhs = self.GetValue('sensitivity',help='Reverse automatic differentiation')
			hfmOut['valueSensitivity'] = spmod.linalg.lsqr(self.spmat.T.tocsr(),rhs)