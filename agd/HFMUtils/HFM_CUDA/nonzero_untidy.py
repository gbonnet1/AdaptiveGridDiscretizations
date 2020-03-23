"""
This file implements a gpu accelerated variant of the 'nonzero' functions of cupy and numpy.
We return the indices of all non-zero entries in the flattened array.
The implementation is untidy, in that an unspecified number of "-1", which can be safely
ignored, might be inserted in between the valid indices. 
This inconvenience is counterbalanced by a significant improvement in computation speed.
"""
import numpy as np
import os
from .cupy_module_helper import GetModule,SetModuleConstant,getmtime_max
from .kernel_traits import kernel_source
from ... import AutomaticDifferentiation as ad
#from packaging.version import Version

class nonzero:
	def __init__(self,arr,shape_i=None,log2_size_i=10,size2_i=1024,
		int_t=np.int32):
		"""
		Inputs : 
		 - arr : whose non-zero entries are to be extracted
		 - shape_i (optional) : block dimension for the Lookup pass. 
		   Should be roughly proportionnal to the shape. (The product of coordinates 
		   should not exceed the maximum block size allowed by the hardware.)
		 - log2_size_i (optional) : base two logarithm of the block dimension.
		  Ignored if shape_i is provided.
		 - size2_i (optional) : block dimension for Compress pass. (Should not exceed
		 the maximum block size allowed by the hardware.)
		"""
		self.arr = arr
		self.shape_tot = np.array(arr.shape)
		self.ndim = arr.ndim
		self.shape_i = shape_i if shape_i is not None else self._select_shape_i(log2_size_i)
		self.size_i = np.prod(self.shape_i)
		self.shape_o = np.ceil(self.shape_tot/self.shape_i).astype(int)
		self.size_o = np.prod(self.shape_o)
		self.size_tot = self.size_o*self.size_i # Exceeds prod(shape_tot)

		self.size2_totmax = self.size_tot
		self.size2_i = size2_i
		self.size2_omax = np.ceil(self.size2_totmax/self.size2_i).astype(int)
		self.size2_totmax = self.size2_omax*self.size2_i

		self.int_t = int_t
		self.boolatom_t = arr.dtype.type

		xp = ad.cupy_generic.get_array_module(arr)
		self.xp = xp
		self.index1 =  xp.empty((self.size_tot,),    dtype=self.int_t)
		self.nindex1 = xp.empty((self.size_o,),      dtype=self.int_t)
		self.index2 =  xp.empty((self.size2_totmax,),dtype=self.int_t)
		self.nindex2 = xp.empty((self.size2_omax,),  dtype=self.int_t)

		self._set_modules()

	def _select_shape_i(self,log2_size_i):
		"""
		Finds a shape roughly proportionnal to self.shape but whose product of coordinates
		does not exceed 2**log2_size_i
		"""
		log2_shape_i = np.ceil(np.log2(self.shape_tot)).astype(int)
		while True:
			s = log2_shape_i.sum()-log2_size_i
			if 0<=s<self.ndim: break
			log2_shape_i = np.maximum(1,log2_shape_i-s//self.ndim)
		log2_shape_i[:s]-=1
		return 2**log2_shape_i

	def _set_modules(self):
		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda/nonzero")
		date_modified = getmtime_max(cuda_path)
		cuoptions = ("-default-device", f"-I {cuda_path}")
		int_t = self.int_t
		
		source1 = kernel_source({'ndim':self.ndim,'shape_i':self.shape_i,
			'BoolAtom':self.boolatom_t,'Int':self.int_t})
		source1 += ('#include "Lookup.h"\n'
			f"// Date last modified {date_modified}\n")
		mod1 = GetModule(source1,cuoptions)
		for key,value in (
			('shape_tot',self.shape_tot), ('shape_o',self.shape_o), ('size_o',self.size_o)):
			SetModuleConstant(mod1,key,value,dtype=int_t)

		source2 = kernel_source({'shape_i':(self.size2_i,)})
		source2+=('#include "Compress.h"\n'
		 	f"// Date last modified {date_modified}\n")
		mod2 = GetModule(source2,cuoptions)

		self.mod1 = mod1
		self.mod2 = mod2
		self.ker1 = mod1.get_function('Lookup')
		self.ker2 = mod2.get_function('Compress')

		# No support yet for other types
		assert self.int_t == np.int32

	def __call__(self):
		"""
		Finds the non-zeros in the arr variable.
		Returns their list and an upper bound on their number 
		(counted from the start of the array).
		"""
		# Call the first kernel
		self.index1.fill(-1); self.nindex1.fill(0)
		self.ker1(self.shape_o,self.shape_i,(self.arr,self.index1,self.nindex1))
		nindex1_max = self.xp.max(self.nindex1)

		size2 = nindex1_max * self.size_o
		size2_o = np.ceil(size2/self.size2_i).astype(int)
		SetModuleConstant(self.mod2,'size_tot',size2)

		# Call the second kernel
		self.index2.fill(-1); self.nindex2.fill(0)
		self.ker2((size2_o,),(self.size2_i,),(self.index1,self.index2,self.nindex2))
		nindex2_max = self.xp.max(self.nindex2)

		return self.index2,(nindex2_max*size2_o)


