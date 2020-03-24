"""
This class is used to propagate the AGSI updates from the current list 
to its relevant neighbors.
"""

import numpy as np
import os
import time
from ... import AutomaticDifferentiation as ad
from .cupy_module_helper import GetModule,SetModuleConstant,traits_header,getmtime_max
class propagate:

	def __init__(self,minChg,
		connectivity='diamond',boundChg=np.inf,
		int_t=np.int32,blockSize=64):

		self.minChg = minChg
		self.ndim = minChg.ndim
		self.shape = minChg.shape
		self.xp = ad.cupy_generic.get_array_module(minChg)

		self.connectivity = connectivity
		if connectivity=='diamond': self.nconnect=2*self.ndim+1
		else: raise ValueError("Unrecognized connectivity")

		self.boundChg=boundChg
		assert boundChg==np.inf

		self.int_t = int_t
		self.tags = self.xp.full(minChg.shape,-1,dtype=self.int_t)

		self.blockSize=blockSize
		self._set_modules()

		self.time = {'TagNeigh_ker':0.,"ReadNeigh_ker":0.}

	def _set_modules(self):
		cuda_path = os.path.join(
			os.path.dirname(os.path.realpath(__file__)),"cuda/propagate")
		cuoptions = ("-default-device", f"-I {cuda_path}")
		date_modified = getmtime_max(cuda_path)
		
		common_source = (traits_header({'ndim':self.ndim,'shape':self.shape,"dummy_minChg_macro":1}) +
			f"// Date last modified {date_modified}\n")

		TagNeigh_source = common_source + '#include "TagNeigh.h"\n'
		self.TagNeigh_mod = GetModule(TagNeigh_source,cuoptions)
		self.TagNeigh_ker = self.TagNeigh_mod.get_function("TagNeigh")

		ReadNeigh_source = common_source + '#include "ReadNeigh.h"\n'
		self.ReadNeigh_mod = GetModule(ReadNeigh_source,cuoptions)
		self.ReadNeigh_ker = self.ReadNeigh_mod.get_function("ReadNeigh")

	def __call__(self,updateNow):
		assert updateNow.dtype.type==self.int_t
		updateNext = self.xp.full((updateNow.size*self.nconnect,),-1,dtype=self.int_t)

		SetModuleConstant(self.TagNeigh_mod, 'index_size',updateNow.size,self.int_t)
		SetModuleConstant(self.ReadNeigh_mod,'index_size',updateNow.size,self.int_t)
		
		size_o = int(np.ceil(updateNow.size/self.blockSize))
		size_i = self.blockSize

		top = time.time()
		self.TagNeigh_ker((size_o,),(size_i,),(self.minChg,updateNow,self.tags))
		if updateNow[0]==-2:print("hey")
		self.time['TagNeigh_ker']+=time.time()-top; top = time.time()
		self.ReadNeigh_ker((size_o,),(size_i,),(updateNow,self.tags,updateNext))
		if updateNow[0]==-2:print("hey")
		self.time['ReadNeigh_ker']+=time.time()-top

#		print(f"updateNow : {updateNow}, next {updateNext}")
#		print(f"minChg {self.minChg}")
#		print(f"tags {self.tags}")

		updateNext = updateNext[updateNext!=-1]
#		print(f"next pruned : {updateNext}")
		self.tags.reshape(-1)[updateNext] = -1 
		self.minChg.reshape(-1)[updateNow]=0
		return updateNext
