# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import os
import time
from collections import OrderedDict,namedTuple
from types import SimpleNamespace

from . import kernel_traits
from . import misc
from . import cupy_module_helper
from .cupy_module_helper import GetModule,SetModuleConstant

# Deferred implementation of Interface member functions
from . import _Kernel
from . import _solvers 
from . import _GetGeodesics
from . import _PostProcess

from ... import AutomaticDifferentiation as ad
from ... import Metrics
from ... import LinearParallel as lp

class Interface(object):
	"""
	This class carries out the RunGPU function work. 
	It should not be used directly.
	"""
	def __init__(self,hfmIn):

		self.hfmIn = hfmIn
		if hfmIn['arrayOrdering'] != 'RowMajor':
			raise ValueError("Only RowMajor indexing supported")

		# Needed for GetValue
		self.help = hfmIn.get('help',[])
		self.hfmOut = {'keys':{
		'used':['exportValues','origin','arrayOrdering'],
		'defaulted':OrderedDict(),
		'visited':[],
		'help':OrderedDict(),
		'kernelStats':OrderedDict(),
		} }
		self.help_content = self.hfmOut['keys']['help']
		self.verbosity = 1

		self.verbosity = self.GetValue('verbosity',default=1,
			help="Choose the amount of detail displayed on the run")
		self.help = self.GetValue('help',default=[], # help on help...
			help="List of keys for which to display help")
		
		self.model = self.GetValue('model',help='Minimal path model to be solved.')
		# Unified treatment of standard and extended curvature models
		if self.model.endswith("Ext2"): self.model=self.model[:-4]+"2"

		self.ndim = len(hfmIn['dims'])
		self.kernel_data = {key:kernel_data_default() 
			for key in ('eikonal','flow','geodesic','forwardAD','reverseAD')}

	def kernel_data_default():
		members = ['traits','source','policy','module','kernel','args','trigger','stats']
		dtype = namedTuple('solve',members,defaults=[None for m in members])
		result = dtype()
		result.traits=dict()
		result.policy=SimpleNamespace() 
		result.args = OrderedDict()
		result.stats = dict()
		return result
		
	def HasValue(self,key):
		self.hfmOut['keys']['visited'].append(key)
		return key in self.hfmIn

	def GetValue(self,key,default="_None",verbosity=2,help=None,array_float=False):
		"""
		Get a value from a dictionnary, printing some help if requested.
		"""
		if key in self.help and key not in self.help_content:
			self.help_content[key]=help
			if self.verbosity>=1:
				if help is None: 
					print(f"Sorry : no help for key {key}")
				else:
					print(f"---- Help for key {key} ----")
					print(help)
					if isinstance(default,str) and default=="_None": 
						print("No default value")
					elif isinstance(default,str) and default=="_Dummy":
						print(f"see out['keys']['defaulted'][{key}] for default")
					else:
						print("default value :",default)
					print("-----------------------------")

		if key in self.hfmIn:
			self.hfmOut['keys']['used'].append(key)
			value = self.hfmIn[key]
			return self.caster(value) if array_float else value
		elif isinstance(default,str) and default == "_None":
			raise ValueError(f"Missing value for key {key}")
		else:
			assert key not in self.hfmOut['keys']['defaulted']
			self.hfmOut['keys']['defaulted'][key] = default
			if verbosity<=self.verbosity:
				print(f"key {key} defaults to {default}")
			return default

	def Warn(self,msg):
		if self.verbosity>=-1:
			print("---- Warning ----\n",msg,"\n-----------------\n")

	def Run(self):
		self.SetKernelTraits()
		self.SetGeometry()
		self.SetArgs()
		self.SetKernel()
		self.Solve()
		self.PostProcess()
		self.SolveAD()
		self.GetGeodesics()
		self.FinalCheck()

		return self.hfmOut

	SetKernelTraits = _Kernel.SetKernelTraits
	SetGeometry = _PreProcess.SetGeometry
	SetArgs = _PreProcess.SetArgs
	SetKernel = _Kernel.SetKernel
	Solve = _solvers.Solve
	PostProcess = _PostProcess.PostProcess
	SolveAD = _PostProcess.SolveAD
	GetGeodesics = _GetGeodesics.GetGeodesics

	CostMetric = _PreProcess.CostMetric
	GetRHS = _PreProcess.GetRHS
	global_iteration = _solvers.global_iteration
	adaptive_gauss_siedel_iteration = _solvers.adaptive_gauss_siedel_iteration
	set_minChg_thres = _solvers.set_minChg_thres


	@property
	def isCurvature(self):
		return self.model in ['ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2']

	@property
	def metric(self):
		if self._metric is None: self._metric = self._dualMetric.dual()
		return self._metric
	@property
	def dualMetric(self):
		if self._dualMetric is None: self._dualMetric = self._metric.dual()
		return self._dualMetric

	
	def FinalCheck(self):
		self.hfmOut['keys']['unused'] = list(set(self.hfmIn.keys())-set(self.hfmOut['keys']['used']))
		if self.verbosity>=1 and self.hfmOut['keys']['unused']:
			print(f"!! Warning !! Unused keys from user : {self.hfmOut['keys']['unused']}")




























