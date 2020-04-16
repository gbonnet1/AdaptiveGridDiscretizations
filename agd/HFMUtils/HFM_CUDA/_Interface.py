# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from collections import OrderedDict
from types import SimpleNamespace

# Deferred implementation of Interface member functions
from . import _Kernel
from . import _solvers 
from . import _GetGeodesics
from . import _PostProcess
from . import _SetGeometry
from . import _SetArgs

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
		self.hfmOut = {'keys':{
		'used':['exportValues','origin','arrayOrdering'],
		'default':OrderedDict(),
		'visited':[],
		'help':OrderedDict(),
		'kernelStats':OrderedDict(),
		} }
		self.verbosity = 1

		self.verbosity = self.GetValue('verbosity',default=1,
			help="Choose the amount of detail displayed on the run")
		
		self.model = self.GetValue('model',help='Minimal path model to be solved.')
		# Unified treatment of standard and extended curvature models
		if self.model.endswith("Ext2"): self.model=self.model[:-4]+"2"

		self.ndim = len(hfmIn['dims'])
		self.kernel_data = {key:SimpleNamespace()
			for key in ('eikonal','flow','geodesic','forwardAD','reverseAD')}
		for value in self.kernel_data.values(): 
			value.__dict__.update({'args':dict(),'policy':SimpleNamespace(),'stats':dict()})
		# ['traits','source','policy','module','kernel','args','trigger','stats']

	@property # Dimension agnostic model
	def model_(self): return self.model[:-1]
		
	def HasValue(self,key):
		self.hfmOut['keys']['visited'].append(key)
		return key in self.hfmIn

	def GetValue(self,key,default="_None",verbosity=2,array_float=False,
		help="Sorry : no help for this key"):
		"""
		Get a value from a dictionnary, printing some help if requested.
		"""
		# We only import arguments once, otherwise risks of issues with multiple defaults
		assert key not in self.hfmOut['keys']['help']

		self.hfmOut['keys']['help'][key] = help
		self.hfmOut['keys']['default'][key] = default

		if key in self.hfmIn:
			self.hfmOut['keys']['used'].append(key)
			value = self.hfmIn[key]
			return self.caster(value) if array_float else value
		elif isinstance(default,str) and default == "_None":
			raise ValueError(f"Missing value for key {key}")
		else:
			if verbosity<=self.verbosity:
				if isinstance(default,str) and default=="_Dummy":
					print(f"see out['keys']['default'][{key}] for default")
				else:print(f"key {key} defaults to {default}")
			return default

	def Warn(self,msg):
		if self.verbosity>=-1:
			print("---- Warning ----\n",msg,"\n-----------------\n")

	def Run(self):
		self.SetKernelTraits()
		self.SetGeometry()
		self.SetArgs()
		self.SetKernel()
		self.Solve('eikonal')
		self.PostProcess()
		self.SolveAD()
		self.GetGeodesics()
		self.FinalCheck()

		return self.hfmOut

	SetKernelTraits = _Kernel.SetKernelTraits
	SetGeometry = _SetGeometry.SetGeometry
	SetArgs = _SetArgs.SetArgs
	SetKernel = _Kernel.SetKernel
	Solve = _solvers.Solve
	PostProcess = _PostProcess.PostProcess
	SolveAD = _PostProcess.SolveAD
	GetGeodesics = _GetGeodesics.GetGeodesics

	CostMetric = _SetGeometry.CostMetric
	SetRHS = _SetArgs.SetRHS
	global_iteration = _solvers.global_iteration
	adaptive_gauss_siedel_iteration = _solvers.adaptive_gauss_siedel_iteration
	set_minChg_thres = _solvers.set_minChg_thres
	SolveLinear = _PostProcess.SolveLinear


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
		self.hfmOut['stats'] = {key:value.stats for key,value in self.kernel_data.items()}
		self.hfmOut['solverGPUTime'] = self.kernel_data['eikonal'].stats['time']
		self.hfmOut['keys']['unused'] = list(set(self.hfmIn.keys()) 
			-set(self.hfmOut['keys']['used']) # Used by interface
			-{'array_float_caster'}) # Set by interface
		if self.verbosity>=1 and self.hfmOut['keys']['unused']:
			print(f"!! Warning !! Unused keys from user : {self.hfmOut['keys']['unused']}")





























