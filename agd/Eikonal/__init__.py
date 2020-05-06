# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import importlib
import functools

from .LibraryCall import GetBinaryDir
from .run_detail import Cache
from .DictIn import dictIn,dictOut,CenteredLinspace


def VoronoiDecomposition(arr):
	"""
	Calls the FileVDQ library to decompose the provided quadratic form(s),
	as based on Voronoi's first reduction of quadratic forms.
	"""
	from ..Metrics import misc
	from . import FileIO
	bin_dir = GetBinaryDir("FileVDQ",None)
	vdqIn ={'tensors':np.moveaxis(misc.flatten_symmetric_matrix(arr),0,-1)}
	vdqOut = FileIO.WriteCallRead(vdqIn, "FileVDQ", bin_dir)
	return np.moveaxis(vdqOut['weights'],-1,0),np.moveaxis(vdqOut['offsets'],[-1,-2],[0,1])