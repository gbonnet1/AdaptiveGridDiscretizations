# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

def round_up(num,den):
	"""
	Returns the least multiple of den after num.
	num and den must be integers. 
	"""
	num,den = np.asarray(num),np.asarray(den)
	return (num+den-1)//den