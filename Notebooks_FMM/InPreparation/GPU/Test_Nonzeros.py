import sys; sys.path.insert(0,"../../..") # Allow import of agd from parent directory 
from agd.HFMUtils.HFM_CUDA import nonzero_untidy

import cupy as cp
import numpy as np

arr = cp.zeros([10,10])
find_nonzero = nonzero_untidy.nonzero(arr)