import sys; sys.path.insert(0,"../../..") # Allow import of agd from parent directory 
from agd.HFMUtils.HFM_CUDA import nonzero_untidy

import cupy as cp
import numpy as np

shape=(10,10)
arr = cp.zeros(shape)
find_nonzero = nonzero_untidy.nonzero(arr,log2_size_i=4,size2_i=8)

arr=arr.reshape(-1)
arr[[2,5,9,17,23,56,74,85]]=1
arr=arr.reshape(shape)
#arr.flatten()[0]=1
print(arr)

nz,count = find_nonzero()
print(f"nz : {nz}, count : {count}")