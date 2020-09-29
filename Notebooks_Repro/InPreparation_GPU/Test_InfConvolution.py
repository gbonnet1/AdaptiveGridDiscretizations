import sys; sys.path.insert(0,"../..")
import cupy as cp
import numpy as np

from agd.HFMUtils.HFM_CUDA import inf_convolution

ndim=2
kernel = inf_convolution.distance_kernel(radius=1,ndim=ndim,dtype=np.uint8,mult=5)
arr = cp.full((10,)*ndim,255,dtype=np.uint8)
arr[5,5]=0
out = inf_convolution.inf_convolution(arr,kernel,niter=4,upper_saturation=255)
print(kernel)
print(arr)
print(out)
