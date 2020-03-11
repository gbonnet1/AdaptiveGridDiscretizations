import cupy as cp
from matplotlib import pyplot as plt
import os
from itertools import chain

# GPUs adapted numeric types
float_t = cp.dtype('float32').type
int_t = cp.dtype('int32').type

def packbits(arr,bitorder='big'):
	# Implements bitorder option, supported by numpy but not cupy
	if bitorder=='little':
		shape = arr.shape
		arr = arr.reshape(-1,8)
		arr = arr[:,::-1]
		arr = arr.reshape(shape)
	return cp.packbits(arr)



with open("../IsotropicUpdate.h",'r') as f:
	loaded_from_source = f.read()
    
cuoptions = ("-default-device",)
isotropic_update = cp.RawKernel(loaded_from_source,'IsotropicUpdate',options=cuoptions)


shape_i = (8,8)
size_i = 64
u    = cp.full( (size_i,), cp.inf, dtype=float_t)
u[0]=0.
cost = cp.full( (size_i,), 1.,     dtype = float_t)
seeds = packbits(u<cp.inf,bitorder='little')
shape = cp.array( (4,4), dtype=int_t)
x_o = cp.array( (0,0), dtype=int_t)
min_chg = cp.full( (1,), 0., dtype=float_t)
tol = float_t(1e-8)

isotropic_update((1,),shape_i,(u,cost,seeds,shape,x_o,min_chg,tol))
cp.cuda.Stream.null.synchronize()
print()
u=u.reshape(shape_i);
print(u)
print(u-u.T)
print(f"min change {min_chg}")