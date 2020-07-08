import sys; sys.path.insert(0,"../..")

from agd import LinearParallel as lp
from agd.Selling import GatherByOffset
from agd.Plotting import savefig; #savefig.dirName = 'Figures/TensorVoronoi'

from agd.Eikonal import VoronoiDecomposition
import numpy as np
from agd.Metrics.misc import flatten_symmetric_matrix

import cupy as cp
device = cp.cuda.Device(0)
print(device.compute_capability)
#print(device.attributes)
#help(cp.cuda.Device)
print(device.mem_info)
#help(cp.cuda.device)
#help(cp.core.core.memory_module)

VoronoiDecomposition.default_mode = 'gpu_transfer'

def MakeRandomTensor(dim,shape = tuple()):
    A = np.random.standard_normal( (dim,dim) + shape )
    ident = np.eye(dim).reshape((6,6)+(1,)*len(shape))
    D = lp.dot_AA(lp.transpose(A),A)
    return  D+0.05*lp.trace(D)*ident

def Reconstruct(coefs,offsets):
     return lp.mult(coefs,lp.outer_self(offsets)).sum(2)
def LInfNorm(a):
    return np.max(np.abs(a))

np.random.seed(44)

D = MakeRandomTensor(6,(10,))
D = D[:,:,6]
print(list(flatten_symmetric_matrix(D)))
coefs,offsets = VoronoiDecomposition(D) #,traits={"debug_print":True})
#VoronoiDecomposition(D,steps="Split") #,traits={"debug_print":True})

#print("Coefficients : ", coefs)
#print("Offsets : \n", offsets.astype(int))

print("Minimal coefficient : ", np.min(coefs))
error = np.max(np.abs(D-Reconstruct(coefs,offsets)),axis=(0,1))
print(np.where(error>1))
print("Reconstruction error : ",np.max(error) )
assert np.allclose(D,Reconstruct(coefs,offsets),atol=1e-4)