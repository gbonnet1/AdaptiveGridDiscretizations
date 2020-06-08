import sys; sys.path.insert(0,"../..")

from agd import LinearParallel as lp
from agd.Selling import GatherByOffset
from agd.Plotting import savefig; #savefig.dirName = 'Figures/TensorVoronoi'

from agd.Eikonal import VoronoiDecomposition
import numpy as np

VoronoiDecomposition.default_mode = 'gpu_transfer'

def MakeRandomTensor(dim,shape = tuple()):
    A = np.random.standard_normal( (dim,dim) + shape )
    return lp.dot_AA(lp.transpose(A),A)

def Reconstruct(coefs,offsets):
     return lp.mult(coefs,lp.outer_self(offsets)).sum(2)
def LInfNorm(a):
    return np.max(np.abs(a))

np.random.seed(42)

D = MakeRandomTensor(5)
coefs,offsets = VoronoiDecomposition(D)
#VoronoiDecomposition(D,steps="Split")

print("Coefficients : ", coefs)
print("Offsets : \n", offsets.astype(int))

print("Minimal coefficient : ", np.min(coefs))
print("Reconstruction error : ", LInfNorm(D-Reconstruct(coefs,offsets)))
assert np.allclose(D,Reconstruct(coefs,offsets))