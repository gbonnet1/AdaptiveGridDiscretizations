import sys; sys.path.insert(0,"../..")
#import cupy as cp
import numpy as np

from agd import AutomaticDifferentiation as ad
from agd.Eikonal.HFM_CUDA.VoronoiDecomposition import VoronoiDecomposition
from agd import Metrics
xp = ad.cupy_generic.cupy_friendly(np)

#m = cp.eye(4,dtype=np.float32)
m = Metrics.Riemann.needle([0.1,7.3,2.4,5.8,1.6],1.,0.1).dual().m
print(m)
weights,offsets = VoronoiDecomposition(m)
print(f"weights : {weights}")
print(f"offsets : {offsets}")