import sys; sys.path.insert(0,"../../..") # Allow import of agd from parent directory 

from agd import HFMUtils
from agd.HFMUtils import HFM_CUDA
import cupy as xp


hfmIn = HFMUtils.dictIn({
    'model':'Isotropic2',
    'arrayOrdering':'RowMajor',
    'seeds':[[0,0]],
    'kernel':"dummy",
    'solver':'globalIteration',
    'verbosity':1,
})
hfmIn.SetRect([[-1,1],[-1,1]],dimx=8)
hfmIn['cost'] = xp.ones(hfmIn['dims'].astype(int),dtype='float32')


hfmOut = hfmIn.RunGPU(returns='in_raw')

print(hfmOut['source'])
print(hfmOut)