import sys; sys.path.insert(0,"../..")
#from Miscellaneous import TocTools; print(TocTools.displayTOC('Flow_GPU','Repro'))

from agd import Eikonal
from agd import AutomaticDifferentiation as ad
from agd import FiniteDifferences as fd
from agd import Metrics
from agd import LinearParallel as lp
norm = ad.Optimization.norm

import numpy as np; xp=np; allclose = np.allclose
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

xp,Eikonal,plt,allclose = [ad.cupy_friendly(e) for e in (xp,Eikonal,plt,allclose)]

n=45
hfmIn = Eikonal.dictIn({
    'model':'ReedsSheppGPU3',
    'cost':1.,
    'seed':[0,0,0,0,0],
    
    # Define euclidean geometry on R^3 x S^2, for illustration
    'eps':1,
    'xi':1,
    'traits':{'decomp_v_align_macro':False},
    
    # Recompute flow at geodesic extraction
    'geodesic_online_flow':True,

#    'dual':True,
#    'forward':True,    
})
hfmIn.SetRect([[-1,1],[-1,1],[-1,1]],dimx=n)
transformations = hfmIn.SetSphere(dimsp=32,separation=False) # projective model
hfmIn.SetUniformTips((3,3,3,3,3))

print("Number of unknowns : ", np.round(hfmIn.size / 10**9,decimals=2), "*10^9")

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

for key,value in hfmIn.items(): print(key,sys.getsizeof(value))
tips=hfmIn['tips']
print("tips",tips.size,sys.getsizeof(tips))

mempool = xp.get_default_memory_pool()
print(mempool.used_bytes())
print(mempool.get_limit())
#help(mempool)

#hfmOut = hfmIn.Run()

def GPUArrays(data,iterables=tuple()):
	gpu_arrays = []
	def check(name,arr):
		if ad.cupy_generic.from_cupy(arr):
			for namelist,value in gpu_arrays:
				if arr.data==value.data:
					namelist.append(name)
					break
			else:
				gpu_arrays.append(([name],arr))

	def check_members(prefix,data):
		for name,value in prepare(data).items():
			name2 = prefix+(name,)
			if isinstance(value,iterables): check_members(name2,value)
			else: check(name2,value)

	def prepare(data):
		if isinstance(data,(list,tuple)): 
			return {i:value for i,value in enumerate(data)}
		elif isinstance(data,SimpleNamespace): return data.__dict__
		else: return data
#		else: return data.__dict__

	check_members(tuple(),data)
	return gpu_arrays

print("---")

hfmIn['mydata'] = [hfmIn['tips'],hfmIn['chart_mapping']]

from types import SimpleNamespace
data = SimpleNamespace() 
data.a = hfmIn['cost']

gpu_arrays = GPUArrays([hfmIn,locals(),data],iterables=(Eikonal.dictIn,dict,list,SimpleNamespace))
for namelist,value in gpu_arrays:
	print(namelist, " with size ", value.nbytes)
#print(GPUArrays(hfmIn,iterables=(dict,)))



