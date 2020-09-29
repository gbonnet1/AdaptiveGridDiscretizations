import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory 

from agd import Eikonal
import numpy as np; xp=np
import time
from agd.AutomaticDifferentiation.Optimization import norm_infinity
from agd import AutomaticDifferentiation as ad

xp,Eikonal = [ad.cupy_friendly(e) for e in (xp,Eikonal)]

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%5.3g" % x))

"""
#n=20; nTheta=60
n=100; nTheta=96
hfmIn = Eikonal.dictIn({
    'model':'Dubins2',
    'seed':[0.,0.,np.pi],
    'exportValues':1,
    'cost':1,
    'xi':0.4,
#    'nitermax_o':1,
#    'traits':{
#    	'OffsetT':np.int32,
#    	'merge_sort_macro':1
#    	}
})
hfmIn.SetRect([[-1,1],[-1,1]],dimx=n+1,sampleBoundary=True)
hfmIn.nTheta = nTheta
"""
xi=0.3
hfmIn = Eikonal.dictIn({
    'model':'Dubins2', # Dubins model, extended (customizable) variant    
    'exportValues':1,
    'xi':xi, # Bound on the radius of curvature
    'speed':1.,    
    
    'seed':(0,0,0), # Central seed, with horizontal tangent
#    'tips':[(np.cos(t),np.sin(t),0) for t in np.linspace(0,2*np.pi,20)], # Tips on circle, with horizontal tangents    
})
hfmIn.SetRect(sides=[[-1.5,1.5],[-1.5,1.5]],dimx = 151) # Physical domain
hfmIn.nTheta = 96 # Angular resolution

hfmOut=hfmIn.Run()
#print(hfmOut['scheme_offsets'][0])
