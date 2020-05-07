eikonal_mode = 'gpu'
import sys; sys.path.insert(0,"../..") # Allow import of agd from parent directory (useless if conda package installed)
#from Miscellaneous import TocTools; print(TocTools.displayTOC('DubinsZermelo','FMM'))

from agd import Eikonal
from agd.Plotting import savefig, quiver; #savefig.dirName = 'Figures/DubinsZermelo'
from agd import AutomaticDifferentiation as ad

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def SetDevice():
    global xp,plt,quiver
    if eikonal_mode == 'cpu': xp = np
    else:
        import cupy
        xp = ad.functional.decorate_module_functions(cupy,ad.cupy_generic.set_output_dtype32) 
        plt = ad.functional.decorate_module_functions(plt,ad.cupy_generic.cupy_get_args)
        quiver = ad.cupy_generic.cupy_get_args(quiver)
        Eikonal.dictIn.default_mode = eikonal_mode
        ad.array.caster = lambda x:cupy.asarray(x,dtype=np.float32)
SetDevice()

xi = 0.3

hfmIn = Eikonal.dictIn({
    'model':'DubinsExt2', # Dubins model, extended (customizable) variant    
    'exportValues':1,
    'xi':xi, # Bound on the radius of curvature
    'speed':1.,    
    
    'seed':(0,0,0), # Central seed, with horizontal tangent
    'tips':[(np.cos(t),np.sin(t),0) for t in np.linspace(0,2*np.pi,20)], # Tips on circle, with horizontal tangents    
})
hfmIn.SetRect(sides=[[-1.5,1.5],[-1.5,1.5]],dimx = 151) # Physical domain
hfmIn.nTheta = 96 # Angular resolution
#hfmIn['stopWhenAllAccepted'] = hfmIn['tips'] # Save a little bit of CPU time with early abort

X,Y,Theta = hfmIn.Grid()
drift = [0.3,0.4]
velocity = ad.asarray([drift[0]+np.cos(Theta), drift[1]+np.sin(Theta)])

hfmIn.update({
    'speed':np.linalg.norm(velocity,axis=0), # total velocity norm
    'theta':np.arctan2(velocity[1],velocity[0]), # total velocity orientation
})
hfmIn['xi'] = xi * hfmIn['speed'] # Needed to enforce the curvature bound constraint in the moving frame

hfmIn.update({
    'seed_Unoriented':[0,0],
    'tips_Unoriented':[(np.cos(t),np.sin(t)) for t in np.linspace(0,2*np.pi,20)] 
})


hfmOut = hfmIn.Run()