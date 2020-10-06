import sys; sys.path.insert(0,"../..")

from agd import Eikonal
from agd import AutomaticDifferentiation as ad
from agd import Metrics
from agd import LinearParallel as lp
from agd.Interpolation import map_coordinates
from agd.AutomaticDifferentiation.Optimization import norm

from agd.ExportedCode.Notebooks_Algo import RollingBall_Models

import numpy as np; xp=np; allclose = np.allclose
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

xp,Eikonal,plt,allclose = [ad.cupy_friendly(e) for e in (xp,Eikonal,plt,allclose)]

plane_to_sphere = RollingBall_Models.quaternion_from_euclidean
sphere_to_plane = RollingBall_Models.euclidean_from_quaternion

def glue(x): 
    """
    Glue map for the projective plane, as parameterized from the equator.
    (Maps a point to the other point with which it identifies.)
    """    
    return sphere_to_plane( - plane_to_sphere(x))


def metric(x):
    """
    Pulls back the Euclidean metric on the sphere onto the plane.
    """
    x_ad = ad.Dense.identity(constant=x,shape_free=(len(x),))
    Jac = np.moveaxis(plane_to_sphere(x_ad).gradient(),0,1)
    return lp.dot_AA(lp.transpose(Jac),Jac)

def cost(x):
    """
    Cost function associated with the conformal metric of the stereographic projection.
    """
    return np.sqrt(metric(x)[0,0]) # Alternatively, use the explicit expression 2/sqrt(1+ |x|^2)

hfmIn = Eikonal.dictIn({
    'model':'Isotropic2',
    'seed':[0.5,0.7],
    'exportValues':True,
    'factoringRadius':10,
})
r=1.1
hfmIn.SetRect([[-r,r],[-r,r]],dimx=101)
X = hfmIn.Grid()

hfmIn['cost']=cost(X)
#hfmIn.SetUniformTips((4,4))

if False:
	# ----------------- No glue ---------------
	hfmOut = hfmIn.Run()

	values = hfmOut['values']
	glued_values = map_coordinates(values,glue(X),grid=X,cval=np.inf)

	plt.figure(figsize=(16,7))
	plt.subplot(1,2,1)
	plt.title('Values of the solution')
	plt.contourf(*X,values) 
	plt.axis('equal')

	plt.subplot(1,2,2)
	plt.title('Glued values')
	plt.contourf(*X,glued_values) 
	plt.axis('equal');

	plt.show()

if True:
	# Second 
	hfmIn['chart_mapping']=glue(X)
	hfmIn['chart_nitermax']=1
	# chart_jump also, and we're done
#	hfmIn['chart'] = {
#		'mapping':glue(X),
#		'paste':xp.full(hfmIn.shape,True,dtype=bool),
#		'niter':1,
#		'jump':norm(X)>=1.05
#	}

	hfmOut = hfmIn.Run()
	values = hfmOut['values']

	plt.figure(figsize=(16,7))
	plt.subplot(1,2,1)
	plt.title('Values of the solution')
	plt.contourf(*X,values) 
	plt.axis('equal')

	plt.show()
