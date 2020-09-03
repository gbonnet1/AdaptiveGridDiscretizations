import cupyx.scipy.ndimage
import scipy.ndimage
import cupy as cp
z = cp.ones((10,10))
x = cp.ones((2,10,10))
scipy.ndimage.map_coordinates(z.get(),x.get())
cupyx.scipy.ndimage.map_coordinates(z,x.reshape((2,-1))) # OK with flattened coordinates array
#cupyx.scipy.ndimage.map_coordinates(z,x) # ! fails !

cp.show_config()