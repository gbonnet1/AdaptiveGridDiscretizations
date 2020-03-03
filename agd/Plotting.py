from os import path
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def SetTitle3D(ax,title):
	ax.text2D(0.5,0.95,title,transform=ax.transAxes,horizontalalignment='center')

def savefig(fig,fileName,dirName=None,ax=None,**kwargs):
	"""Save a figure:
	- in a given directory, possibly set in the properties of the function. 
	 Silently fails if dirName is None
	- with defaulted arguments, possibly set in the properties of the function
	"""
	# Choose the subplot to be saved 
	if ax is not None:
		kwargs['bbox_inches'] = ax.get_tightbbox(
			fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())

	# Set arguments to be passed
	for key,value in vars(savefig).items():
		if key not in kwargs and key!='dirName':
			kwargs[key]=value

	# Set directory
	if dirName is None: 
		if savefig.dirName is None: return 
		else: dirName=savefig.dirName


#		kwargs['bbox_inches']=ax.get_tightbbox(
#			fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
#		kwargs.pop('dpi',None)
#		kwargs.pop('pad_inches',None)
	
	# Save figure
	if path.isdir(dirName):
		fig.savefig(path.join(dirName,fileName),**kwargs) 
	else:
		print("savefig error: No such directory", dirName)
#		raise OSError(2, 'No such directory', dirName)

savefig.dirName = None
savefig.bbox_inches = 'tight'
savefig.pad_inches = 0
savefig.dpi = 300

def animation_curve(X,Y,**kwargs):
    """Animates a sequence of curves Y[0],Y[1],... with X as horizontal axis"""
    fig, ax = plt.subplots(); plt.close()
    ax.set_xlim(( X[0], X[-1]))
    ax.set_ylim(( np.min(Y), np.max(Y)))
    line, = ax.plot([], [])
    def func(i,Y): line.set_data(X,Y[i])
    kwargs.setdefault('interval',20)
    kwargs.setdefault('repeat',False)
    return animation.FuncAnimation(fig,func,fargs=(Y,),frames=len(Y),**kwargs)