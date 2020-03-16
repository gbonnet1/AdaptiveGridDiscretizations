import interface

def RunGPU(hfmIn,*args,**kwargs):
	return interface.Interface(hfmIn).Run(*args,**kwargs)