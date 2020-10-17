hfmIn = Eikonal.dictIn({
    'verbosity':0,
    'model':'Rander2',
    'exportValues':1,
    'seed':[0,0],
    'solver':'AGSI' #TODO : why needed ?
    'metric':Rander(array([[3., 0.],[0., 3.]], dtype=float32), array([-0., -1.], dtype=float32))
})
hfmIn.SetRect([[-1,1],[-1,1]],dimx=101,sampleBoundary=True)

