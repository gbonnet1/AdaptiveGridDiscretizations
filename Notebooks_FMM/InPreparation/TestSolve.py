import numpy as np
import cupy as cp
import time

n=10000
a = cp.empty((n,2,2),dtype=np.float32)
v = cp.empty((n,2),dtype=np.float32)

a[:,0,0]=a[:,1,1]=1; a[:,1,0]=a[:,0,1]=0
v[0]=0; v[1]=1

t0=time.time()
np.linalg.solve(a,v)
t1=time.time()
np.linalg.solve(a.get(),v.get())
t2=time.time()

t_cupy,t_numpy=t1-t0,t2-t1
print(f"t_cupy={t_cupy}, t_numpy={t_numpy}, ratio={t_cupy/t_numpy}")

cp.show_config()