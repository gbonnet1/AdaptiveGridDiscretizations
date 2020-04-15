# Code automatically exported from notebook Notebooks_GPU/Isotropic_Repro.ipynb
# Do not modify
import sys; sys.path.append("../..") # Path to import agd

import cupy as cp
import numpy as np
import itertools
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

from agd import HFMUtils
from agd import AutomaticDifferentiation as ad
from agd import Metrics
import agd.AutomaticDifferentiation.cupy_generic as cugen
norm_infinity = ad.Optimization.norm_infinity
#from agd.HFMUtils import RunGPU,RunSmart

cp = ad.functional.decorate_module_functions(cp,cugen.set_output_dtype32) # Use float32 and int32 types in place of float64 and int64
plt = ad.functional.decorate_module_functions(plt,cugen.cupy_get_args)
HFMUtils.dictIn.RunSmart = cugen.cupy_get_args(HFMUtils.RunSmart,dtype64=True,iterables=(dict,Metrics.Base))
#RunSmart = cugen.cupy_get_args(RunSmart,dtype64=True,iterables=(dict,Metrics.Base))

factor_variants = [
    {}, # Default
    {"seedRadius":2.}, # Spread seed information
    {"factorizationRadius":10,'factorizationPointChoice':'Key'} # Source factorization
]
multip_variants = [
    {'multiprecision':False}, # Default
    {'multiprecision':True}, # Reduces roundoff errors
]

def RunCompare(gpuIn,check=True,variants=None,**kwargs):
    # Dispatch the common variants if requested
    if variants:
        order2 = kwargs.get('order',1)==2 or gpuIn.get('order',1)==2
        for fact in (factor_variants[0],factor_variants[2]) if order2 else factor_variants:
            for multip in multip_variants:
                print(f"\n--- Variant with {fact} and {multip} ---")
                RunCompare(gpuIn,check,**fact,**multip,**kwargs)
        return # variants

    # Run the CPU and GPU solvers
    gpuIn = gpuIn.copy(); gpuIn.update(kwargs)
    gpuOut = gpuIn.RunGPU()
    if gpuIn.get('verbosity',1):  print(f"--- gpu done, turning to cpu ---")
    cpuIn = gpuIn.copy(); 
    for key in ('traits','array_float_caster'): cpuIn.pop(key,None)
    cpuOut = cpuIn.RunSmart()
    
    # Print performance info
    fmTime = cpuOut['FMCPUTime']; stencilTime = cpuOut['StencilCPUTime']; 
    cpuTime = fmTime+stencilTime; gpuTime = gpuOut['solverGPUTime'];
    print(f"Solver time (s). GPU : {gpuTime}, CPU : {cpuTime}. Device acceleration : {cpuTime/gpuTime}")
    
    # Check consistency 
    cpuVals = cpuOut['values']; gpuVals = gpuOut['values'].get()
    print("Max |gpuValues-cpuValues| : ", norm_infinity(gpuVals-cpuVals))
    
    if check is True: assert np.allclose(gpuVals,cpuVals,atol=1e-5,rtol=1e-4)
    elif check is False: pass
    else: assert norm_infinity(gpuVals-cpuVals)<check

    return gpuOut,cpuOut

