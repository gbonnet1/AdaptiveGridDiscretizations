# Code automatically exported from notebook Notebooks_GPU/Flow_Repro.ipynb
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
from agd import FiniteDifferences as fd
from agd import LinearParallel as lp
import agd.AutomaticDifferentiation.cupy_generic as cugen
norm_infinity = ad.Optimization.norm_infinity

from Notebooks_GPU.ExportedCode.Isotropic_Repro import RunCompare
from Notebooks_NonDiv.ExportedCode.LinearMonotoneSchemes2D import streamplot_ij

