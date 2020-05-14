# Adaptive Grid Discretizations using Lattice Basis Reduction (AGD-LBR)
## A set of tools for discretizing anisotropic PDEs on cartesian grids

This repository contains
- the agd library (Adaptive Grid Discretizations)
- a series of *jupyter notebooks* in the Python&reg; language, reproducing my research in Anisotropic PDE discretizations and their applications.

### The AGD library

The recommended way to install is
```console
conda install agd -c agd-lbr --force
```

### The notebooks

The notebooks are intended as documentation and testing for the adg library. They encompass:
* Anisotropic fast marching methods, for shortest path computation.
* Non-divergence form PDEs, including non-linear PDEs such as Monge-Ampere.
* Divergence form anisotropic PDEs, often encountered in image processing.
* Algorithmic tools, related with lattice basis reduction methods, and automatic differentiation.

The notebooks can be visualized online, [view summary online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations_showcase/master/Summary.ipynb
), or executed and/or modified offline.
For offline consultation, please download and install [anaconda](https://www.anaconda.com) or [miniconda](https://conda.io/en/latest/miniconda.html).  
*Optionally*, you may create a dedicated conda environnement by typing the following in a terminal:
```console
conda env create --file agd-hfm.yaml
conda activate agd-hfm
```
In order to open the book summary, type in a terminal:
```console
jupyter notebook Summary.ipynb
```
Then use the hyperlinks to navigate within the notebooks.
