This repository contains a Python re-implementation of `distmesh2d` in *P.-O. Persson, G. Strang, A Simple Mesh Generator in MATLAB. SIAM Review, Volume 46 (2), pp. 329-345, June 2004* (http://persson.berkeley.edu/distmesh/).

This code

*  implements the original function described in the paper and does not include any improvements,
*  uses NumPy (arrays), SciPy (Delaunay triangulation) and matplotlib for plotting,
*  is in an "alpha" stage and will probably stay that way.

It seems to work, though:

![example 3](py_distmesh2d/raw/master/example3.png)
