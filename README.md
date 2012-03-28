This repository contains a Python re-implementation of `distmesh2d` in *P.-O. Persson, G. Strang, A Simple Mesh Generator in MATLAB. SIAM Review, Volume 46 (2), pp. 329-345, June 2004* (http://persson.berkeley.edu/distmesh/).

* This code implements the original function described in the 2004 paper and
  does not include any improvements,
* Using NumPy makes it about as short as the original MATLAB version.
* `scipy.spatial.Delaunay` (present in SciPy >= 0.9.0) is used to compute
   Delaunay triangulations. Alternatively `matplotlib.delaunay` (present in
   `matplotlib` >= 0.8) can be used.
* The script `examples.py` requires `matplotlib` >= 1.0.
* This code is in an *"alpha"* stage and will probably stay that way.

It seems to work, though:

![example 3](py_distmesh2d/raw/master/example3.png)
