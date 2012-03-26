#!/usr/bin/env python

from scipy.spatial import Delaunay
import numpy as np
from numpy import sqrt, sum
from pylab import figure, triplot, axis, axes, show, clf, vstack

def dcircle(pts, *args):
    return sqrt(pts[:,0]**2 + pts[:,1]**2) - 1.0

def drectangle(pts, x1, x2, y1, y2):
    return -np.min(np.min(np.min(-y1+p[:,1], y2-p[:,1]),
                          -x1+p[:,0], x2-p[:,0]))

def example1(pts, *args):
    return sqrt(sum(pts**2, 1)) - 1

def example2(pts, *args):
    return -0.3 + np.abs(0.7 - sqrt(sum(pts**2, 1)))

def huniform(pts, *args):
    return np.ones((pts.shape[0], 1))

def distmesh2d(fd, fh, h0, bbox, pfix, *args):
    dptol = 0.001; ttol = 0.1; Fscale = 1.2; deltat = 0.2;
    geps = 0.001 * h0; deps = sqrt(np.finfo(float).eps) * h0

    bbox = np.array(bbox)
    x, y = np.meshgrid(np.arange(bbox[0,0], bbox[0,1], h0),
                       np.arange(bbox[1,0], bbox[1,1], h0 * sqrt(3) / 2))

    x[1::2,:] += h0 / 2

    p = np.array((x.flatten(), y.flatten())).T

    p = p[fd(p, *args) < geps]
    r0 = 1.0 / fh(p, *args)**2
    selection = np.random.rand(p.shape[0], 1) < r0 / r0.max()
    p = p[selection[:,0]]
    if len(pfix) > 0:
        p = np.vstack((pfix, p))

    pold = np.zeros_like(p)
    Ftot = np.zeros_like(p)

    while True:
        if sqrt(sum((p - pold)**2, 1)).max() > ttol:
            pold[:] = p[:]
            t = np.sort(Delaunay(p).vertices, axis=1)
            pmid = sum(p[t], 1) / 3
            t = t[fd(pmid, *args) < -geps]
            bars = t[:, [[0,1], [1,2], [0,2]]].reshape((-1, 2))
            bars = np.unique(bars.view("i,i")).view("i").reshape((-1,2))
            print ":"

        barvec = p[bars[:,0]] - p[bars[:,1]]
        L = sqrt(sum(barvec**2, 1)).reshape((-1,1))
        hbars = fh((p[bars[:,0]] + p[bars[:,1]]) / 2.0, *args)
        L0 = hbars * Fscale * sqrt(sum(L**2) / sum(hbars**2))

        F = np.maximum(L0 - L, 0)
        Fvec = F * (barvec / L)

        Ftot[:] = 0
        for j in xrange(bars.shape[0]):
            Ftot[bars[j]] += [Fvec[j], -Fvec[j]]

        p += deltat * Ftot

        d = fd(p, *args); ix = d > 0
        dgradx = (fd(vstack((p[ix,0] + deps, p[ix,1])).T, *args)        - d[ix]) / deps
        dgrady = (fd(vstack((p[ix,0],        p[ix,1] + deps)).T, *args) - d[ix]) / deps

        p[ix] -= vstack((d[ix] * dgradx, d[ix] * dgrady)).T

        print "."
        if (sqrt(sum((deltat * Ftot[d < -geps])**2, 1)) / h0).max() < dptol:
            break

    return p, t

figure()

h0 = 0.1
bbox = [[-1, 1], [-1, 1]]
pfix = [[-1,-1], [-1,1], [1,-1], [1,1]]
pts, tri = distmesh2d(example1, huniform, h0, bbox, [])

triplot(pts[:,0], pts[:,1], tri, "b-", lw=2)

axis('tight')
axes().set_aspect('equal')

show()
