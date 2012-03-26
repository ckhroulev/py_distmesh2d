#!/usr/bin/env python

from scipy.spatial import Delaunay
import numpy as np
from numpy import sqrt, sum
from pylab import figure, triplot, axes, show, clf

def fd(pts):
    return sqrt(pts[:,0]**2 + pts[:,1]**2) - 1.0

def fh(pts):
    return np.ones((pts.shape[0], 1))

def distmesh2d(fd, fh, h0, bbox, pfix):
    global x,y, p, pold, t, pmid, bars, barvec, L, L0, F, Fvec, Ftot, dgradx, dgrady, geps, deltat, deps, ix, d
    dptol = 0.001; ttol = 0.1; Fscale = 1.2; deltat = 0.2;
    geps = 0.001 * h0; deps = np.finfo(float).eps * h0

    def hstack(stuff):
        return np.vstack(stuff).T

    bbox = np.array(bbox)
    x, y = np.meshgrid(np.arange(bbox[0,0], bbox[0,1], h0),
                       np.arange(bbox[1,0], bbox[1,1], h0 * sqrt(3) / 2))

    x[1::2,:] += h0 / 2

    p = np.array((x.flatten(), y.flatten())).T

    p = p[fd(p) < geps]
    r0 = 1.0 / fh(p)**2
    selection = np.random.rand(p.shape[0], 1) < r0 / r0.max()
    p = p[selection[:,0]]
    if len(pfix) > 0:
        p = np.vstack((pfix, p))

    N = p.size

    pold = np.inf

    while True:
        if sqrt(sum((p - pold)**2, 1)).max() > ttol:
            pold = p
            t = np.sort(Delaunay(p).vertices, axis=1)
            pmid = sum(p[t], 1) / 3
            t = t[fd(pmid) < -geps]
            bars = t[:, [[0,1], [1,2], [0,2]]].reshape((-1, 2))
            bars = np.unique(bars.view("i,i")).view("i").reshape((-1,2))

            triplot(p[:,0], p[:,1], t, "k-")
            show()

        barvec = p[bars[:,0]] - p[bars[:,1]]
        L = sqrt(sum(barvec**2, 1)).reshape((-1,1))
        hbars = fh((p[bars[:,0]] + p[bars[:,1]]) / 2.0)
        L0 = hbars * Fscale * sqrt(sum(L**2) / sum(hbars**2))

        F = np.maximum(L0 - L, 0)
        Fvec = F * (barvec / L)

        Ftot = np.zeros_like(p)
        for j in xrange(bars.shape[0]):
            Ftot[bars[j]] += [Fvec[j], -Fvec[j]]

        p += deltat * Ftot

        d = fd(p); ix = d > 0
        dgradx = (fd(hstack((p[ix,0] + deps, p[ix,1])))        - d[ix]) / deps
        dgrady = (fd(hstack((p[ix,0],        p[ix,1] + deps))) - d[ix]) / deps

        p[ix] -= hstack((d[ix] * dgradx, d[ix] * dgrady))

        if np.max(np.sum((deltat * Ftot[d < -geps])**2, 1) / h0) < dptol:
            break

    return p, t

figure()

pts, tri = distmesh2d(fd, fh, 0.2, [[-1, 1], [-1, 1]], [])

axes().set_aspect('equal')

show()
