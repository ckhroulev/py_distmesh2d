from pylab import figure, triplot, tripcolor, axis, axes, show, hold, plot
from py_distmesh2d import *
import numpy as np

def example1(pts):
    return dcircle(pts, 0, 0, 1)

def example2(pts):
    return ddiff(dcircle(pts, 0, 0, 0.7), dcircle(pts, 0, 0, 0.3))

def example3(pts):
    return ddiff(drectangle(pts, -1, 1, -1, 1), dcircle(pts, 0, 0, 0.4))

def example3_h(pts):
    return np.minimum(4*np.sqrt(sum(pts**2, 1)) - 1, 2)

def example3_online(pts):
    return ddiff(drectangle(pts, -1, 1, -1, 1), dcircle(pts, 0, 0, 0.5))

def example3_online_h(pts):
    return 0.05 + 0.3 * dcircle(pts, 0, 0, 0.5)

def annulus_h(pts):
    return 0.04 + 0.15 * dcircle(pts, 0, 0, 0.3)

def star(pts):
    return dunion(dintersect(dcircle(pts, np.sqrt(3), 0, 2), dcircle(pts, -np.sqrt(3), 0, 2)),
                  dintersect(dcircle(pts, 0, np.sqrt(3), 2), dcircle(pts, 0, -np.sqrt(3), 2)))

def circle_h(pts):
    return 0.1 - example1(pts)

def plot_mesh(pts, tri, *args):
    if len(args) > 0:
        tripcolor(pts[:,0], pts[:,1], tri, args[0], edgecolor='black', cmap="Blues")
    else:
        triplot(pts[:,0], pts[:,1], tri, "k-", lw=2)
    axis('tight')
    axes().set_aspect('equal')

def plot_nodes(pts, mask, *args):
    boundary = pts[mask == True]
    interior = pts[mask == False]
    plot(boundary[:,0], boundary[:,1], 'o', color="red")
    plot(interior[:,0], interior[:,1], 'o', color="white")
    axis('tight')
    axes().set_aspect('equal')

bbox = [[-1, 1], [-1, 1]]
square = [[-1,-1], [-1,1], [1,-1], [1,1]]

# example 1a
def example_1a():
    figure()
    pts, tri = distmesh2d(example1, huniform, 0.4, bbox, [])
    plot_mesh(pts, tri)
    show()

# example 1b
def example_1b():
    figure()
    pts, tri = distmesh2d(example1, huniform, 0.2, bbox, [])
    plot_mesh(pts, tri)
    show()

# example 1c
def example_1c():
    figure()
    pts, tri = distmesh2d(example1, huniform, 0.1, bbox, [])
    plot_mesh(pts, tri)
    show()

# example 2
def example_2():
    figure()
    pts, tri = distmesh2d(example2, huniform, 0.1, bbox, [])
    plot_mesh(pts, tri)
    show()

# example 3a
def example_3a():
    figure()
    pts, tri = distmesh2d(example3, huniform, 0.15, bbox, square)
    plot_mesh(pts, tri, example3(pts))
    show()

# example 3b
def example_3b():
    figure()
    pts, tri = distmesh2d(example3, example3_h, 0.035, bbox, square)
    plot_mesh(pts, tri)
    show()

# example (current online version)
def example_3_online():
    figure()
    pts, tri = distmesh2d(example3_online, example3_online_h, 0.02, bbox, square)
    boundary = boundary_mask(pts, example3_online, 0.02)
    hold(True)
    plot_mesh(pts, tri)
    plot_nodes(pts, boundary)
    show()

# annulus, non-uniform
def annulus():
    figure()
    pts, tri = distmesh2d(example2, annulus_h, 0.04, bbox, square)
    boundary = boundary_mask(pts, example2, 0.04)
    hold(True)
    plot_mesh(pts, tri)
    plot_nodes(pts, boundary)
    show()

# a "star" built using circles
def star_mesh():
    figure()
    # fake the corners:
    pfix = [[0.25, 0.25], [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25]]
    pts, tri = distmesh2d(star, huniform, 0.1, bbox, pfix)
    boundary = boundary_mask(pts, star, 0.5) # note how large h0 has to be here
    print star(np.array(pfix))
    hold(True)
    plot_mesh(pts, tri)
    plot_nodes(pts, boundary)
    show()

# a circle, finer mesh near the boundary
def circle_nonuniform():
    figure()
    # fake the corners:
    pts, tri = distmesh2d(example1, circle_h, 0.1, bbox, [])
    plot_mesh(pts, tri)
    show()

def ell():
    """L-shaped domain from 'Finite Elements and Fast Iterative Solvers'
    by Elman, Silvester, and Wathen."""

    pfix = [[1,1], [1, -1], [0, -1], [0, 0], [-1, 0], [-1, 1]]

    def d(pts):
        return ddiff(drectangle(pts, -1, 1, -1, 1), drectangle(pts, -2, 0, -2, 0))

    figure()
    pts, tri = distmesh2d(d, huniform, 0.1, bbox, pfix)
    plot_mesh(pts, tri)
    show()
