"""
Solves the first order ODE:

y' + y = 0
y(0)=1

So the solution is y(x) = exp(-x)
"""
from hermes1d import Node, Element, Mesh, DiscreteProblem

from math import pi
from numpy import zeros

def F(i, Y, t):
    if i == 0:
        return -Y[0]
    raise ValueError("Wrong i (i=%d)." % (i))

def DFDY(i, j, Y, t):
    if i == 0 and j == 0:
        return -1
    raise ValueError("Wrong i, j (i=%d, j=%d)." % (i, j))

# interval end points
a = 0.
b = 5.

# number of elements:
N = 4

# x values of the nodes:
x_values =[(b-a)/N * i for i in range(N+1)]

# define nodes:
nodes = [Node(x) for x in x_values]

# define elements of the 1st mesh
elements = [Element(nodes[i], nodes[i+1], order=1) for i in range(N)]

def calculate_sln(F, DFDY, mesh):
    d = DiscreteProblem(meshes=[mesh])
    d.define_ode(F, DFDY)
    d.assign_dofs()
    return d.solve_Y(euler=False, verbose=False), d

m = Mesh(nodes, elements)
m.set_bc(left=True, value=1)
rm = m.copy()
rm.refine_all_elements(increase_porder=True)

Y, d = calculate_sln(F, DFDY, m)
rY, rd = calculate_sln(F, DFDY, rm)

from pylab import plot, legend, show, clf, axis
sln1, = d.linearize(Y, 5)
x1, y1 = sln1
plot(x1, y1, label="$u_1$")
sln1, = rd.linearize(rY, 5)
x1, y1 = sln1
plot(x1, y1, label="$u_2$")
legend()
show()

