"""
Solves the first order ODE:

y' + y = 0
y(0)=1

So the solution is y(x) = exp(-x)
"""
from hermes1d import Node, Element, Mesh, DiscreteProblem

from math import pi
from numpy import zeros

# interval end points
a = 0.
b = 5.

# number of elements:
N = 20

# x values of the nodes:
x_values =[(b-a)/N * i for i in range(N+1)]

# define nodes:
nodes = [Node(x) for x in x_values]

# define elements of the 1st mesh
elements = [Element(nodes[i], nodes[i+1], order=1) for i in range(N)]
m1 = Mesh(nodes, elements)
m1.set_bc(left=True, value=1)

# definition of the ODE system:
d = DiscreteProblem(meshes=[m1])

# definition of the RHS:
def F(i, Y, t):
    if i == 0:
        return -Y[0]
    raise ValueError("Wrong i (i=%d)." % (i))

# definition of the Jacobian matrix
def DFDY(i, j, Y, t):
    if i == 0 and j == 0:
        return -1
    raise ValueError("Wrong i, j (i=%d, j=%d)." % (i, j))

# assign both F and J to the discrete problem:
d.define_ode(F, DFDY)

# enumeration of unknowns:
d.assign_dofs()

Y = d.solve_Y(euler=False)

d.plot_Y(Y)
