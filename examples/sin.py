"""
Solves the second order ODE:

y'' + y = 0
y(0)=0; y'(0)=1

So the solution is y(x) = sin(x)
"""
from hermes1d import Node, Element, Mesh, DiscreteProblem

from numpy import zeros
a = 0.
b = 4.
N = 100
nodes = [Node((b-a)/N * i) for i in range(N)]
elements = [Element(nodes[i], nodes[i+1], order=1) for i in range(N-1)]
m1 = Mesh(nodes, elements)
m1.set_bc(left=True, value=0)
elements = [Element(nodes[i], nodes[i+1], order=1) for i in range(N-1)]
m2 = Mesh(nodes, elements)
m2.set_bc(left=True, value=1)

d = DiscreteProblem(meshes=[m1, m2])
k = 1.0
def J(i, j):
    def f11(y1, y2, t):
        return 0
    def f12(y1, y2, t):
        return 1
    def f21(y1, y2, t):
        return -1
    def f22(y1, y2, t):
        return 0
    if i == 0 and j == 0:
        return f11
    elif i == 0 and j == 1:
        return f12
    elif i == 1 and j == 0:
        return f21
    elif i == 1 and j == 1:
        return f22
    raise ValueError("Wrong i, j (i=%d, j=%d)." % (i, j))
def Phi(i, U, Z, t):
    if i == 0:
        return -U[0]+U[1] - Z[0]
    elif i == 1:
        return U[1] - Z[1]
    raise ValueError("Wrong i (i=%d)." % (i))
def dPhi_dy(i, j, U, Z, t):
    if i == 0:
        if j == 0: return -1
        elif j == 1: return 1
    elif i == 1:
        if j == 0: return 0.
        elif j == 1: return 1
def dPhi_dz(i, j, U, Z, t):
    if i == 0:
        if j == 0: return -1
        elif j == 1: return 0
    elif i == 1:
        if j == 0: return 0
        elif j == 1: return -1
def F(i):
    def f1(y1, y2, t):
        return y2
    def f2(y1, y2, t):
        return -y1
    if i == 0:
        return f1
    elif i == 1:
        return f2
    raise ValueError("Wrong i (i=%d)." % (i))
d.set_rhs(Phi, dPhi_dy, dPhi_dz)
d.assign_dofs()
J = d.assemble_J()
Y = zeros((J.shape[0],))
error = 1e10
i = 0
while error > 1e-3:
    F = d.assemble_F(Y)
    dY = d.solve(J, F)
    error = d.calculate_error_l2_norm(dY)
    print "it=%d, l2_norm=%e" % (i, error)
    Y += dY
    i += 1
x = Y
#print
#print J
#print F
#print x

from pylab import plot, legend, show
sln1, sln2 = d.linearize(x, 5)
x1, y1 = sln1
x2, y2 = sln2
plot(x1, y1, label="$u_1$")
plot(x2, y2, label="$u_2$")
legend()
show()
