import time
from dolfin import *
from mpi4py import MPI

start = time.time()

# Create mesh and define function space
mesh = UnitSquareMesh(200, 200)

# xdmf = XDMFFile(mesh.mpi_comm(), "tmp/mesh.xdmf")
# xdmf.write(mesh)


# mesh = Mesh()
# xdmf = XDMFFile(mesh.mpi_comm(), "tmp/mesh.xdmf")
# xdmf.read(mesh)

# mesh = Mesh()
# hdf5 = HDF5File(MPI.COMM_WORLD, "tmp/mesh.h5", 'r')
# hdf5.read(mesh, 'm', False)


# fFile = HDF5File(MPI.COMM_WORLD, "tmp/mesh.h5", "w")
# fFile.write(mesh, "m")
# fFile.close()


V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

xdmf = XDMFFile(MPI.COMM_WORLD, "tmp/u.xdmf")


u = Function(V, name="u")
# xdmf.write(u, 0)
xdmf.write_checkpoint(u, "u", 0)

# print(u.vector()[:])

for i in range(10):
	solve(a == L, u, bc)
# xdmf.write(u, 1)
# xdmf.write_checkpoint(u, "u", 1)

end = time.time()
print(f"Wall time elapsed: {end-start}")
