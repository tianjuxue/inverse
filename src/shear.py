import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
import shutil
import time
from mpi4py import MPI
from functools import partial
import dolfin_adjoint as da
from pyadjoint.overloaded_type import create_overloaded_object
from . import arguments
from .pdeco import PDECO

rank = MPI.COMM_WORLD.Get_rank()


class Shear(PDECO):
    def __init__(self, problem):
        self.case_name = "shear"
        self.displacements = 1e-1*np.linspace(0, 1., 100)
        # self.displacements = -1e-1*np.linspace(0, 1., 15)
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))
        self.E = 210
        self.nu = 0.3
        self.mu = self.E / (2 * (1 + self.nu))
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)
        self.G_c = 2.7*1e-3
        self.l0 = 0.02
        super(Shear, self).__init__(problem)


    def build_mesh(self, mesh_file=None): 
        num_units_x = 2
        num_units_y = 2
        self.length_unit = 1.
        self.length = num_units_x * self.length_unit
        self.height = num_units_y * self.length_unit

        radius = 0.3
        self.xcenter = 0.5
        self.ycenter = 0.5

        domain = mshr.Polygon([fe.Point(0., 0.),
                               fe.Point(self.length, 0.),
                               fe.Point(self.length, self.height),
                               fe.Point(0., self.height)])
 
        for i in range(num_units_x):
            for j in range(num_units_y):
                domain -= mshr.Circle(fe.Point(i * self.length_unit + self.xcenter, j * self.length_unit + self.ycenter), radius)

        self.mesh = mshr.generate_mesh(domain, 50)
 
        # if self.problem == 'forward' or self.problem == 'debug':
        #     self.mesh = fe.Mesh()
        #     xdmf_file_mesh = fe.XDMFFile(MPI.COMM_WORLD, mesh_file)
        #     xdmf_file_mesh.read(self.mesh)

        # Add dolfin-adjoint dependency
        self.mesh  = create_overloaded_object(self.mesh)

        # Defensive copy
        self.mesh_initial = fe.Mesh(self.mesh)

        length = self.length
        height = self.height
        length_unit = self.length_unit

        class Lower(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0)

        class Upper(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], height)

        class Left(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0)

        class Right(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and fe.near(x[0], length)

        class Hole(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and x[0] > 0 and x[0] < length and x[1] > 0 and x[1] < height

        class Periodic(fe.SubDomain):
            '''Use lower left hole as the reference
            '''
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > 0 and x[0] < length_unit and x[1] > 0 and x[1] < length_unit

            def map(self, x, y):
                y[0] = x[0] -  x[0] // length_unit * length_unit
                y[1] = x[1] -  x[1] // length_unit * length_unit


        class LowerLeftSquare(fe.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= 0 and x[1] >= 0 and  x[0] <= length_unit and x[1] <= length_unit


        self.lower = Lower()
        self.upper = Upper()
        self.left = Left()
        self.right = Right()
        self.hole = Hole()
        self.periodic = Periodic()

        self.periodic = None

        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.upper.mark(self.boundaries, 1)
        self.hole.mark(self.boundaries, 2)
        self.ds = fe.Measure("ds")(domain=self.mesh, subdomain_data=self.boundaries)

        self.lower_left_square = LowerLeftSquare()

        self.materials = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        self.materials.set_all(0)
        self.lower_left_square.mark(self.materials, 1)
        self.dx = fe.Measure("dx")(domain=self.mesh, subdomain_data=self.materials)

        print(f"hmax={self.mesh.hmax()}, hmin={self.mesh.hmin()}")


    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)
        self.presLoad = da.Expression(("t", 0), t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        BC_u_left = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.left)
        BC_u_right = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.right)
        self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        self.BC_d = []


    def update_objective_in_the_loop(self):
        self.J -= da.assemble(self.sigma[0, 1]*self.ds(1))
        return float(da.assemble(self.sigma[0, 1]*self.ds(1)))


    def update_objective_reg(self):
        alpha = 1e2
        Vol0 = fe.assemble(da.Constant(1.) * self.dx(1))
        Vol = da.assemble(da.Constant(1.) * self.dx(1))
        print(f"Vol0={Vol0}")
        self.J += alpha * (Vol - Vol0)**2
 
        (x, y) = fe.SpatialCoordinate(self.mesh)
        Bc1 = (self.length_unit**2 * self.length_unit / 2 - da.assemble(x * self.dx(1))) / (self.length_unit * self.length_unit - Vol)
        Bc2 = (self.length_unit * self.length_unit**2 / 2 - da.assemble(y * self.dx(1))) / (self.length_unit * self.length_unit - Vol)
        beta = 1e2
        self.J += beta * ((Bc1 - self.xcenter)**2 + (Bc2 - self.ycenter)**2)


def main(args):
    begin_time = time.time()

    # for i in range(1):
    #     pde = Shear('debug')
    #     pde.run(i)

    pde = Shear('inverse')
    pde.run()

    # for i in range(3):
    #     pde = Shear('forward')
    #     pde.run(i)

    # for i in range(3):
    #     pde = Shear('post-processing')
    #     pde.run(i)

    end_time = time.time()
    if rank == 0:
        print(f"Wall time elapsed: {end_time-begin_time}")


if __name__ == '__main__':
    args = arguments.args
    main(args)
    plt.show()

 
