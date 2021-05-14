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


class SinglePore(PDECO):
    def __init__(self, problem):
        self.case_name = "single_pore"
        self.displacements = 1e-1*np.linspace(0, 0.1, 150)
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))
        self.E = 210
        self.nu = 0.3
        self.mu = self.E / (2 * (1 + self.nu))
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)
        self.G_c = 2.7*1e-3
        self.l0 = 0.02
        super(SinglePore, self).__init__(problem)


    def build_mesh(self, mesh_file=None): 
        self.length = 1.
        self.height = 1.
 
        radius = 0.3
        self.xcenter = 0.5
        self.ycenter = 0.5
        plate = mshr.Rectangle(fe.Point(0., 0.), fe.Point(self.length, self.height))
        circle = mshr.Circle(fe.Point(self.xcenter, self.ycenter), radius)
        material_domain = plate - circle
        self.mesh = mshr.generate_mesh(material_domain, 50)

        # if self.problem == 'forward' or self.problem == 'debug':
        #     self.mesh = fe.Mesh()
        #     xdmf_file_mesh = fe.XDMFFile(MPI.COMM_WORLD, mesh_file)
        #     xdmf_file_mesh.read(self.mesh)
        #     # self.mesh = fe.Mesh(mesh_file)

        # Add dolfin-adjoint dependency
        self.mesh  = create_overloaded_object(self.mesh)

        # Defensive copy
        self.mesh_initial = fe.Mesh(self.mesh)

        length = self.length
        height = self.height

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

        self.lower = Lower()
        self.upper = Upper()
        self.left = Left()
        self.right = Right()
        self.hole = Hole()

        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.upper.mark(self.boundaries, 1)
        self.hole.mark(self.boundaries, 2)
        self.ds = fe.Measure("ds")(domain=self.mesh, subdomain_data=self.boundaries)


    def set_bcs_staggered(self):
        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0, 0)),  self.lower)
        BC_u_upper = da.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        self.BC_u = [BC_u_lower, BC_u_upper]
        self.BC_d = []


    def update_objective_in_the_loop(self):
        # self.J += da.assemble(0.5 * self.d_new**2 * fe.dx)
        self.J -= da.assemble(self.sigma[1, 1] * self.ds(1))
        return float(da.assemble(self.sigma[1, 1] * self.ds(1)))


    def update_objective_reg(self):
        alpha = 1e2
        Vol0 = fe.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))
        Vol = da.assemble(da.Constant(1.) * fe.dx(domain=self.mesh))
        self.J += alpha * (Vol - Vol0)**2
 
        (x, y) = fe.SpatialCoordinate(self.mesh)
        Bc1 = (self.length**2 * self.height / 2 - da.assemble(x * fe.dx(domain=self.mesh))) / (self.length * self.height - Vol)
        Bc2 = (self.length * self.height**2 / 2 - da.assemble(y * fe.dx(domain=self.mesh))) / (self.length * self.height - Vol)
        beta = 1e2
        self.J += beta * ((Bc1 - self.xcenter)**2 + (Bc2 - self.ycenter)**2)


def main(args):
    begin_time = time.time()

    for i in range(1):
        pde = SinglePore('debug')
        pde.run(i)

    # pde = SinglePore('inverse')
    # pde.run()

    # for i in range(3):
    #     pde = SinglePore('forward')
    #     pde.run(i)

    # for i in range(3):
    #     pde = SinglePore('post-processing')
    #     pde.run(i)

    end_time = time.time()
    if rank == 0:
        print(f"Wall time elapsed: {end_time-begin_time}")


if __name__ == '__main__':
    args = arguments.args
    main(args)
    # plot_comparison('brittle')