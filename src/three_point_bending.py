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


class ThreePointBending(PDECO):
    def __init__(self, problem):
        super(ThreePointBending, self).__init__(problem)
        self.case_name = "three_point_bending"
        self.displacements = -1e-1*np.linspace(0, 2., 100)
        # self.displacements = -1e-1*np.linspace(0, 1., 15)
        self.relaxation_parameters = np.linspace(1, 1, len(self.displacements))
        self.mu = 8.       
        self.lamda = 12.
        self.G_c = 5.*1e-4
        self.l0 = 0.02


    def build_mesh(self, mesh_file=None): 
        num_units_x = 9
        num_units_y = 2
        length_unit = 1.
        self.length = num_units_x * length_unit
        self.height = num_units_y * length_unit

        radius = 0.3
        self.xcenter = 0.5
        self.ycenter = 0.5

        self.notch_length = 0.2
        self.notch_height = 0.4

        domain = mshr.Polygon([fe.Point(0., 0.),
                               fe.Point(self.length/2. - self.notch_length/2., 0.),
                               fe.Point(self.length/2., self.notch_height),
                               fe.Point(self.length/2. + self.notch_length/2., 0.),
                               fe.Point(self.length, 0.),
                               fe.Point(self.length, self.height),
                               fe.Point(self.length/2., self.height),
                               fe.Point(0., self.height)])
 
        for i in range(num_units_x):
            for j in range(num_units_y):
                if i != num_units_x//2 or j != 0:
                    domain -= mshr.Circle(fe.Point(i * length_unit + self.xcenter, j * length_unit + self.ycenter), radius)

        self.mesh = mshr.generate_mesh(domain, 100)
 
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

        class LeftCorner(fe.SubDomain):
            def inside(self, x, on_boundary):
                return fe.near(x[0], 0.) and fe.near(x[1], 0.)

        class RightCorner(fe.SubDomain):
            def inside(self, x, on_boundary):
                return fe.near(x[0], length) and fe.near(x[1], 0.)

        class MiddlePoint(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return fe.near(x[0], length/2.) and fe.near(x[1], height)

        class Hole(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and x[0] > 0 and x[0] < length and x[1] > 0 and x[1] < height


        self.lower = Lower()
        self.upper = Upper()
        self.left = Left()
        self.right = Right()
        self.hole = Hole()
        self.left_corner = LeftCorner()
        self.right_corner = RightCorner()
        self.middle_point = MiddlePoint()


        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.upper.mark(self.boundaries, 1)
        self.hole.mark(self.boundaries, 2)
        self.ds = fe.Measure("ds")(subdomain_data=self.boundaries)

        print(f"hmax={self.mesh.hmax()}, hmin={self.mesh.hmin()}")


    def set_bcs_staggered(self):
        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_left = da.DirichletBC(self.U, fe.Constant((0., 0.)), self.left_corner, method='pointwise')
        BC_u_right = da.DirichletBC(self.U.sub(1), fe.Constant(0.), self.right_corner, method='pointwise')
        BC_u_middle = da.DirichletBC(self.U.sub(1), self.presLoad, self.middle_point, method='pointwise')
        self.BC_u = [BC_u_left, BC_u_right, BC_u_middle]
        self.BC_d = []


def main(args):
    begin_time = time.time()

    for i in range(1):
        pde = ThreePointBending('debug')
        pde.run(i)

    # pde = ThreePointBending('inverse')
    # pde.run()

    # for i in range(3):
    #     pde = ThreePointBending('forward')
    #     pde.run(i)

    # for i in range(3):
    #     pde = ThreePointBending('post-processing')
    #     pde.run(i)

    end_time = time.time()
    if rank == 0:
        print(f"Wall time elapsed: {end_time-begin_time}")


if __name__ == '__main__':
    args = arguments.args
    main(args)



# # Sub domain for Periodic boundary condition
# class PeriodicBoundary(SubDomain):

#     # Left boundary is "target domain" G
#     def inside(self, x, on_boundary):
#         return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

#     # Map right boundary (H) to left boundary (G)
#     def map(self, x, y):
#         y[0] = x[0] - 1.0
#         y[1] = x[1]

# # Create mesh and finite element
# mesh = UnitSquareMesh(32, 32)
# V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

