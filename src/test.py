import fenics as fe
import sys
import time
import numpy as np
import mshr
import matplotlib.pyplot as plt
import glob
import os
import shutil
from functools import partial
import dolfin_adjoint as da
from pyadjoint.overloaded_type import create_overloaded_object
import moola
import scipy.optimize as opt
from . import arguments
from .constitutive import *


fe.parameters["form_compiler"]["quadrature_degree"] = 4


class PDE(object):
    def __init__(self):
        self.preparation()
        self.build_mesh()
        self.staggered_tol = 1e-5
        self.staggered_maxiter = 1000 
        self.display_intermediate_results = False
        self.delta_u_recorded = []
        self.sigma_recorded = []


    def preparation(self):
        data_path_pvd = 'data/pvd/{}'.format(self.case_name)
        print("\nDelete data folder {}".format(data_path_pvd))
        shutil.rmtree(data_path_pvd, ignore_errors=True)

    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 
        self.V = fe.FunctionSpace(self.mesh, 'DG', 0)
       
        # self.E = 300
        # self.E = da.interpolate(da.Constant(210.), self.V)
        # self.E.rename("E", "E")

        self.E = 210

        self.nu = 0.3
        self.mu = self.E / (2 * (1 + self.nu))
        self.lamda = (2. * self.mu * self.nu) / (1. - 2. * self.nu)

        self.G_c = 2.7*1e-3
        self.l0 = 0.02

        self.eta = fe.TestFunction(self.U)
        self.zeta = fe.TestFunction(self.W)

        del_x = fe.TrialFunction(self.U)
        del_d = fe.TrialFunction(self.W)

        self.x_new = da.Function(self.U, name="u")
        self.d_new = da.Function(self.W, name="d")
 
        self.build_weak_form_staggered()
        J_u = fe.derivative(self.G_u, self.x_new, del_x)
        J_d = fe.derivative(self.G_d, self.d_new, del_d) 

        self.set_bcs_staggered()
        p_u = da.NonlinearVariationalProblem(self.G_u, self.x_new, self.BC_u, J_u)
        p_d  = da.NonlinearVariationalProblem(self.G_d,  self.d_new, self.BC_d, J_d)
        self.solver_u = da.NonlinearVariationalSolver(p_u)
        self.solver_d  = da.NonlinearVariationalSolver(p_d)

        vtkfile_u = fe.File(f'data/pvd/{self.case_name}/u.pvd')
        vtkfile_d = fe.File(f'data/pvd/{self.case_name}/d.pvd')


        self.J = 0

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):
            print('\n')
            print('=================================================================================')
            print('>> Step {}, disp boundary condition = {} [mm]'.format(i, disp))
            print('=================================================================================')

            self.presLoad.t = disp

            newton_prm = self.solver_u.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100  
            # newton_prm['absolute_tolerance'] = 1e-8
            newton_prm['relaxation_parameter'] = rp
 
            self.solver_d.solve()

            self.solver_u.solve()

            self.J += da.assemble(0.5 * self.d_new**2 * fe.dx)

            vtkfile_u << self.x_new
            vtkfile_d << self.d_new

            force_upper = float(da.assemble(self.sigma[1, 1]*self.ds(1)))
            print("Force upper {}".format(force_upper))
            self.delta_u_recorded.append(disp)
            self.sigma_recorded.append(force_upper)

            if self.display_intermediate_results and i % 20 == 0:
                self.show_force_displacement()
 

    def adjoint_optimization(self):

        # alpha = da.Constant(0.)
        # self.J += da.assemble(alpha / 2 * self.E ** 2 * fe.dx)

        alpha = 1
        Vol = da.assemble(self.one * fe.dx(domain=self.mesh))
        self.J += alpha * ((self.length * self.height - Vol) - self.Vol0)**2
     
        (x, y) = fe.SpatialCoordinate(self.mesh)
        Bc1 = (self.length**2 * self.height / 2 - da.assemble(x * fe.dx(domain=self.mesh))) / (self.length * self.height - Vol)
        Bc2 = (self.length * self.height**2 / 2 - da.assemble(y * fe.dx(domain=self.mesh))) / (self.length * self.height - Vol)
        beta = 1
        self.J += beta * ((Bc1 - self.xcenter)**2 + (Bc2 - self.ycenter)**2)

        self.object_values = []

        def eval_cb(j, m):
            print("Objective is {}".format(j))
            self.object_values.append(j)
 
        control = da.Control(self.h)
        reduced_functional = da.ReducedFunctional(self.J, control, eval_cb_post=eval_cb)

        vtkfile_u_opt = fe.File(f'data/pvd/{self.case_name}/u_opt.pvd')
        vtkfile_d_opt = fe.File(f'data/pvd/{self.case_name}/d_opt.pvd')
        vtkfile_E = fe.File(f'data/pvd/{self.case_name}/E.pvd')


        def save_parameter(x):
            pass
            # self.E.vector()[:] = x
            # vtkfile_E << self.E
            # print("callback, assign values to E")

        da.minimize(reduced_functional, method="L-BFGS-B", tol=1e-20,  callback=save_parameter,
            options={"disp": True, "maxiter": 50})

        self.d_new.vector()[:] = 0
        self.x_new.vector()[:] = 0

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):
            self.presLoad.t = disp
            self.solver_d.solve()
            self.solver_u.solve()
            vtkfile_u_opt << self.x_new
            vtkfile_d_opt << self.d_new

        print(self.object_values)

        self.show_optimization_progress()

        # self.show_force_displacement()
        # plt.ioff()
        # plt.show()


    def show_optimization_progress(self):
        fig = plt.figure()
        plt.plot(self.object_values[1:], linestyle='--', marker='o')
        plt.tick_params(labelsize=14)
        plt.xlabel("$N$ (L-BFGS-B steps)", fontsize=14)
        plt.ylabel("$J$ (Objective)", fontsize=14)
        plt.show()


    def show_force_displacement(self):
        delta_u_recorded = np.absolute(np.array(self.delta_u_recorded))
        sigma_recorded = np.absolute(np.array(self.sigma_recorded))
        fig = plt.figure(0)
        plt.ion()
        plt.plot(delta_u_recorded, sigma_recorded, linestyle='--', marker='o', color='red', label='full')
 
        # plt.legend(fontsize=14)
        plt.tick_params(labelsize=14)
        plt.xlabel("Vertical displacement of top side", fontsize=14)
        plt.ylabel("Force on top side", fontsize=14)
        plt.grid(True)
        # fig.savefig('data/pdf/{}/force_load.pdf'.format(self.case_name), bbox_inches='tight')
        plt.show()
        plt.pause(0.001)



class TestCase(PDE):
    def __init__(self):
        self.case_name = "brittle"
        super(TestCase, self).__init__()

        # self.displacements = 1e-1*np.concatenate((np.linspace(0, 0.1, 21), np.linspace(0.1, 0.12, 101)))
        self.displacements = 1e-1*np.linspace(0, 0.15, 100)
        self.relaxation_parameters =  np.linspace(1, 1, len(self.displacements))


    def build_mesh(self): 
        self.length = 1.
        self.height = 1.
 
        # domain = mshr.Polygon([fe.Point(self.length / 2, self.height / 2), 
        #                        fe.Point(0, self.height / 2 - 1e-10), 
        #                        fe.Point(0, 0),
        #                        fe.Point(self.length, 0),
        #                        fe.Point(self.length, self.height/2),
        #                        fe.Point(self.length, self.height),
        #                        fe.Point(0, self.height),
        #                        fe.Point(0, self.height/2 + 1e-10)])

        # self.mesh = mshr.generate_mesh(domain, 50)
        

        radius = 0.2
        self.xcenter = 0.5
        self.ycenter = 0.5
        plate = mshr.Rectangle(fe.Point(0., 0.), fe.Point(self.length, self.height))
        circle = mshr.Circle(fe.Point(self.xcenter, self.ycenter), radius)
        material_domain = plate - circle
        self.mesh = mshr.generate_mesh(material_domain, 50)

        # Add dolfin-adjoint dependency
        self.mesh  = create_overloaded_object(self.mesh)


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

        class Corner(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return fe.near(x[0], 0) and fe.near(x[1], 0)

        class Hole(fe.SubDomain):
            def inside(self, x, on_boundary):                    
                return on_boundary and x[0] > 0 and x[0] < length and x[1] > 0 and x[1] < height

        self.lower = Lower()
        self.upper = Upper()
        self.corner = Corner()
        self.left = Left()
        self.right = Right()
        self.hole = Hole()


        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.upper.mark(self.boundaries, 1)
        self.hole.mark(self.boundaries, 2)
        self.ds = fe.Measure("ds")(subdomain_data=self.boundaries)

        self.one = da.Constant(1.)
        self.Vol0 = self.length * self.height - fe.assemble(self.one * fe.dx(domain=self.mesh))

        b_mesh = da.BoundaryMesh(self.mesh, "exterior")
        S_b = fe.VectorFunctionSpace(b_mesh, "CG", 1)
        self.h = da.Function(S_b, name="h")

        # self.h.vector()[:] = 1e1

        zero = da.Constant([0.] * self.mesh.geometric_dimension())

        S = fe.VectorFunctionSpace(self.mesh, "CG", 1)
        s = da.Function(S, name="Mesh perturbation field")
        self.h_V = da.transfer_from_boundary(self.h, self.mesh)
        self.h_V.rename("Volume extension of h", "")

        def mesh_deformation():
            # Compute variable :math:`\mu`
            V = fe.FunctionSpace(self.mesh, "CG", 1)
            u, v = fe.TrialFunction(V), fe.TestFunction(V)

            a = -fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
            l = da.Constant(0.) * v * fe.dx

            mu_min = da.Constant(1., name="mu_min")
            mu_max = da.Constant(500, name="mu_max")
            bcs = []
            for side in [self.lower, self.upper, self.left, self.right]:
                bcs.append(da.DirichletBC(V, mu_min, side))
            bcs.append(da.DirichletBC(V, mu_max, self.hole))

            mu = da.Function(V, name="mesh deformation mu")
            da.solve(a == l, mu, bcs=bcs)

            # Compute the mesh deformation
            S = fe.VectorFunctionSpace(self.mesh, "CG", 1)
            u, v = fe.TrialFunction(S), fe.TestFunction(S)

            def epsilon(u):
                return fe.sym(fe.grad(u))

            def sigma(u, mu=500, lmb=0):
                return 2 * mu * epsilon(u) + lmb * fe.tr(epsilon(u)) * fe.Identity(2)

            a = fe.inner(sigma(u, mu=mu), fe.grad(v)) * fe.dx
            L = fe.inner(self.h_V, v) * self.ds(2)

            bcs = []
            for side in [self.lower, self.upper, self.left, self.right]:
                bcs.append(da.DirichletBC(S, zero, side))

            s = da.Function(S, name="mesh deformation")
            da.solve(a == L, s, bcs=bcs)

            return s

        s = mesh_deformation()
        fe.ALE.move(self.mesh, s)

        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/mesh.pvd')
        vtkfile_mesh << self.mesh


    def set_bcs_staggered(self):
        
        # self.presLoad = da.Expression(("t", 0), t=0.0, degree=1)
        # BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        # BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        # BC_u_left = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.left)
        # BC_u_right = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.right)
        # self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        # self.BC_d = []

        self.presLoad = da.Expression("t", t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0, 0)),  self.lower)
        BC_u_upper = da.DirichletBC(self.U.sub(1), self.presLoad,  self.upper)
        self.BC_u = [BC_u_lower, BC_u_upper]
        self.BC_d = []


    def build_weak_form_staggered(self):
        self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)
        self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
 
        self.sigma = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi)
        self.sigma_plus = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi_plus)
        self.sigma_minus = cauchy_stress_minus(strain(fe.grad(self.x_new)), self.psi_minus)

        self.H_old = self.psi_plus(strain(fe.grad(self.x_new)))
        self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        self.G_u = (g_d(self.d_new) * fe.inner(self.sigma_plus, strain(fe.grad(self.eta))) \
            + fe.inner(self.sigma_minus, strain(fe.grad(self.eta)))) * fe.dx


def test(args):
    pde = TestCase()
    pde.staggered_solve()
    pde.adjoint_optimization()


if __name__ == '__main__':
    args = arguments.args
    test(args)
