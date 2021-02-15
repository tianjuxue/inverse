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
        self.set_boundaries()
        self.staggered_tol = 1e-5
        self.staggered_maxiter = 1000 
        self.display_intermediate_results = False
        self.delta_u_recorded = []
        self.sigma_recorded = []


    def preparation(self):
        data_path_pvd = 'data/pvd/{}'.format(self.case_name)
        print("\nDelete data folder {}".format(data_path_pvd))
        shutil.rmtree(data_path_pvd, ignore_errors=True)


    def set_boundaries(self):
        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.ds = fe.Measure("ds")(subdomain_data=self.boundaries)   


    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 
        self.V = fe.FunctionSpace(self.mesh, 'DG', 0)
       
        # self.E = 300
        self.E = da.interpolate(da.Constant(210.), self.V)
        self.E.rename("E", "E")
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
        solver_u = da.NonlinearVariationalSolver(p_u)
        solver_d  = da.NonlinearVariationalSolver(p_d)

        vtkfile_u = fe.File('data/pvd/{}/u.pvd'.format(self.case_name))
        vtkfile_d = fe.File('data/pvd/{}/d.pvd'.format(self.case_name))
        vtkfile_u_opt = fe.File('data/pvd/{}/u_opt.pvd'.format(self.case_name))
        vtkfile_d_opt = fe.File('data/pvd/{}/d_opt.pvd'.format(self.case_name))
        vtkfile_E = fe.File('data/pvd/{}/E.pvd'.format(self.case_name))

        J = 0

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):
            print('\n')
            print('=================================================================================')
            print('>> Step {}, disp boundary condition = {} [mm]'.format(i, disp))
            print('=================================================================================')

            self.presLoad.t = disp

            newton_prm = solver_u.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100  
            # newton_prm['absolute_tolerance'] = 1e-8
            newton_prm['relaxation_parameter'] = rp
 
            solver_d.solve()

            solver_u.solve()

            J += da.assemble(0.5 * self.d_new**2 * fe.dx)

            vtkfile_u << self.x_new
            vtkfile_d << self.d_new

            force_upper = float(da.assemble(self.sigma[1, 1]*self.ds(1)))
            print("Force upper {}".format(force_upper))
            self.delta_u_recorded.append(disp)
            self.sigma_recorded.append(force_upper)

            if self.display_intermediate_results and i % 20 == 0:
                self.show_force_displacement()

        alpha = da.Constant(0.)
        J += da.assemble(alpha / 2 * self.E ** 2 * fe.dx)

        self.object_values = []

        def eval_cb(j, m):
            print("Objective is {}".format(j))
            self.object_values.append(j)
 
        control = da.Control(self.E)
        reduced_functional = da.ReducedFunctional(J, control, eval_cb_post=eval_cb)

        def save_parameter(x):
            self.E.vector()[:] = x
            vtkfile_E << self.E
            print("callback, assign values to E")

        E_opt = da.minimize(reduced_functional, method="L-BFGS-B", tol=1e-20, bounds = (100, 300), callback=save_parameter,
            options={"disp": True, "maxiter": 20})

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):
            self.presLoad.t = disp
            solver_d.solve()
            solver_u.solve()
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
 
        domain = mshr.Polygon([fe.Point(self.length / 2, self.height / 2), 
                               fe.Point(0, self.height / 2 - 1e-10), 
                               fe.Point(0, 0),
                               fe.Point(self.length, 0),
                               fe.Point(self.length, self.height/2),
                               fe.Point(self.length, self.height),
                               fe.Point(0, self.height),
                               fe.Point(0, self.height/2 + 1e-10)])

        self.mesh = mshr.generate_mesh(domain, 50)
        
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

        self.lower = Lower()
        self.upper = Upper()
        self.corner = Corner()
        self.left = Left()
        self.right = Right()


    def set_bcs_staggered(self):
        self.upper.mark(self.boundaries, 1)
        self.presLoad = da.Expression(("t", 0), t=0.0, degree=1)
        BC_u_lower = da.DirichletBC(self.U, da.Constant((0., 0.)), self.lower)
        BC_u_upper = da.DirichletBC(self.U, self.presLoad, self.upper) 
        BC_u_left = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.left)
        BC_u_right = da.DirichletBC(self.U.sub(1), da.Constant(0),  self.right)
        self.BC_u = [BC_u_lower, BC_u_upper, BC_u_left, BC_u_right] 
        self.BC_d = []


    def build_weak_form_staggered(self):
        self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)
        self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
        self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
 
        self.sigma = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi)
        self.sigma_plus = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi_plus)
        self.sigma_minus = cauchy_stress_minus(strain(fe.grad(self.x_new)), self.psi_minus)

        self.H_old = self.psi_plus(strain(fe.grad(self.x_new)))

        self.G_u = (g_d(self.d_new) * fe.inner(self.sigma_plus, strain(fe.grad(self.eta))) \
            + fe.inner(self.sigma_minus, strain(fe.grad(self.eta)))) * fe.dx

        self.G_d = (self.H_old * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx
 

def test(args):
    pde = TestCase()
    pde.staggered_solve()



if __name__ == '__main__':
    args = arguments.args
    test(args)
