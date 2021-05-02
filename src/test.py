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
# import moola
# import scipy.optimize as opt
from . import arguments
from .constitutive import *


fe.parameters["form_compiler"]["quadrature_degree"] = 4


class PDE(object):
    def __init__(self, problem='inverse'):
        self.case_name = "brittle"
        self.displacements = 1e-1*np.linspace(0, 0.15, 100)
        self.relaxation_parameters =  np.linspace(1, 1, len(self.displacements))
        self.staggered_tol = 1e-5
        self.staggered_maxiter = 1000 
        self.display_intermediate_results = False
        self.delta_u_recorded = []
        self.sigma_recorded = []     
        self.problem = problem 
        self.preparation()
        

    def run(self, opt_step=0):
        if self.problem == 'inverse':
            self.build_mesh()
            self.staggered_solve()
            self.adjoint_optimization()
        elif self.problem == 'forward':
            self.opt_step = opt_step
            self.build_mesh(f'data/xml/{self.case_name}/inverse/mesh_{self.opt_step}.xml')
            self.staggered_solve()
        elif self.problem == 'post-processing':
            self.opt_step = opt_step
            self.plot_force_displacement()
            plt.ioff()
            plt.show()
        else:
            raise ValueError('Unknown problem mode!')


    def preparation(self):
        if self.problem == 'inverse':
            print(f"\nDelete inverse problem related data...")
            shutil.rmtree(f'data/pvd/{self.case_name}/{self.problem}', ignore_errors=True)
            shutil.rmtree(f'data/xml/{self.case_name}/{self.problem}', ignore_errors=True)


    def build_mesh(self, mesh_file=None): 
        self.length = 1.
        self.height = 1.
 
        radius = 0.2
        self.xcenter = 0.5
        self.ycenter = 0.5
        plate = mshr.Rectangle(fe.Point(0., 0.), fe.Point(self.length, self.height))
        circle = mshr.Circle(fe.Point(self.xcenter, self.ycenter), radius)
        material_domain = plate - circle
        self.mesh = mshr.generate_mesh(material_domain, 50)

        if self.problem == 'forward':
            self.mesh = fe.Mesh(mesh_file)

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
        self.S_b = fe.VectorFunctionSpace(b_mesh, "CG", 1)
        self.h = da.Function(self.S_b, name="h")
        # self.h.vector()[:] = 2.
        s = self.mesh_deformation(self.h)
        fe.ALE.move(self.mesh, s)


    def mesh_deformation(self, h):

        zero = da.Constant([0.] * self.mesh.geometric_dimension())

        S = fe.VectorFunctionSpace(self.mesh, "CG", 1)
        s = da.Function(S, name="Mesh perturbation field")
        h_V = da.transfer_from_boundary(h, self.mesh)
        h_V.rename("Volume extension of h", "")

        # Compute variable :math:`\mu`
        V = fe.FunctionSpace(self.mesh, "CG", 1)
        u, v = fe.TrialFunction(V), fe.TestFunction(V)

        a = -fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        l = da.Constant(0.) * v * fe.dx

        mu_min = da.Constant(1., name="mu_min")
        mu_max = da.Constant(2., name="mu_max")
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
        L = fe.inner(h_V, v) * self.ds(2)

        bcs = []
        for side in [self.lower, self.upper, self.left, self.right]:
            bcs.append(da.DirichletBC(S, zero, side))

        s = da.Function(S, name="mesh deformation")
        da.solve(a == L, s, bcs=bcs)

        return s


    def set_bcs_staggered(self):
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

        self.G_d = (self.history * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx

        self.G_u = (g_d(self.d_new) * fe.inner(self.sigma_plus, strain(fe.grad(self.eta))) \
            + fe.inner(self.sigma_minus, strain(fe.grad(self.eta)))) * fe.dx


    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 
        self.V = fe.FunctionSpace(self.mesh, 'DG', 0)
       
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
        self.history = da.Function(self.V)
 
        self.build_weak_form_staggered()
        J_u = fe.derivative(self.G_u, self.x_new, del_x)
        J_d = fe.derivative(self.G_d, self.d_new, del_d) 

        self.set_bcs_staggered()
        p_u = da.NonlinearVariationalProblem(self.G_u, self.x_new, self.BC_u, J_u)
        p_d  = da.NonlinearVariationalProblem(self.G_d,  self.d_new, self.BC_d, J_d)
        self.solver_u = da.NonlinearVariationalSolver(p_u)
        self.solver_d  = da.NonlinearVariationalSolver(p_d)

        if self.problem == 'forward':
            vtkfile_u = fe.File(f'data/pvd/{self.case_name}/{self.problem}/step_{self.opt_step}/u.pvd')
            vtkfile_d = fe.File(f'data/pvd/{self.case_name}/{self.problem}/step_{self.opt_step}/d.pvd')

        self.J = 0

        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):
            print('\n')
            print('=================================================================================')
            print(f'>> Step {i}, disp boundary condition = {disp} [mm]')
            print('=================================================================================')

            self.presLoad.t = disp

            newton_prm = self.solver_u.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100  
            # newton_prm['absolute_tolerance'] = 1e-8
            newton_prm['relaxation_parameter'] = rp
 
            psi_plus = self.psi_plus(strain(fe.grad(self.x_new)))
            self.history.assign(da.project(fe.conditional(fe.gt(psi_plus, self.history), psi_plus, self.history), self.V))

            self.solver_d.solve()

            self.solver_u.solve()

            # self.J += da.assemble(0.5 * self.d_new**2 * fe.dx)
            self.J -= da.assemble(self.sigma[1, 1]*self.ds(1))

            if self.problem == 'forward':
                vtkfile_u << self.x_new
                vtkfile_d << self.d_new

            force_upper = float(da.assemble(self.sigma[1, 1]*self.ds(1)))
            print(f"Force upper {force_upper}")
            self.delta_u_recorded.append(disp)
            self.sigma_recorded.append(force_upper)

            # if self.display_intermediate_results and i % 20 == 0:
            #     self.plot_force_displacement()

        alpha = 1e1
        Vol = da.assemble(self.one * fe.dx(domain=self.mesh))
        self.J += alpha * ((self.length * self.height - Vol) - self.Vol0)**2
     
        (x, y) = fe.SpatialCoordinate(self.mesh)
        Bc1 = (self.length**2 * self.height / 2 - da.assemble(x * fe.dx(domain=self.mesh))) / (self.length * self.height - Vol)
        Bc2 = (self.length * self.height**2 / 2 - da.assemble(y * fe.dx(domain=self.mesh))) / (self.length * self.height - Vol)
        beta = 1e1
        self.J += beta * ((Bc1 - self.xcenter)**2 + (Bc2 - self.ycenter)**2)

        if self.problem == 'forward':
            np.save(f'data/numpy/{self.case_name}/step_{self.opt_step}_u.npy', np.array(self.delta_u_recorded))
            np.save(f'data/numpy/{self.case_name}/step_{self.opt_step}_f.npy', np.array(self.sigma_recorded))

        return float(self.J)


    def adjoint_optimization(self):
        self.object_values = []

        def eval_callback(j, m):
            '''
            Hint: m has the same type as self.h
            '''
            # print(f"Parameter is {np.array(m.vector())}")
            print(f"Objective is {j}")
            self.object_values.append(j)

        control = da.Control(self.h)
        Jhat = reduced_functional = da.ReducedFunctional(self.J, control, eval_cb_post=eval_callback)

        # vtkfile_u_opt = fe.File(f'data/pvd/{self.case_name}/u_opt.pvd')
        # vtkfile_d_opt = fe.File(f'data/pvd/{self.case_name}/d_opt.pvd')

        def iteration_callback(x):
            ''' Callback function for optimization

            Parameter
            ---------
            x is a 1D numpy array
            '''
            iteration_callback.count += 1
            print("BFGS call back...")
            mesh_copy = fe.Mesh(self.mesh_initial)
            h_callback = fe.Function(self.S_b)
            h_callback.vector()[:] = x
            s = self.mesh_deformation(h_callback)
            fe.ALE.move(mesh_copy, s)
            # self.plot_mesh(mesh_copy)
            vtkfile_mesh << mesh_copy
            xml_mesh = fe.File(f'data/xml/{self.case_name}/{self.problem}/mesh_{iteration_callback.count}.xml')
            xml_mesh << mesh_copy
            
            print(f"max h {np.max(np.absolute(x))}")


        iteration_callback.count = 0
        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/{self.problem}/mesh.pvd')
        vtkfile_mesh << self.mesh
        xml_mesh = fe.File(f'data/xml/{self.case_name}/{self.problem}/mesh_{iteration_callback.count}.xml')
        xml_mesh << self.mesh

        h_opt = da.minimize(reduced_functional, method="L-BFGS-B", tol=1e-20, bounds=(-2., 2.), callback=iteration_callback,
            options={"disp": True, "maxiter": 10})

        # np.save(f'data/numpy/{self.case_name}/object_values.npy', np.array(self, object_values))

        # s = self.mesh_deformation(h_opt)
        # fe.ALE.move(self.mesh, s)
        # self.d_new.vector()[:] = 0
        # self.x_new.vector()[:] = 0
        # for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):
        #     self.presLoad.t = disp
        #     self.solver_d.solve()
        #     self.solver_u.solve()
        #     vtkfile_u_opt << self.x_new
        #     vtkfile_d_opt << self.d_new

        # print(self.object_values)

        # self.show_optimization_progress()


    def plot_mesh(self, mesh_copy):
        fig = plt.figure()
        plt.ion()
        initial, _ = fe.plot(self.mesh, color="b", linewidth=0.25, label="Initial mesh")
        optimal, _ = fe.plot(mesh_copy, color="r", linewidth=0.25, label="Optimal mesh")
        plt.legend(handles=[initial, optimal])
        plt.axis("off")
        plt.show()
        plt.pause(0.001)


    def plot_optimization_progress(self):
        fig = plt.figure()
        plt.plot(self.object_values[1:], linestyle='--', marker='o')
        plt.tick_params(labelsize=14)
        plt.xlabel("$N$ (L-BFGS-B steps)", fontsize=14)
        plt.ylabel("$J$ (Objective)", fontsize=14)
        plt.show()


    def plot_force_displacement(self):
        delta_u_recorded = np.load(f'data/numpy/{self.case_name}/step_{self.opt_step}_u.npy')
        sigma_recorded = np.load(f'data/numpy/{self.case_name}/step_{self.opt_step}_f.npy')
        fig = plt.figure(num=self.opt_step)
        plt.ion()
        plt.plot(delta_u_recorded, sigma_recorded, linestyle='--', marker='o', color='red')
        # plt.legend(fontsize=14)
        plt.tick_params(labelsize=14)
        plt.xlabel("Vertical displacement of top side", fontsize=14)
        plt.ylabel("Force on top side", fontsize=14)
        plt.grid(True)
        fig.savefig(f'data/pdf/{self.case_name}/step_{self.opt_step}_force_load.pdf', bbox_inches='tight')
        plt.show()
        plt.pause(0.001)


def plot_comparison(case_name):
    u_initial = np.load(f'data/numpy/{case_name}/step_0_u.npy')
    f_initial = np.load(f'data/numpy/{case_name}/step_0_f.npy')
    u_optimal = np.load(f'data/numpy/{case_name}/step_2_u.npy')
    f_optimal = np.load(f'data/numpy/{case_name}/step_2_f.npy')

    fig = plt.figure()
    plt.plot(u_initial, f_initial, linestyle='--', marker='o', color='blue', label='initial')
    plt.plot(u_optimal, f_optimal, linestyle='--', marker='o', color='red', label='optimized')
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xlabel("Vertical displacement of top side", fontsize=14)
    plt.ylabel("Force on top side", fontsize=14)
    plt.grid(True)
    fig.savefig(f'data/pdf/{case_name}/cmp_force_load.pdf', bbox_inches='tight')
    plt.show()


def test(args):
    # pde = PDE('inverse')
    # pde.run()

    # for i in range(3):
    #     pde = PDE('forward')
    #     pde.run(i)

    for i in range(3):
        pde = PDE('post-processing')
        pde.run(i)



if __name__ == '__main__':
    args = arguments.args
    # test(args)
    plot_comparison('brittle')
