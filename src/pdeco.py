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
from .constitutive import *

rank = MPI.COMM_WORLD.Get_rank()


class PDECO(object):
    def __init__(self, problem):
        self.staggered_tol = 1e-5
        self.staggered_maxiter = 1000
        self.solution_scheme = 'explicit'   
        self.problem = problem 
        self.periodic = None
        self.preparation()
        

    def run(self, opt_step=0):
        if self.problem == 'inverse':
            self.build_mesh()
            self.move_mesh()
            self.staggered_solve()
            self.adjoint_optimization()
        elif self.problem == 'forward':
            self.opt_step = opt_step
            self.build_mesh(f'data/xdmf/{self.case_name}/inverse/mesh_{self.opt_step}.xdmf')
            self.staggered_solve()
        elif self.problem == 'post-processing':
            self.opt_step = opt_step
            self.plot_force_displacement()
            plt.ioff()
            plt.show()
        elif self.problem == 'debug':
            self.opt_step = opt_step
            self.build_mesh(f'data/xdmf/{self.case_name}/inverse/mesh_{self.opt_step}.xdmf')
            self.staggered_solve()
        else:
            raise ValueError('Unknown problem mode!')


    def preparation(self):
        if self.problem == 'inverse' and rank == 0:
            print(f"\nDelete inverse problem related data...")
            shutil.rmtree(f'data/pvd/{self.case_name}/{self.problem}', ignore_errors=True)
            shutil.rmtree(f'data/xdmf/{self.case_name}/{self.problem}', ignore_errors=True)


    def move_mesh(self):
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

        S = fe.VectorFunctionSpace(self.mesh, "CG", 1, constrained_domain=self.periodic)
        u, v = fe.TrialFunction(S), fe.TestFunction(S)

        def epsilon(u):
            return fe.sym(fe.grad(u))

        def sigma(u, mu=1., lmb=0.):
            return 2 * mu * epsilon(u) + lmb * fe.tr(epsilon(u)) * fe.Identity(2)

        a = fe.inner(sigma(u, mu=mu), fe.grad(v)) * fe.dx
        L = fe.inner(h_V, v) * self.ds(2)

        bcs = []
        for side in [self.lower, self.upper, self.left, self.right]:
            bcs.append(da.DirichletBC(S, zero, side))

        s = da.Function(S, name="mesh deformation")
        da.solve(a == L, s, bcs=bcs)

        return s


    def build_weak_form_staggered(self):
        self.psi = partial(psi_linear_elasticity, lamda=self.lamda, mu=self.mu)

        if self.i == 0:
            self.psi_plus = partial(psi_plus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
            self.psi_minus = partial(psi_minus_linear_elasticity_model_A, lamda=self.lamda, mu=self.mu)
            print("use model A")
        else:
            self.psi_plus = partial(psi_plus_linear_elasticity_model_C, lamda=self.lamda, mu=self.mu)
            self.psi_minus = partial(psi_minus_linear_elasticity_model_C, lamda=self.lamda, mu=self.mu)
            print("use model C")

        self.sigma = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi)
        self.sigma_plus = cauchy_stress_plus(strain(fe.grad(self.x_new)), self.psi_plus)
        self.sigma_minus = cauchy_stress_minus(strain(fe.grad(self.x_new)), self.psi_minus)

        if self.solution_scheme == 'explicit':
            history = self.history
        else:
            history = get_history(self.history, self.psi_plus(strain(fe.grad(self.x_new))))

        self.G_d = (history * self.zeta * g_d_prime(self.d_new, g_d) \
                    + self.G_c / self.l0 * (self.zeta * self.d_new + self.l0**2 * fe.inner(fe.grad(self.zeta), fe.grad(self.d_new)))) * fe.dx
  
        self.G_u = (g_d(self.d_new) * fe.inner(self.sigma_plus, strain(fe.grad(self.eta))) \
            + fe.inner(self.sigma_minus, strain(fe.grad(self.eta)))) * fe.dx


    def staggered_solve(self):
        self.U = fe.VectorFunctionSpace(self.mesh, 'CG', 1, constrained_domain=self.periodic)
        self.W = fe.FunctionSpace(self.mesh, 'CG', 1) 
        self.V = fe.FunctionSpace(self.mesh, 'DG', 0)

        self.eta = fe.TestFunction(self.U)
        self.zeta = fe.TestFunction(self.W)

        del_x = fe.TrialFunction(self.U)
        del_d = fe.TrialFunction(self.W)

        self.x_new = da.Function(self.U, name="u")
        self.d_new = da.Function(self.W, name="d")
        self.history = da.Function(self.V)
 
        x_old = fe.Function(self.U)
        d_old = fe.Function(self.W) 

        if self.problem == 'forward' or self.problem == 'debug':
            xdmf_file_sols = fe.XDMFFile(MPI.COMM_WORLD, f'data/xdmf/{self.case_name}/{self.problem}/step_{self.opt_step}/sols.xdmf')
            xdmf_file_sols.parameters["functions_share_mesh"] = True
            vtk_file_u = fe.File(f'data/pvd/{self.case_name}/{self.problem}/step_{self.opt_step}/u.pvd')
            vtk_file_d = fe.File(f'data/pvd/{self.case_name}/{self.problem}/step_{self.opt_step}/d.pvd')

        delta_u_recorded = []
        sigma_recorded = []  

        self.J = 0
        for i, (disp, rp) in enumerate(zip(self.displacements, self.relaxation_parameters)):
            if rank == 0:
                print('\n')
                print('=================================================================================')
                print(f'>> Step {i}, disp boundary condition = {disp} [mm]')
                print('=================================================================================')

            self.i = i
            if self.i == 0 or self.i == 2:
                self.build_weak_form_staggered()
                J_u = fe.derivative(self.G_u, self.x_new, del_x)
                J_d = fe.derivative(self.G_d, self.d_new, del_d)

                self.set_bcs_staggered()
                p_u = da.NonlinearVariationalProblem(self.G_u, self.x_new, self.BC_u, J_u)
                p_d  = da.NonlinearVariationalProblem(self.G_d,  self.d_new, self.BC_d, J_d)
                solver_u = da.NonlinearVariationalSolver(p_u)
                solver_d  = da.NonlinearVariationalSolver(p_d)

            self.presLoad.t = disp

            newton_prm = solver_u.parameters['newton_solver']
            newton_prm['maximum_iterations'] = 100
            newton_prm['relaxation_parameter'] = rp
 
            self.history.assign(da.project(get_history(self.history, self.psi_plus(strain(fe.grad(self.x_new)))), self.V))

            iteration = 0
            err = 1.
            while err > self.staggered_tol:
                iteration += 1

                solver_d.solve()
                solver_u.solve()

                if self.solution_scheme == 'explicit':
                    break

                err_x = fe.errornorm(self.x_new, x_old, norm_type='l2')
                err_d = fe.errornorm(self.d_new, d_old, norm_type='l2')
                err = max(err_x, err_d) 

                x_old.assign(self.x_new)
                d_old.assign(self.d_new)

                print('---------------------------------------------------------------------------------')
                print(f'>> iteration. {iteration}, err_u = {err_x:.5}, err_d = {err_d:.5}, error = {err:.5}')
                print('---------------------------------------------------------------------------------')

                if err < self.staggered_tol or iteration >= self.staggered_maxiter:
                    print('=================================================================================\n')
                    break

            if self.problem == 'forward' or self.problem == 'debug':
                xdmf_file_sols.write(self.x_new, i)
                xdmf_file_sols.write(self.d_new, i)
                vtk_file_u << self.x_new
                vtk_file_d << self.d_new

            force_upper = self.update_objective_in_the_loop()
            if rank == 0:
                print(f"Force upper {force_upper}")

            delta_u_recorded.append(disp)
            sigma_recorded.append(force_upper)

        self.update_objective_reg()

        if self.problem == 'forward':
            np.save(f'data/numpy/{self.case_name}/step_{self.opt_step}_u.npy', np.array(delta_u_recorded))
            np.save(f'data/numpy/{self.case_name}/step_{self.opt_step}_f.npy', np.array(sigma_recorded))

        if self.problem == 'debug':
            self.plot_force_displacement_helper(delta_u_recorded, sigma_recorded)

        return float(self.J)


    def adjoint_optimization(self):
        self.object_values = []

        def eval_callback(j, m):
            '''
            Hint: m has the same type as self.h 
            Might be useful.
            '''
            # print(f"Parameter is {np.array(m.vector())}")
            print(f"Objective is {j}")
            self.object_values.append(j)


        control = da.Control(self.h)
        Jhat = reduced_functional = da.ReducedFunctional(self.J, control, eval_cb_post=eval_callback)

        def iteration_callback(x):
            ''' Callback function for optimization

            Parameter
            ---------
            x: 1D numpy array
            '''
            iteration_callback.count += 1
            mesh_copy = fe.Mesh(self.mesh_initial)
            h_callback = fe.Function(self.S_b)
            h_callback.vector()[:] = x
            s = self.mesh_deformation(h_callback)
            fe.ALE.move(mesh_copy, s)
            # self.plot_mesh(mesh_copy)
            vtkfile_mesh << mesh_copy
            xdmf_file_mesh = fe.XDMFFile(MPI.COMM_WORLD, f'data/xdmf/{self.case_name}/{self.problem}/mesh_{iteration_callback.count}.xdmf')
            xdmf_file_mesh.write(mesh_copy)
            print(f"Optimizer iter call back, max h {np.max(np.absolute(x))}")


        iteration_callback.count = 0
        xdmf_file_mesh = fe.XDMFFile(MPI.COMM_WORLD, f'data/xdmf/{self.case_name}/{self.problem}/mesh_{iteration_callback.count}.xdmf')
        xdmf_file_mesh.write(self.mesh)
        vtkfile_mesh = fe.File(f'data/pvd/{self.case_name}/{self.problem}/mesh.pvd')
        vtkfile_mesh << self.mesh

        # h_opt = da.minimize(reduced_functional, method="CG", tol=1e-20, callback=iteration_callback,
        #     options={"disp": True, "maxiter": 2})

        h_opt = da.minimize(reduced_functional, method="L-BFGS-B", tol=1e-20, bounds=(-5., 5.), callback=iteration_callback,
            options={"disp": True, "maxiter": 2})


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


    def plot_force_displacement_helper(self, delta_u_recorded, sigma_recorded):
        fig = plt.figure(num=self.opt_step)
        plt.plot(delta_u_recorded, sigma_recorded, linestyle='--', marker='o', color='red')
        # plt.legend(fontsize=14)
        plt.tick_params(labelsize=14)
        plt.xlabel("Vertical displacement of top side", fontsize=14)
        plt.ylabel("Force on top side", fontsize=14)
        plt.grid(True)
        return fig


    def plot_force_displacement(self):
        delta_u_recorded = np.load(f'data/numpy/{self.case_name}/step_{self.opt_step}_u.npy')
        sigma_recorded = np.load(f'data/numpy/{self.case_name}/step_{self.opt_step}_f.npy')
        fig = self.plot_force_displacement_helper(delta_u_recorded, sigma_recorded)
        fig.savefig(f'data/pdf/{self.case_name}/step_{self.opt_step}_force_load.pdf', bbox_inches='tight')
        plt.ion()
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

