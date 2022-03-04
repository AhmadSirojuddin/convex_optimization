"""
Copyright 2022 Ahmad Sirojuddin
mail : sirojuddin.p@gmail.com

This file contains the optimization problem format that is usually solved using the water-filling (WF) algorithm.
The given problem example, besides is solved the WF algorithm, is also solved using cvxpy for comparison. All variables
are implemented using the torch framework.
"""

import torch as tr
import time
import cvxpy as cvx     # for comparison

def water_filling(a, b, c, d, f, P, tol, iter_max):
    """
    See the attached file for more explanation about this function!
    Given the following optimization problem:
    min     -sum_{n=1}^N a_n log(b_n+ p_n c_n)
    s.t.    p_n >= d_n,
            f^T p = P,
    this function find vector p that solve the problem.

    Arguments:
    (1) a : 1D positive-entries tensor
    (2) b : 1D non-negative-entries tensor having the same size with a
    (3) c : 1D positive-entries tensor having the same size with a
    (4) d : 1D non-negative-entries tensor having the same size with a
    (5) f : 1D positive-entries tensor having the same size with a
    (6) P : positive scalar
    (7) tol : tolerance for the stopping criterion. It must be a positive scalar
    (8) iter_max : maximum allowed iteration for a stopping criterion. It must be a positive integer scalar.

    Outputs:
    (1) obj_val : the optimal value of the objective function
    (2) p : the optimal primal variable p
    (3) exec_time : the execution time
    (4) g_nu_trace : the recorded value of g (see the attached file) during iteration.
                     this variable may useful to plot the algorithm's convergence.
    """

    if tr.is_complex(a) or tr.any(a <= 0) or a.ndim != 1:
        print("WARNING!!! input 'a' must be real, positive, and number of dimension = 1")
        print("your input 'a' = ", a)
        raise ValueError('INPUT ERROR')
    N = a.size()
    if tr.is_complex(b) or tr.any(b < 0) or b.size() != N:
        print("WARNING!!! input 'b' must be real, non-negative, and has the same size with a")
        print("your input 'b' = ", b)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(c) or tr.any(c <= 0) or c.size() != N:
        print("WARNING!!! input 'c' must be real, positive, and has the same size with a")
        print("your input 'c' = ", c)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(d) or tr.any(d < 0) or d.size() != N:
        print("WARNING!!! input 'd' must be real, non-negative, and has the same size with a")
        print("your input 'd' = ", d)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(f) or tr.any(f <= 0) or f.size() != N:
        print("WARNING!!! input 'f' must be real, positive, and has the same size with a")
        print("your input 'f' = ", f)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(P) or P <= 0 or P.numel() != 1:
        print("WARNING!!! input 'P' must be real, scalar, and positive")
        print("your input 'P' = ", P)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(tol) or tol <= 0 or tol.numel() != 1:
        print("WARNING!!! input 'tol' must be real, scalar, and positive")
        print("your input 'tol' = ", tol)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(iter_max) or iter_max <= 0 or iter_max.numel() != 1 or tr.is_floating_point(iter_max):
        print("WARNING!!! input 'iter_max' must be real, scalar, positive, and integer")
        print("your input 'iter_max' = ", iter_max)
        raise ValueError('INPUT ERROR')

    if tr.dot(f, d) >= P:
        print("WARNING!!! The constraints is not feasible")
        raise ValueError("INFEASIBLE CONSTRAINTS")

    start_timer = time.time()
    iter_th = 0
    nu = (-tr.dot(f, d) + P + tr.dot(f, b/c+d)) / tr.sum(a)  # ; print("nu = ", nu)
    g_nu = 0
    g_nu_trace = tr.empty([1])  # ; print("g_nu_trace = ", g_nu_trace)

    while iter_th <= iter_max:
        # print("iter_th = ", iter_th, " ---------------")
        g_nu = tr.sum(f * tr.max(nu * a / f - b / c - d, tr.zeros(N))) + tr.dot(f, d) - P; # print("   obj_val = ", g_nu)
        g_nu_trace = tr.cat([g_nu_trace, g_nu], dim=0)  # ; print("   g_nu_trace = ", g_nu_trace)
        if tr.abs(g_nu) < tol:
            # print("      obj_val is smaller than tolerance, stop the iteration")
            break
        dg_dnu = tr.sum(a * (nu * a * f - b / c - d >= 0)); # print("   dg_dnu = ", dg_dnu)
        nu = nu - g_nu / dg_dnu  # ; print("   nu = ", nu)
        iter_th += 1
    p = tr.max(nu*a/f - b/c, d)  # ; print("p = ", p); print("sum p = ", p.sum())

    exec_time = time.time() - start_timer
    g_nu_trace = g_nu_trace[1:iter_th+2]
    obj_val = -tr.sum(a * tr.log(b + p * c))  # ; print("obj_val = ", obj_val)
    return obj_val, p, exec_time, g_nu_trace

# Example
def example():
    N = 10
    a = tr.abs(tr.randn(N))
    b = tr.abs(tr.randn(N))
    c = tr.abs(tr.randn(N))
    d = tr.abs(tr.randn(N))
    f = tr.abs(tr.randn(N))
    P = tr.tensor([N * 20])

    # -------------- WATER FILLING -------------- #
    tol = tr.tensor(1e-3)
    iter_max = tr.tensor([50])
    obj_val, p, time_exec, g_nu_trace = water_filling(a, b, c, d, f, P, tol, iter_max)
    print("obj_val (water filling) = ", obj_val)
    print("optimal p (water filling) = ", p)
    print("time_exec (water filling) = ", time_exec)
    # print("g_nu_trace = ", g_nu_trace)
    print("----------------------------------------------")

    # -------------- CVXPY WATER FILLING PROBLEM -------------- #
    start = time.time()
    p = cvx.Variable(N)
    objective = cvx.Minimize(-cvx.sum(cvx.multiply(a, cvx.log(b + cvx.multiply(p, c)))))
    constraints = [p >= d, cvx.sum(cvx.multiply(f, p)) == P]
    prob = cvx.Problem(objective, constraints)
    time_exec = time.time() - start

    obj_val = prob.solve(verbose=False, solver='ECOS')
    print("result (cvx) = ", obj_val)
    print("optimal p (cvx) = ", p.value)
    print("time_exec (cvx)= ", time_exec)


example()
