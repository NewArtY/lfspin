"""
Fixed-step integrators for the convergence study (co-propagating electron):

  (i)   lab-time RK4          : state (x, z, gamma, u, S) advanced in lab time t
  (ii)  Boris--BMT            : standard relativistic Boris push for u (leapfrog,
                                x at integer, u and S at half-integer steps); the
                                spin is advanced by the Cayley transform of the BMT
                                generator evaluated at the integer step, i.e. the
                                same second-order, norm-preserving construction as
                                the Boris velocity rotation
  (iii) light-front RK4       : spin equation in eta, orbit algebraic (Volkov)
"""
import numpy as np

from physics import fields, orbital_solution, lorentz_rhs, bmt_rhs, bmt_matrix, A_E


def rhs_labtime(t, y, pulse, a_e=A_E):
    x, z, g, ux, uy, uz = y[0:6]
    Ex, Ey, Bx, By = fields(pulse, pulse.omega * (t - z))
    E, B, u4 = (Ex, Ey, 0.0), (Bx, By, 0.0), (g, ux, uy, uz)
    return np.concatenate([[ux / g, uz / g], lorentz_rhs(E, B, u4) / g,
                           bmt_rhs(E, B, u4, y[6:10], a_e) / g])


def rk4_fixed(rhs, t0, t1, y0, n_steps, pulse):
    h = (t1 - t0) / n_steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(n_steps):
        k1 = rhs(t, y, pulse)
        k2 = rhs(t + h / 2, y + h / 2 * k1, pulse)
        k3 = rhs(t + h / 2, y + h / 2 * k2, pulse)
        k4 = rhs(t + h, y + h * k3, pulse)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def boris_bmt(t0, t1, y0, n_steps, pulse, a_e=A_E, scheme="boris"):
    """Leapfrog pusher with BMT spin.  scheme = "boris" (Boris 1970) or "hc"
    (Higuera & Cary 2017, which evaluates the magnetic rotation with the gamma that
    makes E + v x B cancellation exact).  Returns the final state in the layout of
    rhs_labtime.  The field vanishes for eta <= 0, so u_{-1/2} = u_0 and
    S_{-1/2} = S_0 exactly."""
    h = (t1 - t0) / n_steps
    x, z, _, ux, uy, uz = y0[0:6]
    u = np.array([ux, uy, uz])
    S = np.array(y0[6:10], dtype=float)
    t = t0
    I4 = np.eye(4)
    for _ in range(n_steps):
        Ex, Ey, Bx, By = fields(pulse, pulse.omega * (t - z))
        E, B = np.array([Ex, Ey, 0.0]), np.array([Bx, By, 0.0])
        eps = -E * h / 2  # (q/m) E h/2 with q/m = -1
        um = u + eps
        if scheme == "boris":
            gm = np.sqrt(1 + um @ um)
            tv = -B * h / (2 * gm)
            sv = 2 * tv / (1 + tv @ tv)
            up = um + np.cross(um + np.cross(um, tv), sv)
            un = up + eps
        else:  # Higuera-Cary
            tau = -B * h / 2
            tau2 = tau @ tau
            ustar = um @ tau
            sigma = 1 + um @ um - tau2
            gnew = np.sqrt(0.5 * (sigma + np.sqrt(sigma ** 2 + 4 * (tau2 + ustar ** 2))))
            tv = tau / gnew
            s = 1 / (1 + tv @ tv)
            up = s * (um + (um @ tv) * tv + np.cross(um, tv))
            un = up + eps + np.cross(up, tv)
        # spin: Cayley transform of the generator at the integer step
        uc = 0.5 * (u + un)
        gc = np.sqrt(1 + uc @ uc)
        A = (h / (2 * gc)) * bmt_matrix(E, B, np.array([gc, *uc]), a_e)
        S = np.linalg.solve(I4 - A, (I4 + A) @ S)
        gn = np.sqrt(1 + un @ un)
        x += h * un[0] / gn
        z += h * un[2] / gn
        u = un
        t += h
    return np.array([x, z, np.sqrt(1 + u @ u), *u, *S])


def lightfront_rk4_fixed(pulse, u0, S0, eta0, eta1, n_steps, a_e=A_E):
    kappa = u0[0] - u0[3]

    def rhs(eta, S):
        orb = orbital_solution(pulse, eta, u0)
        u4 = (orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
        Ex, Ey, Bx, By = fields(pulse, eta)
        return bmt_rhs((Ex, Ey, 0.0), (Bx, By, 0.0), u4, S, a_e) / (pulse.omega * kappa)

    h = (eta1 - eta0) / n_steps
    eta, S = eta0, np.array(S0, dtype=float)
    for _ in range(n_steps):
        k1 = rhs(eta, S)
        k2 = rhs(eta + h / 2, S + h / 2 * k1)
        k3 = rhs(eta + h / 2, S + h / 2 * k2)
        k4 = rhs(eta + h, S + h * k3)
        S = S + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        eta += h
    return float(orbital_solution(pulse, eta1, u0)["gamma"]), S
