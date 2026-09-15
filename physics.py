"""
Core physics: covariant Lorentz force + Bargmann-Michel-Telegdi (BMT) spin
equation for an electron in a plane-wave laser pulse propagating along +z.

Units c = m_e = |e| = 1, metric (+,-,-,-), electron charge q = -1.
Four-vectors x = (t, x, y, z), u = (gamma, gamma v), spin S with S.u = 0, S.S = -1.

Field
-----
Light-front phase eta = omega (t - z).  Normalized potential a_perp(eta) with
u_perp = u_perp0 + a_perp(eta) - a_perp(0) for the electron:
    linear   : a_x = a0 Env(eta) cos(eta + phi),  a_y = 0
    circular : a_x = a0 Env cos(eta + phi)/sqrt2, a_y = a0 Env sin(eta + phi)/sqrt2
Env = cos^2 envelope over N optical cycles, eta in [0, 2 pi N], zero outside.
E_x = -omega a_x', E_y = -omega a_y', B_x = -E_y, B_y = E_x.

Exact plane-wave reduction (Volkov orbit)
-----------------------------------------
Because k_mu F^{mu nu} = 0 for ANY potential a_perp(eta), envelope and CEP
included, kappa = gamma - u_z is conserved, d eta/d tau = omega kappa, and
u_perp - a_perp is conserved.  The orbit is algebraic in eta:
    u_perp = u_perp0 + a_perp(eta) - a_perp(0)
    gamma  = [kappa + (1 + u_perp^2)/kappa] / 2,   u_z = gamma - kappa.

BMT equation (Jackson form, q = -1, g/2 = 1 + a_e):
    dS^mu/dtau = q [ (g/2) F^{mu nu} S_nu - (g/2 - 1) u^mu (u_a F^{a b} S_b) ]

Closed form for linear polarization, electron initially at rest
---------------------------------------------------------------
The spin is an algebraic function of the instantaneous potential a = a_x(eta)
at ANY g:
    S = (zx a - zz a^2/2,  zx - zz a,  0,  zx a + zz (1 - a^2/2)),
    zx = -sin(a_e a),  zz = cos(a_e a),
i.e. the g = 2 null rotation Lambda_0(a) applied to the initial spin rotated
by a_e a about y.  The rest-frame polarization (pure boost) is
zeta = (-sin Sigma, 0, cos Sigma) with Sigma = 2 arctan(a/2) + a_e a.
"""

import numpy as np
from scipy.integrate import solve_ivp

A_E = 0.00115965218  # electron anomaly (g-2)/2
Q_OVER_M = -1.0


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------
def envelope_cos2(eta, N):
    T = 2 * np.pi * N
    return np.where((eta >= 0) & (eta <= T), np.cos(np.pi * (eta - T / 2) / T) ** 2, 0.0)


def envelope_cos2_deriv(eta, N):
    T = 2 * np.pi * N
    arg = np.pi * (eta - T / 2) / T
    return np.where((eta >= 0) & (eta <= T), -2 * np.cos(arg) * np.sin(arg) * (np.pi / T), 0.0)


class LaserPulse:
    """Plane-wave cos^2-envelope pulse.

    pol = "linear"     : a_x = a0 Env cos(eta+phi), a_y = 0
    pol = "elliptical" : a_x = a0 Env cos(eta+phi)/sqrt(1+delta^2),
                         a_y = delta a0 Env sin(eta+phi)/sqrt(1+delta^2)
    pol = "circular"   : elliptical with delta = 1 (peak |a_perp| = a0/sqrt2)
    """

    def __init__(self, a0, N, phi_cep, omega=1.0, pol="linear", delta=1.0):
        self.a0, self.N, self.phi, self.omega, self.pol = a0, N, phi_cep, omega, pol
        self.delta = 1.0 if pol == "circular" else delta
        self.T = 2 * np.pi * N

    def a(self, eta):
        eta = np.asarray(eta, dtype=float)
        env = envelope_cos2(eta, self.N)
        if self.pol == "linear":
            return self.a0 * env * np.cos(eta + self.phi), np.zeros_like(eta)
        f = self.a0 / np.sqrt(1 + self.delta ** 2)
        return f * env * np.cos(eta + self.phi), self.delta * f * env * np.sin(eta + self.phi)

    def dadEta(self, eta):
        eta = np.asarray(eta, dtype=float)
        env, denv = envelope_cos2(eta, self.N), envelope_cos2_deriv(eta, self.N)
        c, s = np.cos(eta + self.phi), np.sin(eta + self.phi)
        if self.pol == "linear":
            return self.a0 * (denv * c - env * s), np.zeros_like(eta)
        f = self.a0 / np.sqrt(1 + self.delta ** 2)
        return f * (denv * c - env * s), self.delta * f * (denv * s + env * c)

    def signed_area(self, n=200001):
        """A = int (a_x a_y' - a_y a_x') d eta  (twice the signed area of the curve)."""
        eta = np.linspace(0, self.T, n)
        ax, ay = self.a(eta)
        dax, day = self.dadEta(eta)
        return np.trapezoid(ax * day - ay * dax, eta)


def fields(pulse, eta):
    dax, day = pulse.dadEta(eta)
    Ex, Ey = -pulse.omega * dax, -pulse.omega * day
    return Ex, Ey, -Ey, Ex  # Ex, Ey, Bx, By


# ---------------------------------------------------------------------------
# Volkov orbit
# ---------------------------------------------------------------------------
def orbital_solution(pulse, eta, u0):
    g0, ux0, uy0, uz0 = u0
    kappa = g0 - uz0
    ax0, ay0 = pulse.a(0.0)
    ax, ay = pulse.a(eta)
    ux, uy = ux0 + (ax - ax0), uy0 + (ay - ay0)
    up2 = ux ** 2 + uy ** 2
    gamma = 0.5 * (kappa + (1 + up2) / kappa)
    return dict(gamma=gamma, ux=ux, uy=uy, uz=gamma - kappa, kappa=kappa)


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------
def lorentz_rhs(E, B, u4):
    """du^mu/dtau for q/m = -1."""
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    g, ux, uy, uz = u4
    return Q_OVER_M * np.array([Ex * ux + Ey * uy + Ez * uz,
                                Ex * g + Bz * uy - By * uz,
                                Ey * g + Bx * uz - Bz * ux,
                                Ez * g + By * ux - Bx * uy])


def bmt_rhs(E, B, u4, S4, a_e=A_E):
    """dS^mu/dtau = q[(g/2) F^{mu nu} S_nu - (g/2-1) u^mu (u_a F^{ab} S_b)]."""
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    g, ux, uy, uz = u4
    S0, S1, S2, S3 = S4
    FS = np.array([Ex * S1 + Ey * S2 + Ez * S3,
                   Ex * S0 + Bz * S2 - By * S3,
                   Ey * S0 + Bx * S3 - Bz * S1,
                   Ez * S0 + By * S1 - Bx * S2])
    uFS = g * FS[0] - ux * FS[1] - uy * FS[2] - uz * FS[3]
    return Q_OVER_M * ((1 + a_e) * FS - a_e * np.array([g, ux, uy, uz]) * uFS)


def bmt_matrix(E, B, u4, a_e=A_E):
    """4x4 matrix M with dS/dtau = M S (same equation as bmt_rhs)."""
    M = np.empty((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1.0
        M[:, j] = bmt_rhs(E, B, u4, e, a_e)
    return M


def rhs_tau(tau, y, pulse, a_e=A_E):
    """Full covariant system in proper time: y = [t,x,y,z, gamma,ux,uy,uz, S0..S3]."""
    t, _, _, z = y[0:4]
    u4, S4 = y[4:8], y[8:12]
    Ex, Ey, Bx, By = fields(pulse, pulse.omega * (t - z))
    E, B = (Ex, Ey, 0.0), (Bx, By, 0.0)
    return np.concatenate([u4, lorentz_rhs(E, B, u4), bmt_rhs(E, B, u4, S4, a_e)])


def integrate_reference_tau(pulse, u0, S0, tau_span, n_dense=4000, rtol=1e-11, atol=1e-13, a_e=A_E):
    y0 = [0.0, 0.0, 0.0, 0.0, *u0, *S0]
    tau_eval = np.linspace(tau_span[0], tau_span[1], n_dense)
    return solve_ivp(rhs_tau, tau_span, y0, method="DOP853", args=(pulse, a_e),
                     rtol=rtol, atol=atol, t_eval=tau_eval)


def rhs_eta_spin(eta, S, pulse, u0, a_e=A_E):
    """Spin equation in the light-front phase along the algebraic Volkov orbit."""
    orb = orbital_solution(pulse, eta, u0)
    u4 = (orb["gamma"], orb["ux"], orb["uy"], orb["uz"])
    Ex, Ey, Bx, By = fields(pulse, eta)
    return bmt_rhs((Ex, Ey, 0.0), (Bx, By, 0.0), u4, S, a_e) / (pulse.omega * orb["kappa"])


def integrate_lightfront(pulse, u0, S0, eta_span, n_dense=4000, rtol=1e-11, atol=1e-13, a_e=A_E):
    """Reference solver: DOP853 in eta for the spin, algebraic orbit."""
    eta_eval = np.linspace(eta_span[0], eta_span[1], n_dense)
    sol = solve_ivp(rhs_eta_spin, eta_span, S0, method="DOP853", args=(pulse, u0, a_e),
                    rtol=rtol, atol=atol, t_eval=eta_eval)
    return sol, orbital_solution(pulse, eta_eval, u0), eta_eval


# ---------------------------------------------------------------------------
# Closed forms (linear polarization, electron initially at rest, S0 = z-hat)
# ---------------------------------------------------------------------------
def closed_form_spin_linear(a, a_e=A_E):
    """Exact S^mu for u0 = (1,0,0,0), S0 = (0,0,0,1), linear polarization, any g."""
    a = np.asarray(a, dtype=float)
    zx, zz = -np.sin(a_e * a), np.cos(a_e * a)
    return np.array([zx * a - zz * a ** 2 / 2, zx - zz * a, np.zeros_like(a),
                     zx * a + zz * (1 - a ** 2 / 2)])


def sigma_closed_form(a, a_e=A_E):
    """Rest-frame polarization angle: zeta = (-sin Sigma, 0, cos Sigma)."""
    return 2 * np.arctan(np.asarray(a) / 2) + a_e * np.asarray(a)


def rest_frame_spin(S, gamma, ux, uy, uz):
    """zeta = S_vec - S^0 u_vec/(gamma+1)  (pure boost to the rest frame)."""
    f = S[0] / (gamma + 1)
    return np.array([S[1] - f * ux, S[2] - f * uy, S[3] - f * uz])


def theta_S_lab(S):
    """Angle between the spatial part of the laboratory four-vector S and z-hat."""
    return np.arctan2(np.hypot(S[1], S[2]), S[3])
