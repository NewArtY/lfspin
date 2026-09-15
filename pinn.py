"""
Physics-informed neural network (PINN) for the light-front orbital + BMT system.

Network: eta -> Fourier features [eta/T, sin(k eta), cos(k eta)]_{k=1..k_max}
         -> tanh MLP -> (gamma, ux, uy, uz, S0, S1, S2, S3).
k_max = 0 gives the plain-input baseline (features = eta/T only).

Initial conditions are hard-wired (switch tanh(eta/w)), gamma(0)=1, u(0)=0,
S(0)=(0,0,0,1).  The loss contains only ODE residuals (orbital and BMT blocks,
d/dtau = kappa d/deta with kappa = 1 for an electron starting at rest) and the
invariant penalties u.u - 1 and S.S + 1; no trajectory data are used.
Everything is float64 and seeded.
"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn

from physics import A_E


def fourier_features(eta, k_max=8, omega=1.0, T=1.0):
    feats = [eta / T]
    for k in range(1, k_max + 1):
        feats.append(torch.sin(k * omega * eta))
        feats.append(torch.cos(k * omega * eta))
    return torch.cat(feats, dim=1)


class PINN(nn.Module):
    def __init__(self, n_hidden=128, n_layers=5, ic_width=0.35, k_max=8, omega=1.0, T=1.0):
        super().__init__()
        self.k_max, self.omega, self.T, self.ic_width = k_max, omega, T, ic_width
        layers = [nn.Linear(1 + 2 * k_max, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 8)]
        self.net = nn.Sequential(*layers).double()

    def forward(self, eta):
        raw = self.net(fourier_features(eta, self.k_max, self.omega, self.T))
        g, ux, uy, uz, S0, S1, S2, S3 = torch.split(raw, 1, dim=1)
        sw = torch.tanh(eta / self.ic_width)
        return (1.0 + sw * torch.nn.functional.softplus(g), sw * ux, sw * uy, sw * uz,
                sw * S0, sw * S1, sw * S2, 1.0 + sw * S3)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def field_torch(eta, a0, N, phi, omega=1.0, delta=None):
    """Plane-wave fields; delta=None: linear (x), otherwise elliptical with ellipticity delta
    (same convention as physics.LaserPulse)."""
    T = 2 * np.pi * N
    inside = (eta >= 0) & (eta <= T)
    arg = np.pi * (eta - T / 2) / T
    env = torch.where(inside, torch.cos(arg) ** 2, torch.zeros_like(eta))
    denv = torch.where(inside, -2 * torch.cos(arg) * torch.sin(arg) * (np.pi / T), torch.zeros_like(eta))
    c, s = torch.cos(eta + phi), torch.sin(eta + phi)
    if delta is None:
        Ex = -omega * a0 * (denv * c - env * s)
        Ey = torch.zeros_like(eta)
    else:
        f = a0 / np.sqrt(1 + delta ** 2)
        Ex = -omega * f * (denv * c - env * s)
        Ey = -omega * delta * f * (denv * s + env * c)
    return Ex, Ey, -Ey, Ex.clone()  # Ex, Ey, Bx, By


def pinn_residuals(model, eta, a0, N, phi, omega=1.0, delta=None):
    eta = eta.clone().requires_grad_(True)
    gamma, ux, uy, uz, S0, S1, S2, S3 = model(eta)

    def d(y):
        return torch.autograd.grad(y, eta, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True)[0]

    Ex, Ey, Bx, By = field_torch(eta, a0, N, phi, omega, delta)
    Ez = Bz = torch.zeros_like(eta)
    q = -1.0
    scale = 1.0 / omega  # kappa = 1

    res_g = d(gamma) - scale * q * (Ex * ux + Ey * uy + Ez * uz)
    res_ux = d(ux) - scale * q * (Ex * gamma + Bz * uy - By * uz)
    res_uy = d(uy) - scale * q * (Ey * gamma + Bx * uz - Bz * ux)
    res_uz = d(uz) - scale * q * (Ez * gamma + By * ux - Bx * uy)

    FS0 = Ex * S1 + Ey * S2 + Ez * S3
    FS1 = Ex * S0 + Bz * S2 - By * S3
    FS2 = Ey * S0 + Bx * S3 - Bz * S1
    FS3 = Ez * S0 + By * S1 - Bx * S2
    uFS = gamma * FS0 - ux * FS1 - uy * FS2 - uz * FS3
    res_S = [d(S) - scale * q * ((1 + A_E) * FS - A_E * uc * uFS)
             for S, FS, uc in ((S0, FS0, gamma), (S1, FS1, ux), (S2, FS2, uy), (S3, FS3, uz))]

    massshell = gamma ** 2 - ux ** 2 - uy ** 2 - uz ** 2 - 1.0
    spinnorm = S0 ** 2 - S1 ** 2 - S2 ** 2 - S3 ** 2 + 1.0
    ode_res = torch.cat([res_g, res_ux, res_uy, res_uz, *res_S], dim=1)
    return ode_res, torch.cat([massshell, spinnorm], dim=1)


def loss_terms(model, eta, a0, N, phi, omega=1.0, delta=None):
    ode_res, inv_res = pinn_residuals(model, eta, a0, N, phi, omega, delta)
    return (torch.mean(ode_res[:, 0:4] ** 2), torch.mean(ode_res[:, 4:8] ** 2),
            torch.mean(inv_res ** 2))


def train_adam(a0, N, phi, omega=1.0, n_steps=32000, n_colloc=150, lr=1e-3, seed=0,
               print_every=2000, k_max=8, n_hidden=128, n_layers=5, ramp_steps=15000,
               grad_clip=1.0, checkpoint=None, delta=None):
    """Adam with cosine LR decay, per-step collocation resampling (uniform +
    pulse-centre + edges), linear weight ramp (3,1) -> (8,3), invariant weight 20,
    best-loss checkpointing.  If `checkpoint` (a file path) is given, the full
    training state (model, optimizer, scheduler, best state, history, RNG states)
    is saved every `print_every` steps and training resumes from it if present;
    a resumed run follows the same trajectory as an uninterrupted one."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = 2 * np.pi * N
    model = PINN(n_hidden=n_hidden, n_layers=n_layers, k_max=k_max, omega=omega, T=T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)
    best_loss, best_state, history = float("inf"), None, []
    start, elapsed0 = 0, 0.0
    if checkpoint and os.path.exists(checkpoint):
        ck = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best_loss, best_state, history = ck["best_loss"], ck["best_state"], ck["history"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"])
        start, elapsed0 = ck["step"], ck["elapsed"]
        print(f"resumed from {checkpoint} at step {start}", flush=True)
    t0 = time.time() - elapsed0
    for step in range(start, n_steps):
        if checkpoint and step > start and step % print_every == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            best_loss=best_loss, best_state=best_state, history=history,
                            torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(),
                            step=step, elapsed=time.time() - t0), checkpoint + ".tmp")
            os.replace(checkpoint + ".tmp", checkpoint)
        opt.zero_grad()
        eta_all = torch.cat([
            torch.rand((n_colloc, 1), dtype=torch.float64) * T,
            torch.rand((n_colloc, 1), dtype=torch.float64) * T * 0.7 + T * 0.15,
            torch.rand((n_colloc // 6, 1), dtype=torch.float64) * (0.05 * T),
            T - torch.rand((n_colloc // 6, 1), dtype=torch.float64) * (0.05 * T),
        ], dim=0)
        l_orb, l_spin, l_inv = loss_terms(model, eta_all, a0, N, phi, omega, delta)
        frac = min(step / ramp_steps, 1.0)
        loss = (3.0 + 5.0 * frac) * l_orb + (1.0 + 2.0 * frac) * l_spin + 20.0 * l_inv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        lval = loss.item()
        if lval < best_loss:
            best_loss, best_state = lval, copy.deepcopy(model.state_dict())
        if step % print_every == 0 or step == n_steps - 1:
            history.append((step, lval, l_orb.item(), l_spin.item(), l_inv.item()))
            print(f"step {step:6d} loss {lval:.4e} orb {l_orb.item():.3e} spin {l_spin.item():.3e} "
                  f"inv {l_inv.item():.3e} best {best_loss:.4e} t={time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    return model, history, best_loss


def polish_lbfgs(model, a0, N, phi, omega=1.0, n_colloc=1200, max_iter=3000, delta=None):
    T = 2 * np.pi * N
    eta_fixed = torch.linspace(0, T, n_colloc, dtype=torch.float64).reshape(-1, 1)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                            max_eval=int(max_iter * 1.25), history_size=100,
                            tolerance_grad=1e-16, tolerance_change=1e-18,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l_orb, l_spin, l_inv = loss_terms(model, eta_fixed, a0, N, phi, omega, delta)
        loss = 8.0 * l_orb + 3.0 * l_spin + 20.0 * l_inv
        loss.backward()
        return loss

    opt.step(closure)
    return model
