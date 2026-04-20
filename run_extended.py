"""
run_extended.py
===============
New experiments for v2 of the paper:

  EXP_IN  : In-span benchmark (u* is a linear combination of K_0..K_{S*}).
            The natural sweet-spot regime for the fNKp method.

  EXP_OUT : Out-of-span benchmark (u*(t,w) = t^{1+alpha} (w(1-w))^r).
            All methods pay an approximation error; used to confirm the old
            v1 observation without cherry-picking.

  EXP_SS  : S-convergence on the in-span benchmark using the linear-head
            variant.  Shows monotone decrease in S (v1 showed divergence).

  EXP_FDM : Classical FDM baseline on the same in-span problem.

All results saved to results_extended.pt.  Pass --only to choose a subset.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

warnings.filterwarnings("ignore")

from fnkp_core_v2 import (
    StableFNKpBasis, LinearHeadFNKpPINN, MLPHeadFNKpPINN,
    VanillaFPINN, BMLPINN,
    InSpanTarget, FDMReference,
    caputo_t, gl_rl_w, rl_derivative_monomial, rl_derivative_w_q_exp_mw,
)

RESULTS_FILE = Path(__file__).parent / "results_extended.pt"

# ==========================================================================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N_seeds", type=int, default=3)
    p.add_argument("--epochs",  type=int, default=800)
    p.add_argument("--Nr",      type=int, default=1024)
    p.add_argument("--Nb",      type=int, default=256)
    p.add_argument("--lr",      type=float, default=3e-3)
    p.add_argument("--lb",      type=float, default=10.0)
    p.add_argument("--width",   type=int,   default=64)
    p.add_argument("--depth",   type=int,   default=4)
    p.add_argument("--alpha",   type=float, default=0.7)
    p.add_argument("--beta",    type=float, default=1.5)
    p.add_argument("--kappa",   type=float, default=1.0)
    p.add_argument("--GL_N",    type=int,   default=20)
    p.add_argument("--n_q",     type=int,   default=12)
    p.add_argument("--device",  type=str,   default="auto")
    p.add_argument("--only",    type=str,   default="IN,OUT,SS,FDM")
    return p.parse_args()


def load_results():
    if RESULTS_FILE.exists():
        return torch.load(RESULTS_FILE, map_location="cpu", weights_only=False)
    return {}


def save_results(r):
    torch.save(r, RESULTS_FILE)


# ==========================================================================
# Sampling utilities
# ==========================================================================

def sample_int(N, device):
    t = torch.rand(N, device=device).clamp(1e-3, 1-1e-3)
    w = torch.rand(N, device=device).clamp(1e-3, 1-1e-3)
    return t, w


def sample_bd(N, device):
    """
    Boundary samples:  (t=0, w), (t, w=0), (t, w=1).
    For the in-span target, u*(t, 0) = 0 is automatic but we still add
    collocation there to help the optimiser.
    """
    n3 = N // 3
    t0 = torch.zeros(n3, device=device); w0 = torch.rand(n3, device=device)
    t1 = torch.rand(n3, device=device);  w1 = torch.zeros(n3, device=device)
    t2 = torch.rand(N - 2*n3, device=device); w2 = torch.ones(N - 2*n3, device=device)
    return torch.cat([t0,t1,t2]), torch.cat([w0,w1,w2])


# ==========================================================================
# Training with a given target (InSpanTarget-like API)
# ==========================================================================

def train_on_target(model, tgt, args, device, method: str, l1_lambda: float = 0.0):
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    alpha, beta, kappa = args.alpha, args.beta, args.kappa
    h_gl = 1.0 / args.GL_N

    history = {"rel_l2_history": [], "loss_r_history": [], "epoch_history": []}

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()

        t_r, w_r = sample_int(args.Nr, device)
        t_b, w_b = sample_bd(args.Nb, device)
        u_b_true = tgt.u(t_b, w_b)
        f_vals   = tgt.source(t_r, w_r, alpha, beta, kappa)

        cap = caputo_t(lambda ti, wi: model(ti, wi), t_r, w_r, alpha,
                       n_q=args.n_q)

        if method == "LinearFNKp":
            rl = model.rl_w(t_r, w_r, beta)
        elif method == "MLPFNKp":
            # Approximate chain rule on the nonlinear head
            Phi        = model.basis(t_r, w_r).detach().requires_grad_(True)
            _, q_val   = model.basis.get_p_q()
            w_q        = w_r.clamp(min=1e-12).pow(q_val).detach()
            v          = model.head(Phi)
            u_via_phi  = w_q * v
            jac        = torch.autograd.grad(u_via_phi.sum(), Phi, create_graph=True)[0]
            # D_w^beta (w^q K_s) for each s  (exact)
            dphi_beta  = model.basis.rl_times_wq(t_r, w_r, beta)
            # Approximate: use the identity linearly on the basis side
            rl         = (jac / w_q.unsqueeze(1) * dphi_beta).sum(dim=1)
        else:   # fPINN / bML
            rl = gl_rl_w(lambda ti, wi: model(ti, wi), t_r, w_r, beta,
                         args.GL_N, h_gl)

        res  = cap - kappa * rl - f_vals
        lr_  = (res**2).mean()
        lb_  = ((model(t_b, w_b) - u_b_true)**2).mean()
        loss = lr_ + args.lb * lb_
        if l1_lambda > 0.0:
            if hasattr(model, "a"):              # LinearHeadFNKpPINN
                w_abs = model.a.abs().sum()
            elif hasattr(model, "head"):         # MLP head
                w_abs = sum(p.abs().sum()
                            for p in model.head.parameters()
                            if p.ndim >= 2)
            else:
                w_abs = torch.zeros((), device=loss.device)
            loss = loss + l1_lambda * w_abs

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

        if ep == 1 or ep % 200 == 0:
            with torch.no_grad():
                t_t = torch.linspace(0.01, 0.99, 80, device=device)
                w_t = torch.linspace(0.01, 0.99, 80, device=device)
                T, W = torch.meshgrid(t_t, w_t, indexing='ij')
                Tf, Wf = T.reshape(-1), W.reshape(-1)
                ut = tgt.u(Tf, Wf); up = model(Tf, Wf)
                rl2 = (torch.norm(up - ut) / torch.norm(ut)).item()
            history["rel_l2_history"].append(rl2)
            history["loss_r_history"].append(float(lr_))
            history["epoch_history"].append(ep)

    # Final dense eval
    model.eval()
    with torch.no_grad():
        N_test = 200
        t_t = torch.linspace(0.005, 0.995, N_test, device=device)
        w_t = torch.linspace(0.005, 0.995, N_test, device=device)
        T, W = torch.meshgrid(t_t, w_t, indexing='ij')
        Tf, Wf = T.reshape(-1), W.reshape(-1)
        ut = tgt.u(Tf, Wf); up = model(Tf, Wf)
        final = (torch.norm(up - ut) / torch.norm(ut)).item()

    final_loss_r = float(lr_)
    return {
        "final_rel_l2": final,
        "final_loss_r": final_loss_r,
        "elapsed": time.time() - t0,
        "history": history,
    }


# ==========================================================================
# Out-of-span target:  u*(t, w) = t^{1+alpha} * w^q * (1 - w)^r * exp(-w)
# (Same class as the v1 benchmark but with BC-respecting prefactor.)
# ==========================================================================

class OutOfSpanTarget:
    def __init__(self, q: float = 0.8, r: float = 1.0):
        self.q = float(q)
        self.r = float(r)

    def u(self, t, w):
        t = t.clamp(min=1e-12); w = w.clamp(min=1e-12, max=1.0 - 1e-12)
        return t.pow(1.0 + 0.7) * w.pow(self.q) * (1.0 - w).pow(self.r) * torch.exp(-w)

    def rl_w_u(self, t, w, beta):
        """
        Series:  w^q (1-w)^r exp(-w)  =  sum_{j,k,m} C_{jkm} w^{q + j + k}
        (expand (1-w)^r as a finite sum if r is integer, else series).
        For r integer >= 1, (1-w)^r = sum_{l=0}^{r} C(r,l) (-w)^l.
        """
        assert float(self.r) == int(self.r) and int(self.r) >= 0, \
            "r must be a non-negative integer"
        r_int = int(self.r)
        result = torch.zeros_like(w)
        for l in range(r_int + 1):
            C = math.comb(r_int, l) * ((-1.0) ** l)
            # w^q * (-w)^l / C = w^{q+l} * C
            # Then times exp(-w) = Σ_k (-1)^k w^k / k!
            part = torch.zeros_like(w)
            for k in range(30):
                coef = ((-1.0) ** k) / math.factorial(k)
                part = part + coef * rl_derivative_monomial(w, self.q + l + k, beta)
            result = result + C * part
        return t.pow(1.0 + 0.7) * result

    def source(self, t, w, alpha, beta, kappa):
        # D_t^alpha [t^{1+alpha}] = Gamma(2+alpha) * t
        cap_t = math.gamma(2.0 + alpha) * t * \
                w.pow(self.q) * (1.0 - w).pow(self.r) * torch.exp(-w)
        rl_w  = self.rl_w_u(t, w, beta)
        return cap_t - kappa * rl_w


# ==========================================================================
# EXP_IN: in-span benchmark
# ==========================================================================

def run_EXP_IN(results, args, device):
    print("\n=== EXP_IN: in-span benchmark ===", flush=True)

    S_star = 2
    p0, q0 = 10.0, 0.8
    coeffs = torch.tensor([1.0, 0.3, -0.2])
    ups    = 1
    tgt    = InSpanTarget(S_star, p0, q0, coeffs, upsilon=ups)

    configs = [
        ("LinearFNKp", lambda: LinearHeadFNKpPINN(S=S_star, upsilon=ups,
                                                    p_init=p0, q_init=q0,
                                                    learn_pq=False)),
        ("fPINN",      lambda: VanillaFPINN(width=args.width, depth=args.depth)),
        ("bML",        lambda: BMLPINN(width=args.width, depth=args.depth,
                                        upsilon=ups, p_init=3.0, q_init=2.0,
                                        n_terms=10)),
    ]
    for name, factory in configs:
        for seed in range(args.N_seeds):
            key = f"IN|{name}|seed{seed}"
            if key in results:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  Running {key} ...", end=" ", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = factory().to(device)
            res   = train_on_target(model, tgt, args, device, name)
            results[key] = res
            save_results(results)
            print(f"rel_l2={res['final_rel_l2']:.4e}  "
                  f"loss_r={res['final_loss_r']:.2e}  "
                  f"({res['elapsed']/60:.1f} min)", flush=True)


# ==========================================================================
# EXP_OUT: out-of-span benchmark (all methods pay approximation floor)
# ==========================================================================

def run_EXP_OUT(results, args, device):
    print("\n=== EXP_OUT: out-of-span benchmark ===", flush=True)

    q_t = 0.8
    r_t = 1
    tgt = OutOfSpanTarget(q=q_t, r=r_t)

    # For the linear-head variant we need an S-truncation; try S=4 to give it
    # enough flexibility.
    S_use  = 4
    p0, q0 = 12.0, q_t       # match basis q to exact solution's w^{q_t} prefactor

    configs = [
        ("LinearFNKp", lambda: LinearHeadFNKpPINN(S=S_use, upsilon=1,
                                                    p_init=p0, q_init=q0,
                                                    learn_pq=False)),
        ("fPINN",      lambda: VanillaFPINN(width=args.width, depth=args.depth)),
        ("bML",        lambda: BMLPINN(width=args.width, depth=args.depth,
                                        upsilon=1, p_init=3.0, q_init=2.0,
                                        n_terms=10)),
    ]
    for name, factory in configs:
        for seed in range(args.N_seeds):
            key = f"OUT|{name}|seed{seed}"
            if key in results:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  Running {key} ...", end=" ", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = factory().to(device)
            res   = train_on_target(model, tgt, args, device, name)
            results[key] = res
            save_results(results)
            print(f"rel_l2={res['final_rel_l2']:.4e}  "
                  f"loss_r={res['final_loss_r']:.2e}  "
                  f"({res['elapsed']/60:.1f} min)", flush=True)


# ==========================================================================
# EXP_SS: S-convergence on in-span benchmark (linear head only)
# ==========================================================================

def run_EXP_SS(results, args, device):
    print("\n=== EXP_SS: S-convergence (linear head) ===", flush=True)

    S_star = 4
    p0, q0 = 12.0, 0.8
    coeffs_full = torch.tensor([1.0, 0.3, -0.2, 0.1, -0.05])
    tgt = InSpanTarget(S_star, p0, q0, coeffs_full, upsilon=1)

    for S in [2, 3, 4, 5, 6]:
        for seed in range(args.N_seeds):
            key = f"SS|S{S}|seed{seed}"
            if key in results:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  Running {key} ...", end=" ", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = LinearHeadFNKpPINN(
                S=S, upsilon=1,
                p_init=max(p0, 2*S + 2.5),
                q_init=q0, learn_pq=False).to(device)
            res = train_on_target(model, tgt, args, device, "LinearFNKp")
            results[key] = res
            save_results(results)
            print(f"rel_l2={res['final_rel_l2']:.4e}  "
                  f"({res['elapsed']/60:.1f} min)", flush=True)


# ==========================================================================
# EXP_NQ: n_q (Caputo Gauss-Jacobi) sensitivity on in-span benchmark
# ==========================================================================

def run_EXP_NQ(results, args, device):
    print("\n=== EXP_NQ: n_q sensitivity (linear head, in-span) ===", flush=True)

    S_star = 2
    p0, q0 = 10.0, 0.8
    coeffs = torch.tensor([1.0, 0.3, -0.2])
    ups    = 1
    tgt    = InSpanTarget(S_star, p0, q0, coeffs, upsilon=ups)

    default_n_q = args.n_q
    for nq in [4, 6, 8, 12, 16, 20]:
        for seed in range(args.N_seeds):
            key = f"NQ|n{nq}|seed{seed}"
            if key in results:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  Running {key} ...", end=" ", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = LinearHeadFNKpPINN(S=S_star, upsilon=ups,
                                       p_init=p0, q_init=q0,
                                       learn_pq=False).to(device)
            args.n_q = nq
            res   = train_on_target(model, tgt, args, device, "LinearFNKp")
            results[key] = res
            results[key]["n_q"] = nq
            save_results(results)
            print(f"rel_l2={res['final_rel_l2']:.4e}  "
                  f"({res['elapsed']/60:.1f} min)", flush=True)
    args.n_q = default_n_q


# ==========================================================================
# EXP_FAIR: fPINN with varying GL_N to address "unfair comparison" critique
# ==========================================================================

def run_EXP_FAIR(results, args, device):
    print("\n=== EXP_FAIR: fPINN GL_N sensitivity on in-span ===", flush=True)

    S_star = 2
    p0, q0 = 10.0, 0.8
    coeffs = torch.tensor([1.0, 0.3, -0.2])
    ups    = 1
    tgt    = InSpanTarget(S_star, p0, q0, coeffs, upsilon=ups)

    default_GL_N = args.GL_N
    for gl in [20, 40, 80, 160]:
        for seed in range(args.N_seeds):
            key = f"FAIR|fPINN|GL{gl}|seed{seed}"
            if key in results:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  Running {key} ...", end=" ", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = VanillaFPINN(width=args.width, depth=args.depth).to(device)
            args.GL_N = gl
            res   = train_on_target(model, tgt, args, device, "fPINN")
            results[key] = res
            results[key]["GL_N"] = gl
            save_results(results)
            print(f"rel_l2={res['final_rel_l2']:.4e}  "
                  f"loss_r={res['final_loss_r']:.2e}  "
                  f"({res['elapsed']/60:.1f} min)", flush=True)
    args.GL_N = default_GL_N


# ==========================================================================
# EXP_L1: l1-regularised linear head at over-parametrised S > S*
# ==========================================================================

def run_EXP_L1(results, args, device):
    print("\n=== EXP_L1: l1 penalty for S > S* ===", flush=True)

    # Same target as EXP_SS: S*=4 in-span.  Test at S=6 with l1 in [1e-4, 1e-1].
    S_star = 4
    p0, q0 = 12.0, 0.8
    coeffs_full = torch.tensor([1.0, 0.3, -0.2, 0.1, -0.05])
    tgt = InSpanTarget(S_star, p0, q0, coeffs_full, upsilon=1)

    S_over = 6
    for l1 in [1e-4, 1e-3, 1e-2, 1e-1]:
        for seed in range(args.N_seeds):
            key = f"L1|S{S_over}|lam{l1:.0e}|seed{seed}"
            if key in results:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  Running {key} ...", end=" ", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = LinearHeadFNKpPINN(
                S=S_over, upsilon=1,
                p_init=max(p0, 2*S_over + 2.5),
                q_init=q0, learn_pq=False).to(device)
            res = train_on_target(model, tgt, args, device, "LinearFNKp",
                                   l1_lambda=l1)
            results[key] = res
            results[key]["l1_lambda"] = l1
            save_results(results)
            print(f"rel_l2={res['final_rel_l2']:.4e}  "
                  f"({res['elapsed']/60:.1f} min)", flush=True)


# ==========================================================================
# EXP_OOS_S: S-convergence on out-of-span (non-polynomial) target
# ==========================================================================

def run_EXP_OOS_S(results, args, device):
    print("\n=== EXP_OOS_S: S-convergence on out-of-span target ===", flush=True)

    q_t = 0.8
    r_t = 1
    tgt = OutOfSpanTarget(q=q_t, r=r_t)
    p0, q0 = 12.0, q_t

    for S in [2, 4, 6, 8, 10]:
        for seed in range(args.N_seeds):
            key = f"OOS_S|S{S}|seed{seed}"
            if key in results:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  Running {key} ...", end=" ", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = LinearHeadFNKpPINN(
                S=S, upsilon=1,
                p_init=max(p0, 2*S + 2.5),
                q_init=q0, learn_pq=False).to(device)
            res = train_on_target(model, tgt, args, device, "LinearFNKp")
            results[key] = res
            save_results(results)
            print(f"rel_l2={res['final_rel_l2']:.4e}  "
                  f"({res['elapsed']/60:.1f} min)", flush=True)


# ==========================================================================
# EXP_GAL: direct Galerkin linear-solve baseline (no Adam)
# ==========================================================================

def run_EXP_GAL(results, args, device):
    print("\n=== EXP_GAL: direct Galerkin linear solve ===", flush=True)

    S_star = 2
    p0, q0 = 10.0, 0.8
    coeffs = torch.tensor([1.0, 0.3, -0.2])
    ups    = 1
    tgt    = InSpanTarget(S_star, p0, q0, coeffs, upsilon=ups)
    alpha, beta, kappa = args.alpha, args.beta, args.kappa

    for S in [2, 4, 6]:
        key = f"GAL|S{S}"
        if key in results:
            print(f"  [skip] {key}", flush=True); continue
        print(f"  Running {key} ...", end=" ", flush=True)
        t0 = time.time()
        torch.manual_seed(0); np.random.seed(0)

        model = LinearHeadFNKpPINN(S=S, upsilon=ups,
                                    p_init=max(p0, 2*S+2.5),
                                    q_init=q0, learn_pq=False).to(device)
        basis = model.basis
        _, q_val = basis.get_p_q()

        t_r, w_r = sample_int(args.Nr, device)
        t_b, w_b = sample_bd(args.Nb, device)
        u_b_true = tgt.u(t_b, w_b)
        f_vals   = tgt.source(t_r, w_r, alpha, beta, kappa)

        cap_mat = torch.zeros(len(t_r), S+1, device=device)
        for s in range(S+1):
            def u_s_fn(ti, wi, _s=s):
                Phi = basis(ti, wi)
                wq  = wi.clamp(min=1e-12).pow(q_val)
                return wq * Phi[:, _s]
            cap_mat[:, s] = caputo_t(u_s_fn, t_r, w_r, alpha, n_q=args.n_q).detach()

        rl_mat = basis.rl_times_wq(t_r, w_r, beta).detach()
        A = (cap_mat - kappa * rl_mat)

        w_b_q = w_b.clamp(min=1e-12).pow(q_val)
        Phi_b = basis(t_b, w_b).detach()
        B     = (w_b_q.unsqueeze(1) * Phi_b)

        sqrt_lb = math.sqrt(args.lb)
        M = torch.cat([A, sqrt_lb * B], dim=0)
        y = torch.cat([f_vals.detach(), sqrt_lb * u_b_true.detach()], dim=0)

        theta = torch.linalg.lstsq(M, y.unsqueeze(1)).solution.squeeze(1)
        with torch.no_grad():
            model.a.copy_(theta)

        model.eval()
        N_test = 200
        t_t = torch.linspace(0.005, 0.995, N_test, device=device)
        w_t = torch.linspace(0.005, 0.995, N_test, device=device)
        T, W = torch.meshgrid(t_t, w_t, indexing='ij')
        Tf, Wf = T.reshape(-1), W.reshape(-1)
        with torch.no_grad():
            u_true = tgt.u(Tf, Wf)
            u_pred = model(Tf, Wf)
            rel_l2 = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()

        cond_M = float(torch.linalg.cond(M.detach()).item())
        elapsed = time.time() - t0
        results[key] = {"final_rel_l2": rel_l2, "elapsed": elapsed,
                         "S": S, "cond_M": cond_M}
        save_results(results)
        print(f"rel_l2={rel_l2:.4e}  cond={cond_M:.2e}  ({elapsed:.1f} s)",
              flush=True)


# ==========================================================================
# EXP_FDM: classical FDM baseline on the same in-span PDE
# ==========================================================================

def run_EXP_FDM(results, args, device):
    print("\n=== EXP_FDM: classical FDM reference ===", flush=True)

    # For FDM we require q0 >= beta so the source f = D_t^a u - kappa D_w^b u
    # is bounded at w = 0.  With q0 < beta, f ~ w^{q0-beta} is singular and
    # standard shifted-Grunwald diverges under mesh refinement.
    S_star = 2
    p0, q0 = 10.0, 2.0
    coeffs = torch.tensor([1.0, 0.3, -0.2])
    tgt    = InSpanTarget(S_star, p0, q0, coeffs, upsilon=1)
    alpha, beta, kappa = args.alpha, args.beta, args.kappa

    def u0_fn(w_arr):
        t = torch.zeros(len(w_arr)); w = torch.tensor(w_arr, dtype=torch.float32)
        return tgt.u(t, w).numpy()

    def g0_fn(t_arr):
        return np.zeros_like(t_arr)

    def gW_fn(t_arr):
        t = torch.tensor(t_arr, dtype=torch.float32); w = torch.ones_like(t)
        return tgt.u(t, w).numpy()

    def src_fn(t_scalar, w_vec):
        t = torch.full((len(w_vec),), float(t_scalar), dtype=torch.float32)
        w = torch.tensor(w_vec, dtype=torch.float32)
        return tgt.source(t, w, alpha, beta, kappa).numpy()

    N_test = 200
    t_t = np.linspace(0.005, 0.995, N_test)
    w_t = np.linspace(0.005, 0.995, N_test)
    T, W = np.meshgrid(t_t, w_t, indexing='ij')
    Tf, Wf = T.reshape(-1), W.reshape(-1)
    u_true = tgt.u(torch.tensor(Tf, dtype=torch.float32),
                    torch.tensor(Wf, dtype=torch.float32)).numpy()

    for Nw in [40, 80, 160, 320]:
        key = f"FDM|Nw{Nw}"
        if key in results:
            print(f"  [skip] {key}", flush=True); continue
        print(f"  Running {key} ...", end=" ", flush=True)
        t0 = time.time()
        solver = FDMReference(alpha, beta, kappa, T=1.0, W=1.0, Nt=Nw, Nw=Nw)
        solver.solve(u0_fn, g0_fn, gW_fn, src_fn)
        u_fdm = solver.evaluate(Tf, Wf)
        rel_l2 = float(np.linalg.norm(u_fdm - u_true) / np.linalg.norm(u_true))
        elapsed = time.time() - t0
        results[key] = {"final_rel_l2": rel_l2, "elapsed": elapsed,
                         "Nw": Nw, "Nt": Nw}
        save_results(results)
        print(f"rel_l2={rel_l2:.4e}  ({elapsed:.1f} s)", flush=True)


# ==========================================================================
def main():
    args = get_args()
    only = set(x.strip().upper() for x in args.only.split(","))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Device: {device}  Seeds: {args.N_seeds}  Epochs: {args.epochs}  Nr: {args.Nr}",
          flush=True)
    print(f"Only: {sorted(only)}", flush=True)
    results = load_results()

    if "IN"    in only: run_EXP_IN    (results, args, device)
    if "OUT"   in only: run_EXP_OUT   (results, args, device)
    if "SS"    in only: run_EXP_SS    (results, args, device)
    if "NQ"    in only: run_EXP_NQ    (results, args, device)
    if "FAIR"  in only: run_EXP_FAIR  (results, args, device)
    if "L1"    in only: run_EXP_L1    (results, args, device)
    if "OOS_S" in only: run_EXP_OOS_S (results, args, device)
    if "GAL"   in only: run_EXP_GAL   (results, args, device)
    if "FDM"   in only: run_EXP_FDM   (results, args, device)

    print("\nAll requested experiments completed.", flush=True)


if __name__ == "__main__":
    main()
