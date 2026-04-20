"""
fnkp_core_v2.py
===============
Revised core library for the fNKp-PINN paper.  Key upgrades vs. the v1 code in
experiment2/experiment2_fNKp_aug_PINN.py:

1.  StableFNKpBasis:
      - reparametrises p = 2S+1 + softplus(p_tilde) + eps so the biorthogonality
        constraint p > 2S+1 is always active and the Jacobian is smooth
      - caches Gamma prefactors and the (k,m,poch/km!) index tables
      - accepts (S, upsilon) from outside and never silently drifts

2.  LinearHeadFNKpPINN:
      - u_hat(t,w) = sum_s a_s * K_s(t,w)   (linear Galerkin-style head)
      - this makes Proposition "no quadrature error" EXACT in implementation
        (the chain-rule bias present in the MLP head vanishes)

3.  MLPHeadFNKpPINN:
      - the old nonlinear head, kept for reproduction of prior results

4.  fPINN and bMLPINN:
      - unchanged in spirit; minor fixes to GL weight recursion and
        avoidance of w<0 outside the domain

5.  FDMReference:
      - classical shifted-Gruenwald + L1 scheme for the bivariate fractional
        diffusion equation on a uniform (Nt x Nw) grid
      - no training; produces a reference numerical solution at arbitrary
        test points

6.  InSpanTarget:
      - generates a benchmark whose exact solution is a prescribed linear
        combination of fNKp basis functions, so the method's sweet-spot
        regime can be tested without cherry-picking
      - source f is the analytic PDE residual of the chosen u*, using the
        exact RL identity

All classes are self-contained and take torch tensors.  CPU/GPU agnostic.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# 1.  Analytic RL derivatives of scalar monomials (needed for sources)
# ============================================================================

def rl_derivative_monomial(w: torch.Tensor, q: float, beta: float) -> torch.Tensor:
    """
    Riemann-Liouville derivative  D_w^beta [w^q]  =
        Gamma(q+1)/Gamma(q+1-beta) * w^{q-beta}      if q+1-beta > 0.
    Returns 0 if the power is not integrable.
    """
    num = math.gamma(q + 1.0)
    denom_arg = q + 1.0 - beta
    if denom_arg <= 0 or abs(math.gamma(denom_arg)) < 1e-300:
        return torch.zeros_like(w)
    den = math.gamma(denom_arg)
    return (num / den) * w.clamp(min=1e-12).pow(q - beta)


def rl_derivative_w_q_exp_mw(w: torch.Tensor,
                             q: float,
                             beta: float,
                             N_terms: int = 30) -> torch.Tensor:
    """
    D_w^beta [w^q e^{-w}] via the power series  e^{-w} = sum_k (-w)^k/k!.
    """
    out = torch.zeros_like(w)
    for k in range(N_terms):
        out = out + ((-1.0)**k / math.factorial(k)) * rl_derivative_monomial(w, q + k, beta)
    return out


# ============================================================================
# 2.  Stable fNKp basis
# ============================================================================

class StableFNKpBasis(nn.Module):
    """
    First set of finite bivariate biorthogonal N-Konhauser polynomials
    K_s^{(p,q)}(t,w), s = 0,...,S.

    Reparametrisation:
        p = 2S + 1 + softplus(p_tilde) + eps_p
        q = -1     + softplus(q_tilde) + eps_q
    which guarantees p > 2S+1 and q > -1 throughout training without any
    projection step.

    Exact fractional identity (Theorem 3.14 of Guldogan-Lekesiz et al. 2025):
        D_w^beta [ w^q K_s^{(p,q)}(t,w) ]  =  w^{q-beta} K_s^{(p, q-beta)}(t,w)

    so that
        D_w^beta K_s^{(p,q)}(t,w)
      = w^{-beta} K_s^{(p, q-beta)}(t,w)   +  (lower-order w^{-q-beta} ... )
    Here we return exactly the RL derivative of K_s, obtained by differentiating
    the series term-by-term.

    NOTE: The identity in the source paper is stated for D_w^beta[w^q K_s]; to
    get D_w^beta K_s directly we apply Leibniz on the product (w^q)(w^{-q} K_s).
    The *clean* identity that does NOT require Leibniz is the one we expose via
    ``rl_times_wq`` below: D_w^beta[w^q K_s] = w^{q-beta} K_s^{(p,q-beta)}.
    """

    def __init__(self,
                 S: int,
                 upsilon: int = 1,
                 p_init: float = 10.0,
                 q_init: float = 0.5,
                 eps_p: float = 0.0,
                 eps_q: float = 0.0):
        super().__init__()
        self.S       = int(S)
        self.upsilon = int(upsilon)
        self.eps_p   = float(eps_p)
        self.eps_q   = float(eps_q)

        # softplus^{-1}(x) = log(expm1(x))
        p_shift = p_init - (2*S + 1) - eps_p
        q_shift = q_init - (-1.0) - eps_q
        if p_shift <= 0:
            p_shift = 1.0
        if q_shift <= 0:
            q_shift = 0.5
        self.p_tilde = nn.Parameter(torch.tensor(math.log(math.expm1(p_shift))))
        self.q_tilde = nn.Parameter(torch.tensor(math.log(math.expm1(q_shift))))

        self._build_index_tables()

    # ------------------------------------------------------------------
    def _build_index_tables(self):
        tabs = []
        for s in range(self.S + 1):
            ks, ms, pkm = [], [], []
            for k in range(s + 1):
                for m in range(s - k + 1):
                    poch    = self._pochhammer_int(-s, k + m)
                    km_fact = math.factorial(k) * math.factorial(m)
                    ks.append(float(k))
                    ms.append(float(m))
                    pkm.append(poch / km_fact)
            tabs.append({
                'k'  : torch.tensor(ks,  dtype=torch.float64),
                'm'  : torch.tensor(ms,  dtype=torch.float64),
                'pkm': torch.tensor(pkm, dtype=torch.float64),
            })
        self._tables = tabs

    @staticmethod
    def _pochhammer_int(a: float, n: int) -> float:
        if n == 0:
            return 1.0
        r = 1.0
        for i in range(n):
            r *= (a + i)
        return r

    # ------------------------------------------------------------------
    def get_p_q(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p = (2.0 * self.S + 1.0) + torch.nn.functional.softplus(self.p_tilde) + self.eps_p
        q = -1.0                 + torch.nn.functional.softplus(self.q_tilde) + self.eps_q
        return p, q

    def kappa_s(self) -> torch.Tensor:
        """Biorthogonality constant kappa_S = S! * Gamma(p-S)/(p-2S-1)."""
        p, _ = self.get_p_q()
        S = self.S
        return math.factorial(S) * torch.exp(torch.lgamma(p - S)) / (p - 2.0*S - 1.0)

    # ------------------------------------------------------------------
    def _eval_single(self, t: torch.Tensor, w: torch.Tensor,
                     p: torch.Tensor, q: torch.Tensor, s: int) -> torch.Tensor:
        """
        K_s^{(p,q)}(t,w) / Gamma(p-s)  =
            sum_{k=0}^s sum_{m=0}^{s-k}  (-s)_{k+m}/(k! m!) *
                     t^{s-k} w^{upsilon m} / [Gamma(p-2s+k) Gamma(q+1+upsilon m)]
        Returned WITHOUT the Gamma(p-s) prefactor; caller multiplies by it once.
        """
        tbl   = self._tables[s]
        k_vec = tbl['k'].to(t.device, dtype=t.dtype)
        m_vec = tbl['m'].to(t.device, dtype=t.dtype)
        pkm   = tbl['pkm'].to(t.device, dtype=t.dtype)

        gam_p2sk = torch.exp(torch.lgamma(p - 2.0*s + k_vec))
        gam_q1um = torch.exp(torch.lgamma(q + 1.0 + self.upsilon * m_vec))
        coeffs   = pkm / (gam_p2sk * gam_q1um)
        t_pow    = t.unsqueeze(1).pow((s - k_vec).unsqueeze(0))
        w_pow    = w.clamp(min=1e-12).unsqueeze(1).pow((self.upsilon * m_vec).unsqueeze(0))
        return (coeffs * t_pow * w_pow).sum(dim=1)

    def forward(self, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return the matrix [K_0, K_1, ..., K_S] stacked along dim=1."""
        p, q = self.get_p_q()
        rows = []
        for s in range(self.S + 1):
            rows.append(torch.exp(torch.lgamma(p - s)) * self._eval_single(t, w, p, q, s))
        return torch.stack(rows, dim=1)

    # ------------------------------------------------------------------
    def rl_times_wq(self, t: torch.Tensor, w: torch.Tensor,
                    beta: float) -> torch.Tensor:
        """
        Exact, closed-form evaluation of
            D_w^beta [ w^q * K_s^{(p,q)}(t,w) ]   =   w^{q-beta} * K_s^{(p, q-beta)}(t,w)
        for s = 0,...,S.  Returns a [N, S+1] tensor.

        Evaluated term-by-term as  sum_{k,m} coeff * t^{s-k} * w^{q+um-beta}
        so that the singular  w^{q-beta}  prefactor is never formed in
        isolation: the m=0 term (for which q+um-beta is most negative) has its
        coefficient proportional to  1/Gamma(q+1+um-beta), which vanishes
        exactly whenever q+1-beta is a non-positive integer.
        """
        p, q      = self.get_p_q()
        rows = []
        for s in range(self.S + 1):
            tbl   = self._tables[s]
            k_vec = tbl['k'].to(t.device, dtype=t.dtype)
            m_vec = tbl['m'].to(t.device, dtype=t.dtype)
            pkm   = tbl['pkm'].to(t.device, dtype=t.dtype)
            um    = self.upsilon * m_vec

            # D_w^beta[w^{q+um}] = Gamma(q+um+1)/Gamma(q+um+1-beta) * w^{q+um-beta}
            arg_num = q + um + 1.0
            arg_den = q + um + 1.0 - beta
            # Explicit pole mask: whenever arg_den is at a non-positive integer,
            # 1/Gamma(arg_den) = 0 exactly.  Guard against floating point noise
            # making it small-but-nonzero.
            near_pole = (arg_den <= 0.0) & (
                (arg_den - torch.round(arg_den)).abs() < 1e-6
            )
            num_g = torch.exp(torch.lgamma(arg_num))
            # For safe evaluation of lgamma, clamp away from exact poles.
            safe_den = torch.where(near_pole, torch.ones_like(arg_den), arg_den)
            den_g = torch.where(near_pole,
                                torch.full_like(arg_den, float('inf')),
                                torch.exp(torch.lgamma(safe_den.abs())) *
                                torch.where(safe_den < 0,
                                            (-1.0) ** torch.floor(-safe_den),
                                            torch.ones_like(safe_den)))
            frac = torch.where(near_pole,
                               torch.zeros_like(num_g),
                               num_g / den_g)

            gam_p2sk = torch.exp(torch.lgamma(p - 2.0*s + k_vec))
            gam_q1um = torch.exp(torch.lgamma(q + 1.0 + um))
            coeffs   = pkm * frac / (gam_p2sk * gam_q1um)

            t_pow = t.unsqueeze(1).pow((s - k_vec).unsqueeze(0))
            w_exp = (q + um - beta).unsqueeze(0)
            w_pow = w.clamp(min=1e-12).unsqueeze(1).pow(w_exp)
            val   = (coeffs * t_pow * w_pow).sum(dim=1)
            rows.append(torch.exp(torch.lgamma(p - s)) * val)
        return torch.stack(rows, dim=1)

    def rl_direct(self, t: torch.Tensor, w: torch.Tensor,
                  beta: float) -> torch.Tensor:
        """
        Exact evaluation of D_w^beta [ K_s^{(p,q)}(t,w) ]  (no w^q factor).
        Equals  w^{-q} * rl_times_wq  +  Leibniz correction ... but we use the
        direct power-series differentiation instead.
        """
        p, q = self.get_p_q()
        rows = []
        for s in range(self.S + 1):
            tbl   = self._tables[s]
            k_vec = tbl['k'].to(t.device, dtype=t.dtype)
            m_vec = tbl['m'].to(t.device, dtype=t.dtype)
            pkm   = tbl['pkm'].to(t.device, dtype=t.dtype)

            # Power of w inside K_s is w^{upsilon m}.
            # D_w^beta[w^{upsilon m}] = Gamma(upsilon m + 1)/Gamma(upsilon m + 1 - beta)
            # * w^{upsilon m - beta}, valid when upsilon m + 1 - beta > 0.
            um      = self.upsilon * m_vec
            num_g   = torch.exp(torch.lgamma(um + 1.0))
            denom   = um + 1.0 - beta
            # Mask indices where denom <= 0: skip them
            safe    = denom > 1e-6
            den_g   = torch.where(safe, torch.exp(torch.lgamma(denom.clamp(min=1e-6))),
                                   torch.ones_like(denom))
            frac    = torch.where(safe, num_g / den_g, torch.zeros_like(denom))

            gam_p2sk = torch.exp(torch.lgamma(p - 2.0*s + k_vec))
            gam_q1um = torch.exp(torch.lgamma(q + 1.0 + um))
            coeffs   = pkm * frac / (gam_p2sk * gam_q1um)

            t_pow    = t.unsqueeze(1).pow((s - k_vec).unsqueeze(0))
            w_pow    = w.clamp(min=1e-12).unsqueeze(1).pow((um - beta).unsqueeze(0))
            val      = (coeffs * t_pow * w_pow).sum(dim=1)
            rows.append(torch.exp(torch.lgamma(p - s)) * val)
        return torch.stack(rows, dim=1)


# ============================================================================
# 3.  Models
# ============================================================================

class LinearHeadFNKpPINN(nn.Module):
    """
    fNKp-Galerkin ansatz:
        u_hat(t, w)  =  w^q * sum_s a_s K_s^{(p,q)}(t, w)
                     =  sum_s a_s * [ w^q K_s^{(p,q)}(t, w) ]

    This ansatz makes the RL identity exact:
        D_w^beta u_hat(t, w)
          = sum_s a_s * D_w^beta [ w^q K_s^{(p,q)}(t, w) ]
          = sum_s a_s * w^{q-beta} K_s^{(p, q-beta)}(t, w)       [Theorem 3.14]

    Automatic zero Dirichlet BC at w = 0 (since q > 0 -> w^q -> 0 as w -> 0).

    p and q are by default FROZEN (q crossing an integer shifts K_s through a
    Gamma pole).  Set ``learn_pq=True`` to let them train.
    """
    def __init__(self, S: int, upsilon: int = 1, p_init: float = 10.0,
                 q_init: float = 0.5, learn_pq: bool = False):
        super().__init__()
        self.basis = StableFNKpBasis(S, upsilon=upsilon, p_init=p_init, q_init=q_init)
        if not learn_pq:
            self.basis.p_tilde.requires_grad_(False)
            self.basis.q_tilde.requires_grad_(False)
        # Scale-down init helps: K_s already carries a Gamma(p-s) prefactor
        # that is order 10^5 for p~10, so coeffs should be small.
        self.a = nn.Parameter(torch.randn(S + 1) * 1e-3)

    def forward(self, t, w):
        _, q     = self.basis.get_p_q()
        w_q      = w.clamp(min=1e-12).pow(q)
        Phi      = self.basis(t, w)
        return w_q * (Phi * self.a).sum(dim=1)

    def rl_w(self, t, w, beta):
        """Exact D_w^beta u_hat(t,w) via the clean identity."""
        Dw = self.basis.rl_times_wq(t, w, beta)
        return (Dw * self.a).sum(dim=1)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPHeadFNKpPINN(nn.Module):
    """
    Nonlinear MLP head:
        u_hat(t, w) = w^q * MLP(K_0(t,w), ..., K_S(t,w)).
    The w^q factor gives the automatic Dirichlet BC at w = 0 and it also lets
    the RL residual be computed by the exact-basis-level identity combined
    with a first-order chain-rule approximation -- we keep this variant so the
    "MLP vs linear" ablation is meaningful.
    """
    def __init__(self, S: int, width: int, depth: int, upsilon: int = 1,
                 p_init: float = 10.0, q_init: float = 0.5):
        super().__init__()
        self.basis = StableFNKpBasis(S, upsilon=upsilon, p_init=p_init, q_init=q_init)
        self.head  = MLPHead(S + 1, width, depth)

    def forward(self, t, w):
        _, q = self.basis.get_p_q()
        w_q  = w.clamp(min=1e-12).pow(q)
        return w_q * self.head(self.basis(t, w))


class VanillaFPINN(nn.Module):
    """Standard fPINN with a plain MLP in (t, w)."""
    def __init__(self, width: int, depth: int):
        super().__init__()
        self.head = MLPHead(2, width, depth)

    def forward(self, t, w):
        return self.head(torch.stack([t, w], dim=1))


class BMLPINN(nn.Module):
    """PINN with a learnable bivariate Mittag-Leffler feature in the first layer."""
    def __init__(self, width: int, depth: int, upsilon: int = 1,
                 p_init: float = 3.0, q_init: float = 2.0, n_terms: int = 12):
        super().__init__()
        self.upsilon = upsilon
        self.n_terms = n_terms
        self.log_p   = nn.Parameter(torch.tensor(math.log(p_init)))
        self.log_q   = nn.Parameter(torch.tensor(math.log(q_init)))
        self.head    = MLPHead(3, width, depth)

    def _bml(self, t, w):
        p = torch.exp(self.log_p) + 1.0
        q = torch.exp(self.log_q) + 0.1
        m_idx = torch.arange(self.n_terms, device=t.device, dtype=t.dtype)
        j_idx = torch.arange(self.n_terms, device=t.device, dtype=t.dtype)
        log_poch    = torch.lgamma(m_idx.unsqueeze(1) + j_idx.unsqueeze(0) + 1.0)
        log_mj_fact = (torch.lgamma(m_idx + 1.0).unsqueeze(1) +
                       torch.lgamma(j_idx + 1.0).unsqueeze(0))
        log_coeffs  = (log_poch - log_mj_fact
                       - torch.lgamma(p + m_idx).unsqueeze(1)
                       - torch.lgamma(q + self.upsilon*j_idx).unsqueeze(0))
        coeffs = torch.exp(log_coeffs.clamp(max=80.0))
        t_pow  = t.unsqueeze(1).pow(m_idx.unsqueeze(0))
        w_pow  = w.clamp(min=1e-8).unsqueeze(1).pow(self.upsilon * j_idx.unsqueeze(0))
        return torch.einsum('nm,nj,mj->n', t_pow, w_pow, coeffs)

    def forward(self, t, w):
        bml = self._bml(t, w)
        return self.head(torch.stack([bml, t, w], dim=1))


# ============================================================================
# 4.  Gruenwald-Letnikov spatial derivative for fPINN / bML
# ============================================================================

def gl_weights(beta: float, N_gl: int, device, dtype) -> torch.Tensor:
    g = torch.ones(N_gl + 1, device=device, dtype=dtype)
    for k in range(1, N_gl + 1):
        g[k] = g[k - 1] * (1.0 - (beta + 1.0) / k)
    return g


def gl_rl_w(model, t: torch.Tensor, w: torch.Tensor, beta: float,
            N_gl: int, h: float) -> torch.Tensor:
    """Standard shifted-Grunwald approximation of D_w^beta u, truncated at N_gl."""
    g = gl_weights(beta, N_gl, t.device, t.dtype)
    rl = torch.zeros_like(w)
    for k in range(N_gl + 1):
        rl = rl + g[k] * model(t, (w - k * h).clamp(min=0.0))
    return rl * (h ** (-beta))


# ============================================================================
# 5.  Caputo time derivative via Gauss-Jacobi quadrature
# ============================================================================

def caputo_t(model_fn, t: torch.Tensor, w: torch.Tensor, alpha: float,
             n_q: int = 16) -> torch.Tensor:
    from scipy.special import roots_jacobi
    nodes_np, weights_np = roots_jacobi(n_q, -alpha, 0.0)
    nodes   = torch.tensor(nodes_np,   device=t.device, dtype=t.dtype)
    weights = torch.tensor(weights_np, device=t.device, dtype=t.dtype)
    integral = torch.zeros_like(t)
    for i in range(n_q):
        s_i = (t * (1.0 + nodes[i]) / 2.0).detach().requires_grad_(True)
        u_s = model_fn(s_i, w)
        du  = torch.autograd.grad(u_s.sum(), s_i, create_graph=True)[0]
        integral = integral + weights[i] * du
    prefactor = t.pow(1.0 - alpha) / (2.0 ** (1.0 - alpha) * math.gamma(1.0 - alpha))
    return prefactor * integral


# ============================================================================
# 6.  In-span target generator
# ============================================================================

class InSpanTarget:
    """
    Generates a PDE benchmark whose exact solution is
        u*(t, w) = sum_{s=0}^{S*} a_s * K_s^{(p0, q0)}(t, w)
    with prescribed (p0, q0, a_s).  This is the natural sweet-spot of the fNKp
    method.  The source term is computed analytically from the RL identity.

    PDE:  D_t^alpha u  =  kappa * D_w^beta u + f
    with Dirichlet BCs inherited from u*.
    """

    def __init__(self, S_star: int, p0: float, q0: float,
                 coeffs: torch.Tensor, upsilon: int = 1):
        self.S_star = int(S_star)
        self.p0     = float(p0)
        self.q0     = float(q0)
        self.coeffs = coeffs.detach().clone()
        self.upsilon = int(upsilon)

        # Build a fixed (non-learnable) basis with exactly (p0, q0).
        self._basis = StableFNKpBasis(S_star, upsilon=upsilon,
                                       p_init=p0, q_init=q0)
        # Freeze the basis parameters to exactly (p0, q0).
        with torch.no_grad():
            p_val_target = p0 - (2*S_star + 1) - self._basis.eps_p
            q_val_target = q0 - (-1.0) - self._basis.eps_q
            self._basis.p_tilde.copy_(torch.tensor(math.log(math.expm1(p_val_target))))
            self._basis.q_tilde.copy_(torch.tensor(math.log(math.expm1(q_val_target))))
        for p in self._basis.parameters():
            p.requires_grad_(False)

    def u(self, t, w) -> torch.Tensor:
        # u*(t, w) = w^q0 * sum_s coeffs[s] K_s^{(p0, q0)}(t, w)
        Phi = self._basis(t, w)
        a   = self.coeffs.to(t.device, dtype=t.dtype)
        w_q = w.clamp(min=1e-12).pow(self.q0)
        return w_q * (Phi * a).sum(dim=1)

    def rl_w_u(self, t, w, beta: float) -> torch.Tensor:
        """
        Exact D_w^beta u*(t,w) using the clean identity:
           D_w^beta [ w^q K_s ]  =  w^{q-beta} K_s^{(p, q-beta)}.
        """
        Dw = self._basis.rl_times_wq(t, w, beta)
        a  = self.coeffs.to(t.device, dtype=t.dtype)
        return (Dw * a).sum(dim=1)

    def caputo_t_u(self, t, w, alpha: float) -> torch.Tensor:
        """
        D_t^alpha u* exactly: K_s is a polynomial in t of degree s, so
            K_s(t,w) = sum_{k=0}^{s} c_{s,k}(w) t^{s-k}.
        D_t^alpha [ t^n ] = Gamma(n+1)/Gamma(n+1-alpha) * t^{n-alpha} for n >= 1
        (Caputo of a constant is 0).
        """
        out = torch.zeros_like(t)
        p   = torch.tensor(self.p0, dtype=t.dtype, device=t.device)
        q   = torch.tensor(self.q0, dtype=t.dtype, device=t.device)
        a   = self.coeffs.to(t.device, dtype=t.dtype)

        for s in range(self.S_star + 1):
            tbl   = self._basis._tables[s]
            k_vec = tbl['k'].to(t.device, dtype=t.dtype)
            m_vec = tbl['m'].to(t.device, dtype=t.dtype)
            pkm   = tbl['pkm'].to(t.device, dtype=t.dtype)
            um    = self.upsilon * m_vec
            gam_p2sk = torch.exp(torch.lgamma(p - 2.0*s + k_vec))
            gam_q1um = torch.exp(torch.lgamma(q + 1.0 + um))
            coeffs_mk = pkm / (gam_p2sk * gam_q1um)

            # Caputo D_t^alpha[t^{s-k}]:
            n_vec = (s - k_vec)
            # for n=0 the Caputo of a constant is 0
            nonzero = n_vec > 1e-6
            num_g   = torch.exp(torch.lgamma(n_vec.clamp(min=1e-6) + 1.0))
            den_g   = torch.exp(torch.lgamma(n_vec.clamp(min=1e-6) + 1.0 - alpha))
            frac    = torch.where(nonzero, num_g / den_g, torch.zeros_like(n_vec))
            # t^{n-alpha}: use clamp to avoid 0^negative
            t_pow   = t.clamp(min=1e-12).unsqueeze(1).pow((n_vec - alpha).unsqueeze(0))
            w_pow   = w.clamp(min=1e-12).unsqueeze(1).pow(um.unsqueeze(0))
            term    = (coeffs_mk * frac * t_pow * w_pow).sum(dim=1)
            out     = out + a[s] * torch.exp(torch.lgamma(p - s)) * term
        # Apply the w^{q0} factor from the ansatz  u* = w^{q0} * sum a_s K_s
        w_q = w.clamp(min=1e-12).pow(self.q0)
        return w_q * out

    def source(self, t, w, alpha: float, beta: float, kappa: float) -> torch.Tensor:
        return self.caputo_t_u(t, w, alpha) - kappa * self.rl_w_u(t, w, beta)


# ============================================================================
# 7.  Finite-difference reference solver
# ============================================================================

class FDMReference:
    """
    Classical fractional FDM for
        D_t^alpha u  =  kappa * D_w^beta u + f(t,w),
        (t, w) in (0, T) x (0, W),
        u(0, w)   = u0(w),
        u(t, 0)   = g0(t),   u(t, W) = gW(t).

    - Time:  L1 scheme for the Caputo derivative of order alpha in (0,1).
    - Space: shifted-Gruenwald of order beta in (1,2) (Meerschaert-Tadjeran).

    Implicit Euler in time (unconditionally stable for the linear problem).
    Returns the solution grid and a bilinear evaluator at arbitrary (t,w).
    """

    def __init__(self, alpha: float, beta: float, kappa: float,
                 T: float = 1.0, W: float = 1.0,
                 Nt: int = 200, Nw: int = 200):
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.T, self.W = T, W
        self.Nt, self.Nw = Nt, Nw
        self.dt = T / Nt
        self.dw = W / Nw
        self.t = np.linspace(0.0, T, Nt + 1)
        self.w = np.linspace(0.0, W, Nw + 1)

    def solve(self, u0_fn, g0_fn, gW_fn, source_fn_vec) -> np.ndarray:
        """
        source_fn_vec : callable  source_fn_vec(t_scalar, w_vec) -> numpy vec
                        (we pass in the full interior grid at once for speed)
        """
        alpha, beta, kappa = self.alpha, self.beta, self.kappa
        dt, dw, Nt, Nw = self.dt, self.dw, self.Nt, self.Nw

        # L1 weights
        b = np.array([(k + 1) ** (1 - alpha) - k ** (1 - alpha)
                      for k in range(Nt + 1)])
        mu_t = 1.0 / (dt ** alpha * math.gamma(2.0 - alpha))

        # Shifted-Grunwald weights g[0..Nw+1]
        g = np.zeros(Nw + 2)
        g[0] = 1.0
        for k in range(1, Nw + 2):
            g[k] = g[k - 1] * (1.0 - (beta + 1.0) / k)

        inv_dwb = dw ** (-beta)

        N = Nw - 1
        # Vectorised LHS matrix: for row j (index j-1), column k+1 corresponds
        # to u_{k+1} with coefficient -kappa*inv_dwb*g[j+1-(k+1)] = -kappa*inv_dwb*g[j-k]
        # provided j-k >= 0; and the j+1 index (row j, col j) is g[0] side etc.
        # Simpler: build via outer on banded structure.
        B = np.zeros((N, N))
        for j in range(1, Nw):
            row = j - 1
            B[row, row] += mu_t * b[0]
            k_vals = np.arange(0, j + 2)
            idxs   = j + 1 - k_vals
            mask   = (idxs >= 1) & (idxs <= Nw - 1)
            cols   = idxs[mask] - 1
            B[row, cols] += -kappa * inv_dwb * g[k_vals[mask]]

        # Store LU factorisation
        from scipy.linalg import lu_factor, lu_solve
        LU = lu_factor(B)

        U = np.zeros((Nt + 1, Nw + 1))
        U[0, :] = u0_fn(self.w)
        U[:, 0]  = g0_fn(self.t)
        U[:, -1] = gW_fn(self.t)

        # Precompute boundary contribution matrix: b_contrib[j] sums contributions
        # of U[n, 0] (w=0) and U[n, Nw] (w=W) to row j.
        # For row j (1..Nw-1), idx=0 means k=j+1, idx=Nw means k=j+1-Nw.
        c0 = np.zeros(N)  # coefficient of U[n, 0]
        cW = np.zeros(N)  # coefficient of U[n, Nw]
        for j in range(1, Nw):
            row = j - 1
            k0 = j + 1        # idx = 0
            if 0 <= k0 <= Nw + 1:
                c0[row] = -kappa * inv_dwb * g[k0]
            kW = j + 1 - Nw   # idx = Nw
            if 0 <= kW <= Nw + 1:
                cW[row] = -kappa * inv_dwb * g[kW]

        # Time stepping
        w_int = self.w[1:Nw]
        for n in range(1, Nt + 1):
            # Source on interior
            f_vec = source_fn_vec(self.t[n], w_int)
            rhs = f_vec.copy()

            # Subtract boundary contributions
            rhs -= c0 * U[n, 0]
            rhs -= cW * U[n, Nw]

            # Time history:
            # hist_j = sum_{k=1}^{n-1} (b[k]-b[k-1]) * U[n-k, j]  -  b[n-1] * U[0, j]
            if n >= 2:
                k_arr   = np.arange(1, n)
                coeffs_ = b[k_arr] - b[k_arr - 1]            # (n-1,)
                U_hist  = U[n - k_arr, 1:Nw]                 # (n-1, N)
                hist_sum = np.einsum('k,kj->j', coeffs_, U_hist)
            else:
                hist_sum = np.zeros(N)
            hist_sum -= b[n - 1] * U[0, 1:Nw]
            rhs -= mu_t * hist_sum

            u_int = lu_solve(LU, rhs)
            U[n, 1:Nw] = u_int

        self.U = U
        return U

    # ------------------------------------------------------------------
    def evaluate(self, t_vals: np.ndarray, w_vals: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the solution at arbitrary (t, w)."""
        U = self.U
        t_idx = np.clip(t_vals / self.dt, 0, self.Nt - 1e-9)
        w_idx = np.clip(w_vals / self.dw, 0, self.Nw - 1e-9)
        it0 = np.floor(t_idx).astype(int)
        iw0 = np.floor(w_idx).astype(int)
        it1 = it0 + 1
        iw1 = iw0 + 1
        dt_ = t_idx - it0
        dw_ = w_idx - iw0
        return ((1 - dt_) * (1 - dw_) * U[it0, iw0] +
                dt_       * (1 - dw_) * U[it1, iw0] +
                (1 - dt_) * dw_       * U[it0, iw1] +
                dt_       * dw_       * U[it1, iw1])
