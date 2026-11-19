"""Spatial grid-convergence study for the MOL column model.

Solves the steady cell balances on N = 20/40/80/160/320 cells. The steady state
is a root of the same RHS the transient integrator marches, so the two routes
solve the same equations; the steady route only skips the 4000 s of physical
time needed to relax onto the fixed point.

Solving route:
  1. one transient march at N=20, used only to seed the Newton solver (the flat
     initial condition is outside its basin of attraction);
  2. sparse Newton with a coloured finite-difference Jacobian on the correct
     block-pentadiagonal pattern (the minmod face value at f reads cells f-2..f,
     so a tridiagonal pattern is wrong and silently stalls), then continuation
     20->40->80->160->320 with profiles interpolated between grids.

Reported per grid: capture, RHS residual, the two independent mass closures
(gas-side drop, solvent loading swing) as relative residuals, and the bottom
loading against the bottom equilibrium loading.

The swing residual is an ALGEBRAIC IDENTITY of the cell-centred discretisation
-- every cell holds both inventories and both see rate_i*dz -- so at an exact
root it is zero. What the study reports is the root solve's error, which is why
the assertion is against max(1e-10, 10*|rhs|inf) rather than a flat number.
"""

import math
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from rate_based_column import (simulate_column_mol, solve_steady_state,
                              steady_state_report)

NS = (20, 40, 80, 160, 320)
CRITERION_PP = 0.2


def _state(res, n):
    """Pack the final transient profile as [C_G, alpha, T_gas, T_liq] per cell."""
    out = []
    for i in range(n):
        out += [res['C_G_CO2'][-1][i], res['alpha'][-1][i],
                res['T_gas'][-1][i], res['T_liquid'][-1][i]]
    return np.asarray(out, dtype=float)


def _interp(y_old, n_old, n_new):
    """Interpolate a cell-centred profile onto a finer cell-centred grid."""
    z_old = (np.arange(n_old) + 0.5) / n_old
    z_new = (np.arange(n_new) + 0.5) / n_new
    y_new = np.zeros(4 * n_new)
    for v in range(4):
        y_new[v::4] = np.interp(z_new, z_old, y_old[v::4])
    return y_new


def main():
    t_start = time.time()

    # --- seed: one transient march at the coarsest grid ----------------------
    t0 = time.time()
    _, res = simulate_column_mol(duration=4000.0, dt=2.0, n_zones=20,
                                 disturbance=None, disturbance_magnitude=0.0)
    seed = _state(res, 20)
    print(f"seed  transient N=20 t=4000s  capture={res['capture_efficiency_pct']:.4f}%"
          f"  ({time.time() - t0:.0f} s)", flush=True)

    caps, reports, residuals = {}, {}, {}
    y_prev, n_prev = seed, 20

    for N in NS:
        if N != 20:
            y_guess = _interp(y_prev, n_prev, N)
        else:
            y_guess = seed
        t0 = time.time()
        y, res_inf, cfg, rp = solve_steady_state(N, y0=y_guess)
        rep = steady_state_report(y, N, cfg, rp)
        caps[N], reports[N], residuals[N] = rep['capture_pct'], rep, res_inf
        y_prev, n_prev = y, N
        print(f"N={N:4d} capture={rep['capture_pct']:.4f}%  |rhs|inf={res_inf:.2e}"
              f"  swing={rep['swing_rel']:.2e}  gas={rep['n_gas_rel']:.2e}"
              f"  a_b={rep['alpha_bottom']:.6f} a_t={rep['alpha_top']:.6f}"
              f"  a*={rep['alpha_star_bottom']:.6f}"
              f"  n={rep['n_flux']:.5f} mol/s  ({time.time() - t0:.1f} s)",
              flush=True)

    # --- convergence ---------------------------------------------------------
    print("\n--- successive differences ---")
    delta = {}
    for a, b in zip(NS[:-1], NS[1:]):
        d = abs(caps[a] - caps[b])
        delta[(a, b)] = d
        print(f"  {a:4d} -> {b:4d}: {d:.6f} pp")

    orders = []
    for k in range(len(NS) - 2):
        a, b, c = NS[k], NS[k + 1], NS[k + 2]
        d1, d2 = delta[(a, b)], delta[(b, c)]
        if d1 > 0 and d2 > 0:
            p = math.log(d1 / d2) / math.log(2.0)
            print(f"  order over {a}->{b}->{c}: p = {p:.3f}")
        else:
            p = float('nan')
        orders.append(p)

    d_last = delta[(NS[-2], NS[-1])]
    d_prev = delta[(NS[-3], NS[-2])]
    p_last = math.log(d_prev / d_last) / math.log(2.0) if d_last > 0 else float('inf')
    if p_last > 0:
        rich = caps[NS[-1]] + (caps[NS[-1]] - caps[NS[-2]]) / (2.0 ** p_last - 1.0)
        print(f"\n  last-pair order p = {p_last:.3f}")
        print(f"  Richardson (N={NS[-1]}, p={p_last:.2f}): capture_inf = {rich:.4f} %")
    else:
        rich = float('nan')
        print(f"\n  last-pair order p = {p_last:.3f} (no Richardson estimate)")

    shrinking = all(delta[(NS[k], NS[k + 1])] > delta[(NS[k + 1], NS[k + 2])]
                    for k in range(len(NS) - 2))
    print(f"  increments shrinking monotonically: {shrinking}")
    passed = bool(d_last < CRITERION_PP and 1.0 <= p_last <= 2.5 and shrinking)
    print(f"  criterion delta < {CRITERION_PP} pp with p ~ 1-2: "
          f"{'PASS' if passed else 'FAIL'}")

    # --- identity assertions -------------------------------------------------
    # On a cell-centred grid the gas drop, the solvent loading swing and the cell
    # integral A*dz*sum(rate_i) are three evaluations of the same algebraic sum,
    # so at a root they agree exactly. What is left is the root solve's own
    # error: an iterate with |rhs|inf = e leaves each closure off by roughly e
    # times the cell inventory scale, never below it. The check is therefore
    # against the achieved residual as well as a flat floor -- a flat 1e-10 that
    # fires on a solver-limited grid would report a leak that is not there.
    print("\n--- mass closures (identity; residual is the solver's, not the scheme's) ---")
    worst_swing = max(reports[N]['swing_rel'] for N in NS)
    worst_gas = max(reports[N]['n_gas_rel'] for N in NS)
    worst_res = max(residuals[N] for N in NS)
    ok = True
    for N in NS:
        sw, ga, rr = (reports[N]['swing_rel'], reports[N]['n_gas_rel'], residuals[N])
        bound = max(1e-10, 10.0 * rr)
        good = max(sw, ga) <= bound
        ok = ok and good
        print(f"  N={N:4d} swing={sw:.3e}  gas={ga:.3e}  rhs={rr:.3e}"
              f"  bound={bound:.1e}  {'ok' if good else 'EXCEEDED'}")
    print(f"  worst swing={worst_swing:.3e}  worst gas={worst_gas:.3e}"
          f"  worst rhs={worst_res:.3e}")
    print(f"  identity holds at every N (<= max(1e-10, 10*rhs)): "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  flat 1e-10 test, for reference: "
          f"{'PASS' if max(worst_swing, worst_gas) < 1e-10 else 'FAIL'}")

    # --- thermodynamic ceiling ----------------------------------------------
    rep = reports[NS[-1]]
    n_MEA, a_lean = rep['n_MEA'], rep['alpha_lean']
    da_capacity = rep['alpha_star_bottom'] - a_lean
    n_capacity = n_MEA * da_capacity
    n_gas_in = rep['n_flux'] / (rep['capture_pct'] / 100.0)
    print("\n--- thermodynamic ceiling at the current solvent rate ---")
    print(f"  alpha_bottom = {rep['alpha_bottom']:.5f}"
          f"   alpha_star(P_CO2={rep['P_CO2_bottom']:.0f} Pa, T={rep['T_bottom']:.1f} C)"
          f" = {rep['alpha_star_bottom']:.5f}")
    print(f"  alpha_bottom / alpha_star = {rep['alpha_bottom'] / rep['alpha_star_bottom']:.4f}"
          f"  ({100 * rep['alpha_bottom'] / rep['alpha_star_bottom']:.2f} % of equilibrium)")
    print(f"  available swing = {da_capacity:.5f} mol CO2/mol MEA"
          f"  ->  n_max = {n_capacity:.4f} mol/s of {n_gas_in:.4f} in"
          f"  =  {100 * n_capacity / n_gas_in:.2f} % capture")
    target = 0.90
    n_req = target * n_gas_in
    print(f"  90 % needs n = {n_req:.4f} mol/s; capacity at m_solvent=0.7697 kg/s"
          f" supplies {n_capacity:.4f} mol/s")
    if da_capacity > 0:
        print(f"  => required solvent rate: m_solvent >="
              f" {0.7697 * (n_req / n_capacity):.4f} kg/s"
              f"  (+{100 * (n_req / n_capacity - 1.0):.1f} %)")
        print(f"  => or required lean loading: alpha_lean <="
              f" {rep['alpha_star_bottom'] - n_req / n_MEA:.4f}"
              f"  (currently {a_lean:.3f})")

    print(f"\ntotal wall time {time.time() - t_start:.0f} s")
    return caps, reports, passed, p_last, rich


if __name__ == '__main__':
    main()
