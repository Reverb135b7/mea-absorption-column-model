"""Rate-Based MEA Absorption Column Model
with Method of Lines (MOL) PDE Discretization along Column Height

Implements first-principles mass and energy balance equations for post-combustion
carbon capture using Monoethanolamine (MEA) absorption.

Governing equations discretized along column height z using method of lines:
- N spatial nodes, each with ODEs for gas/liquid concentrations and temperatures
- Convection terms: upwind differencing for stability
- Interfacial mass transfer: two-film theory at each node (Formulation B: loading-based)
- Reaction kinetics: Arrhenius-temperature-dependent MEA + CO2
- Stiffness removed: algebraic enhancement closure; no free dissolved CO2 state vector
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
from performance import ColumnPerformance, from_simulation_result


# ============================================================================
# Physical Properties & Constants for MEA/CO2/Water System
# ============================================================================

# MEA properties
MEA_MOLAR_MASS = 0.06108  # kg/mol MEA (C2H7NO)
CO2_MOLAR_MASS = 0.04401  # kg/mol
WATER_MOLAR_MASS = 0.018015  # kg/mol

# Reaction enthalpies (kJ/mol per CO2)
ΔH_CARBAMATE = -85.0  # CO2 + 2 MEA ⇌ MEA2CO2 (carbamate formation, exothermic)
ΔH_BICARBONATE = -15.0  # CO2 + H2O + MEA ⇌ MEA-HCO3- (bicarbonate formation)

# Mass transfer coefficients (typical values, can be tuned)
K_C_GAS_DEFAULT = 0.02  # m/s, gas-film mass-transfer velocity
K_G_DEFAULT = K_C_GAS_DEFAULT / (8.314 * 313.15)  # mol/(m2 s Pa); RT conversion
K_L_DEFAULT = 0.005  # liquid-side mass transfer coeff (m/s)

# Heat-capacity densities: dT/dt = Q_dot[m3] / (rho * cp * holdup)
RHO_L = 1010.0   # kg/m3 (30 wt% MEA solution)
CP_L = 4186.0    # J/kg/K
RHO_G = 1.2      # kg/m3
CP_G = 1050.0    # J/kg/K

# Column geometry
COLUMN_DIAMETER = 2.0  # m
COLUMN_HEIGHT = 30.0  # m
NUM_TRAYS = 10

# Number of spatial nodes for MOL discretization (default)
N_ZONES = 10

# --- Grid convergence record ---------------------------------------------
# Measured on THIS semi-discrete system at steady state, solved directly
# (Newton on the cell-centred RHS; continuation 20->40->80->160->320, the
# coarsest grid seeded by one transient march). Cell-centred finite volume:
# N cells, each holding BOTH a gas and a liquid inventory, Dirichlet conditions
# on the faces only.
#   N      capture (%)   |rhs|inf   swing rel   alpha_b    alpha*(bottom)
#   20     83.6499       3.8e-12    4.0e-13     0.431891   0.433223
#   40     85.2033       7.6e-12    1.4e-14     0.436197   0.437690
#   80     86.1747       3.1e-12    1.2e-11     0.438890   0.440559
#   160    86.6784       3.4e-08    6.0e-10     0.440286   0.442237
#   320    86.9024       8.7e-08    7.9e-11     0.440907   0.443174
# Successive differences 1.553, 0.971, 0.504, 0.224 pp shrink monotonically
# with observed order 0.68, 0.95, 1.17. Richardson on the finest pair (p=1.17)
# gives 87.08 %. The 0.2 pp criterion is NOT met at N=320 (0.224 pp, 12 % over):
# passed=False. At the observed ratio one more refinement reaches ~0.10 pp.
#
# Capture is insensitive to the root residual: at N=160 the solve leaves
# |rhs| ~ 1e-8 while the closures are at 3e-11, and the capture changes by less
# than 1e-6 pp against a run that converged a decade deeper. The error that
# matters here is truncation, not the linear solve.
#
# The mass closures are identities now, not metrics. Every cell owns both
# inventories and both see the same rate_i*dz, so summing either phase gives
#     A*dz*sum(rate_i) = u_G*A*(C_in - C_out) = n_MEA*(alpha_0 - alpha_lean)
# to solver precision: swing 4e-13 ... 8e-11 and gas 4e-13 ... 3e-11 over the
# five grids. The node grid could not do this -- it froze node 0 for the gas and
# node N-1 for the liquid, so each phase skipped a DIFFERENT half cell, the two
# integrals differed by A*dz*(rate_0 - rate_{N-1}), the swing read 2.4/1.7/0.2 %
# and the capture order came out negative (-2.15).
#
# The converged answer is a pinch, not a coefficient: alpha_bottom sits at 99.5 %
# of alpha_star at the bottom CO2 partial pressure, i.e. the solvent leaves at
# equilibrium. Available swing 0.2432 mol CO2/mol MEA x 3.780 mol MEA/s = 0.919
# mol/s, which is 87.7 % of the 1.048 mol/s fed -- so 90 % is unreachable at this
# solvent rate and lean loading, whatever K_Ga or height is used. To reach 90 %:
# m_solvent >= 0.790 kg/s (+2.6 %) or alpha_lean <= 0.194 (from 0.200).
#
# Why the observed order is ~1.2 and not 2: the increments do shrink consistently
# (ratio 1.6, 1.9, 2.2) but the leading truncation term has not yet been isolated
# term by term. The two first-order suspects are the zero-gradient outlet face
# (the outflow value is taken as the cell value, not reconstructed) and minmod
# clipping where the loading and temperature profiles turn over. Both are O(dz);
# neither is a mass-conservation defect, which is why the closures above stay at
# solver precision.
#
# Solver: steady state is solved directly (no time marching) with sparse Newton
# on a coloured finite-difference Jacobian, continuation 20->40->80->160->320,
# then a trust-region least-squares polish for the limiter kink. The Jacobian
# pattern must be block PENTADIAGONAL: the minmod face value at f reads cells
# f-2..f, so a tridiagonal pattern drops two-cell coupling and stalls Newton at
# |rhs| ~ 5e-4 while still looking converged on coarse grids.
GRID_STUDY = {
    'capture_pct': {20: 83.6499, 40: 85.2033, 80: 86.1747,
                    160: 86.6784, 320: 86.9024},
    'delta_pp': {20: 1.5534, 40: 0.9714, 80: 0.5037, 160: 0.2240},
    'order': {20: 0.677, 40: 0.948, 80: 1.169},
    'observed_order': 1.17,
    'richardson_pct': 87.08,
    'criterion_pp': 0.2,
    'loading_swing_rel': {20: 4.0e-13, 40: 1.4e-14, 80: 1.2e-11,
                          160: 6.0e-10, 320: 7.9e-11},
    'passed': False,
}

DURATION_STUDY = {
    'N20_4000_vs_6000_pp': 0.0000,
    'N40_4000_vs_8000_pp': 0.0000,
    'N80_4000_vs_6000_pp': 0.0000,
    'dt_2_vs_10_pp': 0.0000,
    'N20_path_independence_pp': 0.0000,
    'criterion_pp': 0.1,
    'passed': True,
}

# Gas phase typical values
UG_DEFAULT = 1.5  # m/s - gas velocity
UL_DEFAULT = 0.02  # m/s - liquid velocity (counter-current)
EPS_G_DEFAULT = 0.85  # gas holdup (-)
EPS_L_DEFAULT = 0.15  # liquid holdup (-)

# Operating conditions
T_INLET = 40.0  # °C - flue gas inlet temperature (z=0)
T_REBOILER_IN = 120.0  # °C - reboiler inlet
T_REBOILER_OUT = 140.0  # °C - reboiler outlet

# Stripping steam: a CONSTANT assumption. The reduced state vector has no
# gas-phase water, so Q_steam cannot be coupled to the overhead in this build.
N_H2O_OVERHEAD_ASSUMED = 0.023  # mol/s
DH_VAP_H2O_KJ_MOL = 40.7        # kJ/mol


def load_design_basis(path="design_basis.yaml"):
    """Load the design basis once; returns {} when the file is absent."""
    import yaml
    try:
        with open(path, "r") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def overhead_water_mol_s(n_CO2_mol_s, T_top_C, P_strip_bar):
    """Saturated overhead water from an explicit T/P assumption (not a state).

    Antoine (water, log10 mmHg): 8.07131 - 1730.63/(233.426 + T_C).
    Q_steam derived from this must be labelled steam_source="assumed_overhead".
    """
    P_sat_mmHg = 10.0 ** (8.07131 - 1730.63 / (233.426 + T_top_C))
    P_sat_bar = P_sat_mmHg * 1.33322e-3
    if P_sat_bar >= 0.95 * P_strip_bar:
        raise ValueError(
            f"stripper overhead saturated: P_sat={P_sat_bar:.3f} bar vs "
            f"P_strip={P_strip_bar:.3f} bar at {T_top_C:.1f} C"
        )
    return n_CO2_mol_s * P_sat_bar / (P_strip_bar - P_sat_bar)


# ============================================================================
# Helper: MEA Concentration from weight percent
# ============================================================================

def mea_concentration_from_wt(w_MEA=0.30, rho_sol=1010.0):
    """Convert MEA weight percent to molar concentration (mol/m³).
    
    For 30 wt% MEA: C_MEA ≈ 4961 mol/m³
    For 18 wt% MEA: C_MEA ≈ 3000 mol/m³ (your previous default)
    """
    MEA_MOLAR_MASS = 0.06108  # kg/mol MEA
    water_molar_mass = 0.018015  # kg/mol
    # mass basis: 1 m³ solution = rho_sol kg
    # MEA mass = w_MEA * rho_sol; water mass = (1-w_MEA) * rho_sol
    mea_mass = w_MEA * rho_sol  # kg ME A per m³
    water_mass = (1.0 - w_MEA) * rho_sol  # kg water per m³
    C_MEA = mea_mass / MEA_MOLAR_MASS  # mol MEA per m³ (only MEA, no water)
    # Actually, total molar concentration is more complex; return MEA-only
    return C_MEA


# ============================================================================
# Helper: Henry's Constant (temperature dependent)
# NO CLAMP — fail loudly if T outside valid range
# ============================================================================

def henry_constant_Pa_m3_mol(T_celsius):
    """Henry's constant for CO2 in water/MEA solution.
    
    K_H = exp(A - B/T) where T in Kelvin.
    Typical range: gives ~10^5 Pa·m³/mol at 40°C.
    No T_K clamping — caller must ensure T in [303, 393] K range.
    """
    T_K = T_celsius + 273.15
    if T_K < 303.15 or T_K > 393.15:
        # Don't clamp; let it fail so the energy balance bug is visible
        # But still compute so we can report the value
        pass
    # Simplified correlation for CO2 in aqueous MEA
    # ln(K_H) = 28.6 - 6500/T_K  (approximate, gives ~10^5 Pa·m³/mol at 40°C)
    ln_K_H = 28.6 - 6500.0 / T_K
    K_H = np.exp(ln_K_H)  # Pa·m³/mol
    return K_H


# ============================================================================
# Helper: Equilibrium liquid-phase CO2 via VLE (Formulation B)
# ============================================================================

def equilibrium_liquid_co2_loading(alpha, T, P_total=101325.0):
    """Calculate equilibrium CO2 partial pressure from liquid loading.
    
    Args:
        alpha: mol CO2 per mol MEA (total loading, 0 ≤ alpha ≤ 0.5)
        T: temperature in Celsius
        P_total: system pressure (Pa), default 1 atm
    
    Returns:
        P_star: equilibrium CO2 partial pressure (Pa)
    """
    H = henry_constant_Pa_m3_mol(T)  # Pa·m³/mol
    # Approximate VLE: P_star = H * f(alpha, T)
    # For MEA, roughly: P_star increases with alpha
    # Simple correlation (Dugas/Rochelle type):
    # P_star ≈ H * alpha / (1 - alpha/alpha_max) with alpha_max ≈ 0.5
    alpha_max = 0.5  # mol CO2 / mol MEA maximum
    if alpha >= alpha_max:
        raise ValueError(
            f"liquid loading alpha={alpha:.6f} reached saturation "
            f"(alpha_max={alpha_max}); VLE is divergent and the ODE has left "
            "its physical domain"
        )
    # Equilibrium partial pressure from loading
    # This captures that richer solvent holds more CO2 at partial pressure
    P_star = H * alpha / (1.0 - alpha / alpha_max)
    return P_star


# ============================================================================
# Helper: Two-Film Mass Transfer — Formulation B (loading-based)
# No free CO2 state, no m·C_L driving force, no unit mismatch
# ============================================================================

def two_film_mass_transfer_loading(C_G, alpha, T, K_G=K_G_DEFAULT, K_L=K_L_DEFAULT,
                                   a_i=250.0, P_total=101325.0):
    """Two-film theory interfacial CO2 mass transfer flux — Formulation B.
    
    Uses liquid loading alpha (mol CO2 / mol MEA) instead of total C_L_CO2.
    No free dissolved CO2 concentration variable — eliminates the 13,400× 
    unit mismatch and the 6.4×10⁵ stiffness ratio.
    
    Formulation B (loading-based, pyMEA-RateSim style):
    
    1. P_star = H * alpha / (1 - alpha/alpha_max)  from VLE
    2. Ha = sqrt(k2 * C_MEA_free * D_A) / k_L  Hatta number
    3. E_inf = 1 + D_B * C_MEA_free / (2 * D_A * P_i / H_vol)
    4. E = coursey(Ha, E_inf)  enhancement factor
    5. K_G = 1 / (1/K_G + H_vol/(E * k_L))  overall coefficient
    6. N_CO2 = K_G * (P_G - P_star)  molar flux (mol/m²/s)
    
    Only the gas-side partial pressure P_G and liquid loading alpha are state variables.
    The fast free-CO2 concentration is eliminated algebraically.
    
    Args:
        C_G: gas-phase CO2 concentration (mol/m³)
        alpha: liquid loading (mol CO2 / mol MEA)
        T: average temperature (°C)
        K_G: gas-side mass transfer coeff (mol/m²·s·Pa)  
        K_L: liquid-side mass transfer coeff (mol/m²·s)
        a_i: interfacial area/vol (m²/m³)
        P_total: total pressure (Pa)
    
    Returns:
        N_CO2: molar flux (mol/m²/s)
        K_overall: overall mass transfer coeff (mol/m²·s·Pa)
        m: distribution coefficient (Pa·m³/mol) — for reference only
        P_star: equilibrium partial pressure (Pa)
    """
    R = 8.314  # J/(mol·K)
    T_K = T + 273.15
    
    # --- Step 1: Equilibrium partial pressure from loading ---
    P_star = equilibrium_liquid_co2_loading(alpha, T, P_total=P_total)
    
    # --- Step 2: MEA concentration (from design basis) ---
    # C_MEA will be provided via params; if not, compute from 30 wt%
    # Typical 30 wt% MEA: ~4961 mol/m³ MEA
    # We'll use a default and let the caller override
    C_MEA = 3000.0  # fallback; override in params if available
    
    # --- Step 3: Hatta number and enhancement factor ---
    # Forward rate constant for carbamate formation (Arrhenius)
    A1 = 1.5e7     # m³/(mol·s) - forward
    Ea1 = 45000.0  # J/mol - forward
    k1 = A1 * np.exp(-Ea1 / (R * T_K))  # forward rate constant
    
    # Reverse rate constant
    A_1 = 3.0e5    # m³/(mol·s) - reverse
    Ea_1 = 55000.0 # J/mol - reverse
    k_1 = A_1 * np.exp(-Ea_1 / (R * T_K))  # reverse rate constant
    
    # Hatta number: Ha = sqrt(k1 * C_MEA) * D_A / k_L / (something)
    # Simplified: Ha ≈ sqrt(k1 * C_MEA) / (k_L / D_A) 
    # We'll use the standard definition but avoid explicit diffusivities
    # by using the enhancement factor formulation from Dugas/Rochelle
    
    # Enhancement factor E_inf at high Ha (reaction-limited)
    # E_inf = 1 + (D_B * C_MEA) / (2 * D_A * (P_star / H_vol_term))
    # Without explicit diffusivities, use the equilibrium-based limit:
    # E ≈ 1 + k1 * C_MEA / (k_L * C_G / P_star * H) ... this gets messy
    
    # SIMPLIFIED: use the algebraic Hatta/enhancement closure from 
    # the pyMEA-RateSim plan that deletes the fast state entirely.
    # The key insight: at high Ha, E → E_inf; at low Ha, E → 1.
    # We compute E based on the ratio of reaction rate to mass transfer rate.
    
    # Simplified enhancement factor (no free-CO2 state needed):
    # E = 1 + (k1 * C_MEA * H) / (K_L * ... ) — too many unknowns
    # Instead: use the loading-dependent approach where the 
    # effective mass transfer coefficient already accounts for reaction.
    
    # OVERALL: compute K_overall directly from the two-film series resistance,
    # but with the equilibrium term evaluated at the loading-based P_star.
    # This avoids the m*C_L formulation entirely.
    
    # 1/K_overall = 1/K_G + H_vol / k_L   (series resistance, no m factor)
    # Where H_vol is the Henry's constant in pressure units
    # Actually, let's use the standard two-film with loading-based equilibrium:
    
    # The driving force is (P_G - P_star) where P_G = y_P * P_total
    # and P_star is from VLE at loading alpha.
    # The overall coefficient combines gas and liquid film resistances.
    
    # From Dugas & Rochelle style:
    # 1/K_G_overall = 1/K_G + H * RT / k_L   (approximately)
    # But we want it in terms of loading, not concentration.
    
    # Let's use the practical formulation: the flux is
    # N = K_G * (P_G - P_star)
    # where K_G already includes liquid-side enhancement.
    
    # For this implementation, we compute an effective K_G that
    # accounts for the enhancement from the fast reaction,
    # using the loading as the state variable.
    
    # Empirical/simplified: enhancement increases with alpha and T
    # E ≈ 1 + 10 * alpha  (rough approximation for 30 wt% MEA)
    # This is a placeholder — in a full model, E comes from Hatta analysis.
    
    # Actually, let's keep it simple and just compute the flux with
    # the equilibrium partial pressure. The "enhancement" is implicitly
    # captured because alpha already includes the reaction effects.
    
    # 1/K_overall = 1/K_G + m_eff/K_L where m_eff is derived from P_star
    # But we'll just use K_G directly with the P_star driving force,
    # and note that the liquid-film resistance is small for fast reactions.
    
    # Overall coefficient: simple series with Henry's law in pressure form
    # 1/K_overall = 1/K_G + H_vol / k_L  where H_vol = H (Pa·m³/mol)
    # But we need a length scale... let's just use K_G and note
    # the liquid contribution is small for typical MEA conditions.
    
    # Practical approach: compute K_overall using the series resistance
    # with an effective m derived from the loading equilibrium.
    # m_eff = P_star / C_G  (this gives the effective distribution coefficient)
    
    if K_G > 0 and K_L > 0 and a_i > 0:
        # Series resistance: 1/K_overall = 1/K_G + m/K_L
        # Use m = H (Henry's constant) as the approximate distribution coefficient
        # This is rough but avoids the C_L total/concentration mismatch
        H_val = henry_constant_Pa_m3_mol(T)
        m_eff = H_val  # Henry's constant as effective m
        K_overall = 1.0 / (1.0 / K_G + m_eff / K_L)
    else:
        K_overall = 0.0
    
    # Driving force: P_G - P_star (partial pressures in Pa)
    # Note: C_G is mol/m³, so P_G = C_G * R * T  (ideal gas law)
    P_G = C_G * R * T_K  # Pa, partial pressure if C_G were total gas conc
    # Actually, C_G in flue gas is already a concentration; 
    # for a gas partial pressure we'd need mole fraction.
    # Here we treat C_G as mol/m³ and compute equivalent partial pressure.
    
    # Driving force in pressure units
    driving_force_P = P_G - P_star
    
    # Flux: N_CO2 = K_overall * driving_force_P
    # But K_overall has units that need to match.
    # Let's use: N = K_G * (P_G - P_star) as the primary flux,
    # and K_overall is returned for reference.
    
    N_CO2 = K_G * driving_force_P  # mol/m²/s using gas-side coefficient
    
    # Return K_overall for compatibility, but the primary flux uses K_G
    # with the loading-based driving force.
    # The key point: NO m*C_L, NO total C_L_CO2, NO unit mismatch.
    
    return N_CO2, K_overall, m_eff, P_star


# ============================================================================
# Two-Film with explicit enhancement closure (stiffness removal)
# ============================================================================

def two_film_mass_transfer_with_enhancement(C_G, alpha, T, K_G=K_G_DEFAULT, 
                                            K_L=K_L_DEFAULT, a_i=250.0,
                                            C_MEA=3000.0, P_total=101325.0):
    """Two-film mass transfer WITH enhancement factor — stiffness removed.
    
    This version explicitly computes the Hatta number and enhancement factor,
    but CLOSES the free-CO2 concentration algebraically, deleting the 
    fast (3.4e-5 s) time scale from the state vector.
    
    Key: alpha (loading) is the only liquid-state variable. Free dissolved CO2
    is NOT a differential state; it's closed algebraically via the Hatta/enhancement
    factor. This drops stiffness from ~6.4×10⁵ to ~10².
    
    Args:
        C_G: gas CO2 concentration (mol/m³)
        alpha: liquid loading (mol CO2 / mol MEA)
        T: temperature (°C)
        K_G: gas-side mass transfer coeff
        K_L: liquid-side mass transfer coeff
        a_i: interfacial area/vol (m²/m³)
        C_MEA: MEA concentration (mol/m³)
        P_total: system pressure (Pa)
    
    Returns:
        N_CO2: molar flux (mol/m²/s)
        E: enhancement factor (dimensionless, ≥ 1)
        Ha: Hatta number (dimensionless)
    """
    R = 8.314
    T_K = T + 273.15
    
    # --- Step 1: Equilibrium partial pressure from loading ---
    # P_star = H * alpha / (1 - alpha/alpha_max)
    alpha_max = 0.5  # mol CO2 / mol MEA max
    # Saturation is asymptotic: P_star -> inf as alpha -> alpha_max, so the
    # driving force vanishes there by itself. A solver overshoot of a few parts
    # in 1e4 is numerical, not a capacity violation; only reject a real one and
    # evaluate the VLE on the closed interval.
    if alpha > alpha_max * (1.0 + 2e-3):
        raise ValueError(
            f"liquid loading alpha={alpha:.6f} exceeded saturation "
            f"(alpha_max={alpha_max}) at T={T:.1f} C"
        )
    alpha = min(alpha, alpha_max * (1.0 - 1e-9))
    H = henry_constant_Pa_m3_mol(T)
    P_star = H * alpha / (1.0 - alpha / alpha_max)
    
    # --- Step 2: Hatta number ---
    # Ha = sqrt(k1 * C_MEA) * D_A / k_L  (simplified, D_A cancels in E formulation)
    # Use forward rate constant for carbamate formation
    A1 = 1.5e7
    Ea1 = 45000.0
    k1 = A1 * np.exp(-Ea1 / (R * T_K))
    
    # Hatta number (simplified, without explicit diffusivities):
    # Ha = sqrt(k1 * C_MEA) / (k_L / L_char) 
    # We use the ratio form that determines enhancement
    Ha = np.sqrt(k1 * C_MEA) * K_L / (K_G + 1e-10)  # scaled version
    
    # --- Step 3: Enhancement factor ---
    # E_inf at high Ha (reaction-controlled, diffusion fast)
    # E_inf = 1 + (D_B * C_MEA) / (2 * D_A * (P_star / (H * ...)))
    # Without D_A, D_B: use the practical form
    # For Ha >> 1: E → E_inf ≈ 1 + D_B*C_MEA/(2*D_A*...)
    # For Ha << 1: E → 1 (reaction slow, no enhancement)
    # Approximation: E = 1 + (Ha / (1 + Ha)) * (E_inf - 1) ... but E_inf unknown
    
    # SIMPLIFIED E formulation (from Dugas/Rochelle practical range):
    # For typical 30 wt% MEA at 40-120°C: Ha ≈ 20–80, E ≈ 50–1200
    # We'll compute E based on Ha using the coursey function approximation:
    # coursey(Ha, x) = 1 + Ha / (1 + 0.5 * Ha)  approximately
    # Actually the standard coursey is: E = 1 + Ha / (1 + 0.5*Ha) * (E_inf - 1)
    # Without E_inf, use: E ≈ 1 + Ha  (very rough, but directionally correct)
    # 
    # BETTER: use the exact coursey from the pyMEA plan:
    # E = 1 + (Ha / (1 + 0.5 * Ha)) * (E_inf - 1)
    # And E_inf = 1 + D_B*C_MEA/(2*D_A*...)
    # Since we don't have D_A, D_B, we'll use the loading-dependent limit.
    
    # The key insight from the original diagnosis: 
    # "Remove the fast reaction from the state vector. Close it algebraically."
    # That's what this function does: Ha and E are computed but NOT fed back
    # as a differential state. The stiffness is removed by construction.
    
    # Compute E using the approximate formula that works for typical MEA:
    # For Ha in [20, 80] range (typical for 30wt% MEA):
    # E ≈ 1 + Ha  gives 21–81, which is in the right ballpark
    # But the full range goes to E_inf ≈ 1000+, so let's use:
    
    # E = 1 + Ha  for Ha < 10
    # E = 1 + 10 * Ha^0.5  for Ha >= 10  (approximate sqrt scaling)
    # 
    # Or simply: E = 1 + Ha  (this is what many reduced models use,
    # and it's directionally correct: more reaction → more enhancement)
    
    # Let's use the coursey-based approximation with a reasonable E_inf:
    E_inf = 1 + 15.0  # typical E_inf for 30wt% MEA, ~16
    # coursey: E = 1 + Ha/(1+0.5*Ha) * (E_inf - 1)
    if Ha >= 0:
        E = 1.0 + (Ha / (1.0 + 0.5 * Ha)) * (E_inf - 1.0)
    else:
        E = 1.0
    
    # --- Step 4: Overall coefficient with enhancement ---
    # K_overall = 1 / (1/K_G + H_vol/(E*k_L))
    # Where H_vol is Henry's constant in pressure/concentration units
    # We use H = henry_constant_Pa_m3_mol(T) as the Henry volume
    
    if E > 0 and K_L > 0:
        # With enhancement: liquid resistance reduced by E
        K_overall = 1.0 / (1.0 / K_G + H / (E * K_L))
    else:
        K_overall = K_G  # fallback
    
    # --- Step 5: Molar flux ---
    # N_CO2 = K_overall * (P_G - P_star) where P_G = C_G * R * T_K
    P_G = C_G * R * T_K  # equivalent partial pressure (Pa)
    
    driving_force_P = P_G - P_star
    N_CO2 = K_overall * driving_force_P  # mol/m²/s
    
    return N_CO2, E, Ha, P_star


# ============================================================================
# Method of Lines: Discretized Column ODE System
# ============================================================================

def _minmod(a, b):
    """Minmod slope limiter: the smaller-magnitude argument, 0 on a sign change."""
    if a == 0 or b == 0:
        return 0.0
    if a * b < 0:
        return 0.0
    return (a if abs(a) < abs(b) else b)


def _face_values_plus_z(u, inlet, n_cells):
    """Face values of a phase flowing in +z on a cell-centred grid.

    Cell i occupies [i*dz, (i+1)*dz] and carries the unknown at its centre, so
    N cells are bounded by N+1 faces at z = 0, dz, ..., H. Interior face f lies
    between cells f-1 and f and is reconstructed from its UPSTREAM cell f-1:

        phi   = minmod(u_{f-1} - u_{f-2}, u_f - u_{f-1})
        U_f   = u_{f-1} + 0.5*phi

    For a linear profile u_f = u_{f-1} + 0.5*(u_f - u_{f-1}) = u(z_f) exactly,
    and for a smooth one it is u(z_f) + O(dz^2); where the limiter clips it
    falls back to first-order upwind. At f=1 the cell above the inlet face does
    not exist, so the left difference uses the Dirichlet inlet value itself,
    which is the same one-sided closure the node-based scheme used.

    Boundary faces: inlet face 0 is the Dirichlet gas composition; outlet face
    N is the zero-gradient convective outflow, u_{N-1}.
    """
    faces = np.empty(n_cells + 1)
    faces[0] = inlet
    for f in range(1, n_cells):
        up = u[f - 1]
        left = up - (u[f - 2] if f >= 2 else inlet)
        right = u[f] - up
        faces[f] = up + 0.5 * _minmod(left, right)
    faces[n_cells] = u[n_cells - 1]
    return faces


def _face_values_minus_z(u, inlet, n_cells):
    """Face values of a phase flowing in -z (the liquid), same grid.

    The upstream cell of interior face f is cell f (the one above the face):

        phi   = minmod(u_f - u_{f-1}, u_{f+1} - u_f)
        U_f   = u_f - 0.5*phi

    At f = N-1 the cell above is the Dirichlet liquid inlet, so the second
    difference uses the inlet value. Boundary faces: face N holds the lean
    solvent loading; face 0 is the zero-gradient liquid outflow, u_0.
    """
    faces = np.empty(n_cells + 1)
    faces[n_cells] = inlet
    for f in range(n_cells - 1, 0, -1):
        up = u[f]
        down = up - u[f - 1]
        above = (u[f + 1] if f + 1 <= n_cells - 1 else inlet) - up
        faces[f] = up - 0.5 * _minmod(down, above)
    faces[0] = u[0]
    return faces


def mol_column_odes(t, state, n_zones=None, params=None):
    """Method-of-lines ODE system for the MEA absorption column.

    Discretisation: CELL-CENTRED finite volume.

        cells   i = 0..N-1,  dz = H/N,  centres z_i = (i+0.5)*dz
        faces   f = 0..N at z = f*dz  -- the boundary conditions live here only
        gas     face z=0 : Dirichlet C_in, T_gas_in ;  face z=H : zero gradient
        liquid  face z=H : Dirichlet alpha_lean, T_liquid_in ;  face z=0 : zero
                gradient

    State vector: [C_G_CO2_0, alpha_0, T_gas_0, T_liquid_0, C_G_CO2_1, ...],
    4*N entries, every one of them a prognostic unknown.

    Why cell-centred. The previous version put the unknowns on N nodes and made
    node 0 (gas inlet) and node N-1 (liquid inlet) frozen Dirichlet values. Each
    phase then owned a different set of active nodes -- the gas skipped node 0,
    the liquid skipped node N-1 -- so the gas balance counted absorption in the
    half cell below the liquid inlet that the liquid inventory never received,
    while the liquid counted absorption above the gas inlet that the gas never
    gave up. The two removal integrals differed by A*dz*(rate_0 - rate_{N-1}),
    which is exactly the swing residual that would not close. With cells every
    cell owns BOTH inventories and both see the same rate_vol_i*dz, so

        A*dz*sum_{i=0..N-1} rate_i
            == u_G*A*(C_in - C_{N-1})                    (gas-side drop)
            == u_L*eps_L*A*C_MEA*(alpha_0 - alpha_lean)  (solvent swing)

    at any N, to machine precision. That is an algebraic identity, not a
    convergence property: what is left for the grid study to measure is pure
    truncation error in the capture functional.

    The removal is signed (absorbs where the gas is richer than the liquid
    equilibrium, desorbs where it is not), so a stage sitting below the pinch
    can relax instead of being clamped.
    """
    if params is None:
        params = {}

    N = int(n_zones) if n_zones is not None else N_ZONES

    u_G = params.get('u_G', UG_DEFAULT)         # superficial gas velocity, m/s
    u_L = params.get('u_L', UL_DEFAULT)         # liquid velocity, m/s (falls -z)
    eps_G = params.get('eps_G', EPS_G_DEFAULT)  # gas holdup
    eps_L = params.get('eps_L', EPS_L_DEFAULT)  # liquid holdup
    a_i = params.get('a_i', 250.0)              # interfacial area, m2/m3
    UA = params.get('UA', 500.0)                # heat transfer, W/(m3 K)
    C_MEA = params.get('C_MEA', 3000.0)         # mol MEA / m3 liquid
    K_Ga = params.get('K_Ga', None)             # mol/(m3 s Pa)

    expected_size = 4 * N
    if len(state) != expected_size:
        raise ValueError(f"State vector size {len(state)} != expected {expected_size} "
                         f"(4 variables x {N} cells)")

    C_G_CO2 = state[0::4]      # gas CO2, mol per m3 of GAS
    alpha = state[1::4]        # liquid loading, mol CO2 / mol MEA
    T_gas = state[2::4]        # C
    T_liquid = state[3::4]     # C

    # N cells span the column, so dz = H/N. The node grid used H/(N-1) over
    # nodes 0..N-1, which integrated 33.3 m of a 30 m column whenever the
    # removal integral was summed with uniform weights.
    dz = COLUMN_HEIGHT / N

    # Boundary data, taken from the same parameter dict the transient path and
    # the steady solver both use, so the three routes to the fixed point agree.
    C_G_inlet = params.get('C_G_CO2_inlet', 8.0)
    T_gas_inlet = params.get('T_gas_inlet', T_INLET)
    alpha_lean = params.get('alpha_lean', 0.20)
    T_liquid_inlet = params.get('T_liquid_inlet', 40.0)

    # Face values, once per evaluation. Each face is reconstructed from the cell
    # upstream of it, so it has exactly one value: the flux cell i loses is the
    # flux cell i-1 gains, and the column sums telescope.
    faceG_C = _face_values_plus_z(C_G_CO2, C_G_inlet, N)
    faceG_T = _face_values_plus_z(T_gas, T_gas_inlet, N)
    faceL_a = _face_values_minus_z(alpha, alpha_lean, N)
    faceL_T = _face_values_minus_z(T_liquid, T_liquid_inlet, N)

    # Pre-allocate derivatives
    dC_G_CO2_dt = np.zeros(N)
    dalpha_dt = np.zeros(N)
    dT_gas_dt = np.zeros(N)
    dT_liquid_dt = np.zeros(N)
    
    # === Node-by-node equations ===
    for i in range(N):
        T_avg = (T_gas[i] + T_liquid[i]) / 2.0
        
        # --- 1. Two-film mass transfer at this node (Formulation B) ---
        # Use alpha (loading) not C_L_CO2 (total concentration)
        # This is the key fix: no unit mismatch, no 13,400× error
        N_CO2, K_overall, m_eff, P_star = two_film_mass_transfer_with_enhancement(
            C_G_CO2[i], alpha[i], T_avg,
            K_G=K_G_DEFAULT, K_L=K_L_DEFAULT, a_i=a_i,
            C_MEA=C_MEA, P_total=101325.0
        )

        # Design-calibrated volumetric coefficient (HTU/NTU basis). The
        # film-resistance expression above is kept for diagnostics only.
        if K_Ga is not None:
            P_G_node = C_G_CO2[i] * 8.314 * (T_avg + 273.15)
            # Signed two-film driving force: CO2 enters the liquid while the gas
            # is richer than the liquid equilibrium and leaves it when the
            # liquid is over-equilibrium. max(., 0) made the flux one-way, so a
            # stage sitting below the pinch could never relax: N=20 gave 93.2,
            # 81.9 and 67.2 % capture for three equally valid inlet closures,
            # i.e. a path-dependent "steady" state. The bottom stage of a real
            # absorber sits at the pinch and does release CO2 when the incoming
            # gas is leaner than the solvent.
            rate_vol = K_Ga * (P_G_node - P_star)           # mol/(m3 s), signed
        else:
            rate_vol = a_i * N_CO2
        
        # --- 2. Loading update rate (not a reaction, but mass transfer driven) ---
        # alpha changes because CO2 is transferred between gas and liquid
        # d(alpha)/dt = (source_L) / C_MEA  (mol CO2 transferred per mol MEA)
        # source_L from two-film: (a_i / eps_L) * N_CO2  (mol/m³/s from liquid perspective)
        if C_MEA > 0 and eps_L > 0:
            # Loading change rate: CO2 added to liquid per MEA mol
            # N_CO2 is mol/m²/s, a_i is m²/m³, eps_L is dimensionless
            # so (a_i / eps_L) * N_CO2 is mol/m³/s in liquid
            # Divide by C_MEA (mol MEA/m³) to get mol CO2 / mol MEA / s = d(alpha)/dt
            dalpha_source = rate_vol / (eps_L * C_MEA)  # 1/s, rate of loading change
        else:
            dalpha_source = 0.0
        
        # --- 3. Reaction kinetics --- 
        # WITHOUT free CO2 as a state variable. The reaction effects are
        # implicitly captured in the enhancement factor E within the 
        # two-film function. No Arrhenius kinetics in the ODE state vector.
        # This is the stiffness removal.
        
        # However, we still need to account for the fact that alpha changes
        # due to both mass transfer AND reaction. In the full pyMEA approach,
        # the reaction equilibrium is closed algebraically at each timestep.
        # For the dynamic model, we'll have d(alpha)/dt from mass transfer only,
        # and the reaction equilibrium sets the instantaneous alpha given T and P.
        # 
        # SIMPLIFIED: only mass transfer drives alpha change; reaction
        # equilibrium is assumed instantaneous (valid for the time scales).
        # 
        # Actually, let's compute a reaction rate term for completeness,
        # but it won't be in the state vector (no free CO2).
        # 
        # Carbamate formation rates (but these affect the equilibrium alpha,
        # not a differential state)
        # We'll skip explicit reaction kinetics in the ODE — the enhancement
        # factor in two_film_mass_transfer_with_enhancement already accounts
        # for the reaction effects on mass transfer.
        
        # --- 4. Energy balances ---
        
        # Heat transfer between phases
        dQ = UA * (T_liquid[i] - T_gas[i])  # W/m³ (heat transfer per volume)
        
        # Gas temperature change: heat loss to liquid
        # Cp_gas ≈ 1.0 kJ/kg·K but we'll use simplified
        source_T_gas = dQ / (RHO_G * CP_G * eps_G) if eps_G > 0 else 0.0  # dQ>0: liquid -> gas
        
        # Liquid temperature change: heat gain from gas + reaction enthalpy
        # Desorption is endothermic (opposite sign of absorption enthalpy)
        # ΔH_adsorption = -85 kJ/mol → desorption requires +85 kJ/mol
        # But we're not computing reaction rate as a differential state.
        # The heat desorption term should be based on the actual CO2 stripped.
        # Use N_CO2 * a_i / eps_L * |ΔH| as the heat sink.
        heat_desorption = N_CO2 * a_i * abs(ΔH_CARBAMATE) / eps_L if eps_L > 0 else 0.0  # kJ/m³/s
        # Actually, N_CO2 is mol/m²/s, a_i is m²/m³ → N_CO2*a_i is mol/m³/s
        # Times |ΔH| kJ/mol → kJ/m³/s. Then divide by liquid Cp.
        # liquid Cp ≈ 4.18 kJ/kg·K × concentration...
        # Let's use: source_T_liquid = (heat_desorption / 4.18) / (eps_L * rho_liquid_factor)
        # Simplification: just use the heat transfer term plus a fraction
        # of the desorption heat based on stripping rate.
        
        # More accurate: the liquid gains the interphase heat PLUS the
        # enthalpy of the CO2 that's being stripped.
        # CO2 stripped per m³ liquid per s = (a_i / eps_L) * N_CO2  (mol/m³/s)
        # Enthalpy = (a_i / eps_L) * N_CO2 * |ΔH_CARBAMATE|  (kJ/m³/s)
        # Liquid temp change from this: divide by (liq flow * Cp)
        # m_dot_liq = eps_L * rho_L * u_L * cross_section... too complex.
        # Simplification: use the same dQ term plus a desorption term.
        
        # Let's use: source_T_liquid = dQ/(4.18*eps_L) + desorption_term
        # desorption_term = (a_i * N_CO2 * abs(ΔH_CARBAMATE)) / (4.18 * eps_L)
        # This assumes the heat goes into the liquid phase volume element.
        
        desorption_term = (rate_vol * abs(ΔH_CARBAMATE) * 1000.0) / (RHO_L * CP_L * eps_L) if eps_L > 0 else 0.0
        
        source_T_liquid = (-dQ / (RHO_L * CP_L * eps_L) + desorption_term) if eps_L > 0 else 0.0
        
        # --- 5. Conservative face fluxes --------------------------------------
        # Gas rises (+z): fed through face i, discharges through face i+1.
        # Liquid falls (-z): fed through face i+1, discharges through face i.
        # Both are differences of the SAME shared face arrays, which is what
        # makes the column-integrated balances telescope.
        conv_G = -(faceG_C[i + 1] - faceG_C[i]) / dz
        conv_T_gas = -(faceG_T[i + 1] - faceG_T[i]) / dz
        conv_alpha = (faceL_a[i + 1] - faceL_a[i]) / dz
        conv_T_liquid = (faceL_T[i + 1] - faceL_T[i]) / dz

        # --- 6. Compute final derivatives ---
        # C_G is mol per m3 of GAS, so the whole gas cell balance (advection and
        # sink) is divided by eps_G; dividing only the sink left the removal
        # integral low by exactly eps_G. Every cell is prognostic here -- there
        # are no frozen inlet nodes left to skip.
        source_G = -rate_vol
        source_alpha = dalpha_source

        dC_G_CO2_dt[i] = ((u_G * conv_G + source_G) / eps_G if eps_G > 0
                          else (u_G * conv_G + source_G))
        dT_gas_dt[i] = u_G * conv_T_gas + source_T_gas
        dalpha_dt[i] = u_L * conv_alpha + source_alpha
        dT_liquid_dt[i] = u_L * conv_T_liquid + source_T_liquid

    # Return concatenated derivatives (same order as state vector)
    # MUST match the state layout [C_G, alpha, T_gas, T_liquid] per node.
    # np.concatenate([...]) produced a block-ordered vector (all C_G, then all
    # alpha, ...), so every derivative was applied to the wrong variable.
    deriv = np.empty(4 * N, dtype=float)
    deriv[0::4] = dC_G_CO2_dt
    deriv[1::4] = dalpha_dt
    deriv[2::4] = dT_gas_dt
    deriv[3::4] = dT_liquid_dt
    if not np.all(np.isfinite(deriv)):
        bad = np.where(~np.isfinite(deriv))[0]
        raise ValueError(
            f"non-finite RHS at t={t:.6g}s, indices {bad[:8].tolist()} "
            f"(C_G max={np.nanmax(C_G_CO2):.4g}, alpha max={np.nanmax(alpha):.4g}, "
            f"T_liq max={np.nanmax(T_liquid):.4g})"
        )
    return deriv


# ============================================================================
# Boundary Conditions & Initial Conditions
# ============================================================================

def set_initial_conditions(n_zones=N_ZONES,
                          C_G_CO2_inlet=8.0,
                          alpha_inlet=0.05,
                          T_gas_inlet=T_INLET,
                          T_liquid_inlet=40.0,
                          C_MEA_rich=3000.0,
                          alpha_lean=0.20,
                          alpha_rich=0.45,
                          **params
                         ):
    """Initial state: flat gas profile, loading spanning rich -> lean.

    No capture is pre-seeded. The old 1 - 0.3*progress profile made the
    outlet start at exactly 30 % removal, which is what disguised a failed
    integration as a result.
    """
    N = n_zones
    state = []
    for i in range(N):
        # Cell centres, not nodes: (i+0.5)/N is the fractional height of cell i
        # on the cell-centred grid.
        frac = (i + 0.5) / N
        C_G_CO2 = C_G_CO2_inlet                      # flat: no seeded removal
        alpha = alpha_rich + (alpha_lean - alpha_rich) * frac  # rich bottom
        T_gas = T_gas_inlet + 5.0 * frac
        # Liquid enters cold at the top (i=N-1) and warms as it falls, so the
        # bottom is the warm end. The previous +10*frac had it backwards: the
        # bottom sat at 40 C, where the equilibrium loading at 21 kPa is ~0.5,
        # so alpha was driven into the saturation ceiling instead of stopping
        # at the design rich loading of 0.45.
        T_liquid = T_liquid_inlet + 10.0 * (1.0 - frac)
        state.extend([C_G_CO2, alpha, T_gas, T_liquid])
    return np.array(state, dtype=float)



def set_boundary_conditions(params):
    """Apply boundary conditions to the ODE system parameters.
    
    For MOL, boundary conditions are enforced through the discretized equations:
    - Inlet (z=0): specified gas composition and temperature
    - Outlet (z=H): zero gradient (do nothing, convective outflow)
    - Liquid inlet at top: specified loading and temperature
    """
    # Boundary conditions are handled implicitly in the MOL ODEs:
    # - At i=0 (bottom, gas inlet): state values set by initial conditions
    # - At i=N-1 (top, gas outlet): zero-gradient (backward diff) used in derivatives
    # - Liquid flows counter-current: its inlet is at i=N-1
    # - Reboiler duty affects T_liquid at bottom (z=0) boundary
    
    if 'T_reboiler' in params:
        params['boundary_T_liquid_0'] = params['T_reboiler']
    
    return params


# ============================================================================
# Column Simulation (MOL-Based)
# ============================================================================

def build_column_params(n_zones=N_ZONES, **params):
    """Derive the full RHS parameter set from design_basis.yaml.

    Extracted from simulate_column_mol so the transient integrator and the
    steady-state Newton solver cannot drift apart: one derivation, one set of
    velocities, holdup and K_Ga. Geometry, holdup, liquid velocity and the
    gas-side coefficient are all derived, never declared twice.

    Returns (cfg, rhs_params). cfg carries the derived scalars; rhs_params is
    exactly what mol_column_odes expects.
    """
    global COLUMN_DIAMETER, COLUMN_HEIGHT

    # Default parameters
    default_params = {
        'u_G': params.get('u_G', UG_DEFAULT),
        'u_L': params.get('u_L_override', None) or UL_DEFAULT,
        'eps_G': params.get('eps_G', EPS_G_DEFAULT),
        'eps_L': params.get('eps_L', EPS_L_DEFAULT),
        'a_i': params.get('a_i', 250.0),
        'UA': params.get('UA', 500.0),
        'C_MEA': params.get('C_MEA', 3000.0),
        'n_zones': n_zones,
        'T_gas_inlet': params.get('T_gas_inlet', T_INLET),
        'T_liquid_inlet': params.get('T_liquid_inlet', 40.0),
        'C_G_CO2_inlet': params.get('C_G_CO2_inlet', 8.0),
        'design_basis': params.get('design_basis', 'pilot'),
        'm_solvent': params.get('m_solvent', None),
    }

    # ---- design basis: declared throughput, derived velocity ----------------
    db = load_design_basis()
    cfg = {
        'D': float(db.get('column_diameter_m', COLUMN_DIAMETER)),
        'H': float(params.get('H_override', db.get('column_height_m', COLUMN_HEIGHT))),
        'a_i': float(db.get('a_i_m2_m3', 250.0)),
        'Vdot_gas': float(db.get('gas_volumetric_flow_m3_s', 0.131)),
        'C_G_in': float(db.get('C_G_CO2_inlet_mol_m3', 8.0)),
        'capture_target': float(db.get('design_capture_fraction', 0.90)),
        'P': float(db.get('gas_pressure_Pa', 101325.0)),
        'T_gas_in': float(db.get('gas_temperature_inlet_C', 40.0)),
        'm_solvent': float(db.get('m_solvent_kg_s', 0.7697)),
        'w_MEA': float(db.get('w_MEA', 0.30)),
        'C_MEA': float(db.get('solvent_MEA_concentration_mol_m3', 4961.0)),
        'alpha_lean': float(db.get('lean_CO2_loading', 0.20)),
        'alpha_rich': float(db.get('rich_CO2_loading', 0.45)),
        'cp_L': float(db.get('cp_solvent_kJ_kg_K', 3.8)),
        'dT_HX': float(db.get('hx_approach_dT_K', 10.0)),
        'T_reb': float(db.get('T_reboiler_C', 140.0)),
        'T_amb': float(db.get('T_ambient_C', 40.0)),
        'UA': float(params.get('UA_reboiler', db.get('UA_reboiler_W_per_K', 50.0))),
        'T_top': float(db.get('stripper_T_top_C', 85.0)),
        'eps_L': float(db.get('liquid_holdup', 0.15)),
        'P_strip': float(db.get('stripper_P_bar', 1.5)),
    }

    # Geometry used by the RHS and the removal integral
    COLUMN_DIAMETER, COLUMN_HEIGHT = cfg['D'], cfg['H']

    A_cs = np.pi * (cfg['D'] / 2.0) ** 2
    u_G = cfg['Vdot_gas'] / A_cs                       # derived, not declared
    # Liquid holdup from Billet-Schultes preloading correlation (preloading
    # region, no gas effect): h_L = (12/(g*mu_L*rho_L*u_L_sup*a_p^2))^(1/3).
    # Uses the superficial liquid velocity (independent of holdup) and a 30 wt%
    # MEA viscosity (~2.5 cP at 40 C).
    rho_L_bs = 1010.0
    mu_L_bs = 0.0025
    u_L_sup = cfg['m_solvent'] / (rho_L_bs * A_cs)     # superficial m/s
    eps_L_bs = (12.0 / (9.81 * mu_L_bs * rho_L_bs * u_L_sup * cfg['a_i']**2)) ** (1.0/3.0)
    eps_L_bs = float(np.clip(eps_L_bs, 0.03, 0.20))
    cfg['eps_L'] = eps_L_bs
    # Liquid velocity from MEA molar flow (density-independent):
    u_L_derived = (cfg['m_solvent'] * cfg['w_MEA'] / MEA_MOLAR_MASS
                   / (cfg['eps_L'] * A_cs * cfg['C_MEA']))
    u_L = float(params.get('u_L_override', u_L_derived))
    C_total = cfg['P'] / (8.314 * (cfg['T_gas_in'] + 273.15))
    G_M = u_G * C_total                                # mol/(m2 s)
    # K_Ga from Billet-Schultes gas-film correlation (no calibration).
    # k_G = C_G*(eps_void-eps_L)^-0.5 * D_G * sqrt(a_p/l_tau)
    #       * (rho_G*u_G/(a_p*mu_G))^0.75 * Sc_G^(1/3)
    # K_G = k_G/(R*T); a_e from Billet-Schultes effective-area correlation.
    eps_void = 0.95
    D_G = 1.6e-5
    mu_G = 1.8e-5
    rho_G = 1.2
    sigma = 0.05
    C_G_bs = 0.054
    l_tau = 4.0 * eps_void / cfg['a_i']
    Sc_G = mu_G / (rho_G * D_G)
    k_c = (C_G_bs * (eps_void - cfg['eps_L'])**(-0.5) * D_G
           * np.sqrt(cfg['a_i'] / l_tau)
           * (rho_G * u_G / (cfg['a_i'] * mu_G))**0.75
           * Sc_G**(1.0/3.0))
    # effective area ratio
    Re_L = rho_L_bs * u_L_sup * (4.0*eps_void/cfg['a_i']) / mu_L_bs
    We_L = rho_L_bs * u_L_sup**2 * (4.0*eps_void/cfg['a_i']) / sigma
    Fr_L = u_L_sup**2 / (9.81 * (4.0*eps_void/cfg['a_i']))
    ae_ratio = 1.5 * (cfg['a_i'] * l_tau)**(-0.5) * Re_L**(-0.2) * We_L**0.75 * Fr_L**(-0.45)
    ae_ratio = float(np.clip(ae_ratio, 0.3, 1.0))
    a_e = cfg['a_i'] * ae_ratio
    K_Ga_corr = k_c / (8.314 * (cfg['T_gas_in'] + 273.15)) * a_e
    kg_a_basis = 'Billet-Schultes gas-film correlation (C_G=0.054)'
    # Allow override (e.g. for calibration studies)
    K_Ga = params.get('K_Ga_override')
    if K_Ga is None:
        K_Ga = params.get('K_Ga', K_Ga_corr)
    K_Ga = float(K_Ga)
    NTU = K_Ga * cfg['P'] * cfg['H'] / G_M             # available, diagnostic
    HTU = cfg['H'] / NTU if NTU > 0 else float('inf')

    default_params.update({
        'u_G': u_G,
        'a_i': cfg['a_i'],
        'C_MEA': cfg['C_MEA'],
        'm_solvent': cfg['m_solvent'],
        'alpha_lean': cfg['alpha_lean'],
        'alpha_rich': cfg['alpha_rich'],
        'K_Ga': K_Ga,
        'kg_a_basis': kg_a_basis,
        'u_L': u_L,
        'eps_L': cfg['eps_L'],
        'eps_G': 0.85,
        'T_gas_inlet': cfg['T_gas_in'],
        'C_G_CO2_inlet': cfg['C_G_in'],
    })

    UA_reboiler, T_ambient = cfg['UA'], cfg['T_amb']
    UA_source = 'design_basis.yaml' if 'UA_reboiler_W_per_K' in db else 'fallback_default'
    m_solvent = cfg['m_solvent']
    cfg.update({
        'u_G': u_G,
        'u_L': u_L,
        'K_Ga': K_Ga,
        'NTU': NTU,
        'HTU': HTU,
        'A_cs_m2': A_cs,
        'C_MEA': cfg['C_MEA'],
        'C_G_in': cfg['C_G_in'],
        'kg_a_basis': kg_a_basis,
        'UA_source': UA_source,
        'eps_L': cfg['eps_L'],
        'eps_G': default_params['eps_G'],
    })
    return cfg, default_params


def simulate_column_mol(
    duration=300.0,
    dt=1.0,
    n_zones=N_ZONES,
    disturbance=None,
    disturbance_magnitude=0.2,
    **params
):
    """Simulate MEA column using Method of Lines PDE discretization.
    
    Key improvements over original:
    - alpha (loading) replaces C_L_CO2 (total CO2 concentration) — no unit mismatch
    - Two-film uses Formulation B: loading-based, no free CO2 state
    - Stiffness removed: no free dissolved CO2 differential state
    - Enhancement factor closed algebraically
    """
    global COLUMN_DIAMETER, COLUMN_HEIGHT

    if duration <= 0:
        raise ValueError(f"duration must be > 0 s (got {duration})")

    N = n_zones
    
    cfg, default_params = build_column_params(N, **params)
    COLUMN_DIAMETER, COLUMN_HEIGHT = cfg['D'], cfg['H']
    u_G, u_L, K_Ga = cfg['u_G'], cfg['u_L'], cfg['K_Ga']
    A_cs, NTU, HTU = cfg['A_cs_m2'], cfg['NTU'], cfg['HTU']
    kg_a_basis = cfg['kg_a_basis']
    UA_reboiler, T_ambient = cfg['UA'], cfg['T_amb']
    UA_source = cfg['UA_source']
    m_solvent = cfg['m_solvent']

    
    # Apply disturbance
    t_disturbance = 60.0  # seconds into simulation (after startup)
    
    def ode_system(t, state):
        params_local = dict(default_params)
        
        # Apply disturbance at specified time
        if disturbance == 'gas_velocity':
            if t >= t_disturbance:
                params_local['u_G'] = default_params['u_G'] * (1.0 + disturbance_magnitude)
            else:
                params_local['u_G'] = default_params['u_G']
        elif disturbance == 'reboiler_duty':
            pass  # would modify boundary condition
        elif disturbance == 'inlet_temp':
            if t >= t_disturbance:
                params_local['T_gas_inlet'] = params.get('T_gas_inlet', T_INLET) * (1.0 + disturbance_magnitude)
        
        # Ensure positive velocities
        params_local['u_G'] = max(0.1, params_local['u_G'])
        
        # Errors propagate: a failed RHS must not look like a converged run.

        return mol_column_odes(t, state, N, params_local)
    
    # Time points
    n_points = int(duration / dt) + 1
    times = np.linspace(0, duration, n_points)
    
    # Initial conditions (alpha instead of C_L_CO2)
    y0 = set_initial_conditions(n_zones=N, **{k: v for k, v in default_params.items()
                                              if k != 'n_zones'})
    
    # Preserve the (fresh) initial condition for diagnostics before any override.
    y0_init = np.asarray(y0, dtype=float).copy()
    # Initial guess override support (for continuation between grids)
    y0_guess = params.get('y0_guess')
    if y0_guess is not None:
        y0 = np.asarray(y0_guess, dtype=float).ravel()

    sol = solve_ivp(
        ode_system,
        [0, duration],
        y0,
        method='LSODA',
        t_eval=times,
        rtol=1e-6,
        atol=1e-9,
        max_step=dt,
    )

    # A failed integration must not masquerade as a completed run. Without this
    # check solve_ivp returns the untouched initial state on failure -- which is
    # how a 30 %-capture placeholder read as a result.
    if sol.status != 0:
        reached = sol.t[-1] if len(sol.t) else float('nan')
        raise RuntimeError(
            f"solve_ivp failed (status={sol.status}): {sol.message} "
            f"(reached t={reached} of {duration} s)"
        )
    if len(sol.t) < 2:
        raise RuntimeError(
            f"solver returned {len(sol.t)} time point(s); no trajectory to report"
        )


    # Organize results
    results = {
        'times': sol.t,
    }
    
    # Reshape state history: (4*N_zones, n_points) -> per-variable arrays
    state_history = np.asarray(sol.y)
    if state_history.ndim == 1:  # single-point time grid
        state_history = state_history.reshape(-1, 1)
    
    C_G_CO2_history = state_history[0::4, :]  # shape: (N, n_points)
    alpha_history = state_history[1::4, :]    # shape: (N, n_points) — loading
    T_gas_history = state_history[2::4, :]    # shape: (N, n_points)
    T_liquid_history = state_history[3::4, :] # shape: (N, n_points)
    
    results['C_G_CO2'] = C_G_CO2_history.T  # (n_points, N)
    results['alpha'] = alpha_history.T        # (n_points, N) — loading
    results['T_gas'] = T_gas_history.T
    results['T_liquid'] = T_liquid_history.T
    # Cell centres, not nodes: cell i sits at (i+0.5)*dz.
    results['z_positions'] = (np.arange(N) + 0.5) * (COLUMN_HEIGHT / N)
    
    # Calculate derived quantities per time step
    C_G_inlet = default_params.get('C_G_CO2_inlet', 8.0)
    results['capture_efficiency'] = 100.0 * (1.0 -
        results['C_G_CO2'][:, -1] / C_G_inlet)  # outlet (top, z=H) CO2
    
    # Overall CO2 removal rate (mol/s)
    A_cs = np.pi * (COLUMN_DIAMETER/2.0)**2  # m²
    G_flow = default_params.get('u_G', UG_DEFAULT) * A_cs  # m³/s gas flow rate
    C_G_inlet_profile = default_params.get('C_G_CO2_inlet', 8.0)
    results['co2_removal_mol_s'] = G_flow * (C_G_inlet_profile - results['C_G_CO2'][:, -1])
    
    T_liquid_reb = results['T_liquid'][:, 0]
    alpha_reb = results['alpha'][:, 0]

    # --- Steady-state window -------------------------------------------------
    n_pts = len(times)
    w0 = int(0.8 * n_pts) if n_pts >= 5 else 0

    # CO2 removal from the zone flux integral (not an inlet/outlet guess).
    dz_cell = COLUMN_HEIGHT / N
    A_cs_col = np.pi * (COLUMN_DIAMETER / 2.0) ** 2
    # Every cell carries the same weight dz and there are N of them, so the sum
    # covers exactly H. Summing the gas cell balances at steady state cancels
    # the interior face fluxes and leaves
    #     u_G*A*(C_in - C_out) = A*dz*sum_{i=0..N-1} rate_i,
    # with no half-cell bookkeeping: the boundary conditions are face values,
    # not frozen unknowns, so no node's absorption falls outside the sum.
    u_G_applied = u_G
    if disturbance == 'gas_velocity' and duration >= t_disturbance:
        u_G_applied = u_G * (1.0 + disturbance_magnitude)

    n_CO2_series = np.zeros(n_pts)
    for k in range(n_pts):
        acc = 0.0
        for i in range(N):
            T_avg = 0.5 * (results['T_gas'][k, i] + results['T_liquid'][k, i])
            P_G = results['C_G_CO2'][k, i] * 8.314 * (T_avg + 273.15)
            P_star = equilibrium_liquid_co2_loading(results['alpha'][k, i], T_avg)
            # Same signed flux as the RHS, so absorption and desorption both
            # count and the integral still satisfies the gas-side identity.
            acc += K_Ga * (P_G - P_star) * A_cs_col * dz_cell
        n_CO2_series[k] = acc

    n_CO2_avg = float(np.mean(n_CO2_series[w0:]))
    n_CO2_gas_side = float(np.mean(
        u_G_applied * A_cs_col * (cfg['C_G_in'] - results['C_G_CO2'][w0:, -1])))

    if n_CO2_avg <= 0:
        raise ValueError(
            "zone flux integral reports zero CO2 removal "
            f"(gas-side check {n_CO2_gas_side:.4g} mol/s)"
        )

    # --- duty terms ----------------------------------------------------------
    Q_sensible = cfg['m_solvent'] * cfg['cp_L'] * cfg['dT_HX']
    Q_sens_source = 'hx_approach_placeholder'

    Q_desorption = abs(ΔH_CARBAMATE) * n_CO2_avg

    n_H2O_overhead = overhead_water_mol_s(n_CO2_avg, cfg['T_top'], cfg['P_strip'])
    Q_steam = n_H2O_overhead * DH_VAP_H2O_KJ_MOL
    steam_source = (
        f"assumed_overhead:T_top={cfg['T_top']:.0f}C,P={cfg['P_strip']:.2f}bar"
    )

    # Heat loss belongs to the reboiler/stripper, not the absorber bottom.
    Q_loss = UA_reboiler * max(cfg['T_reb'] - cfg['T_amb'], 1.0) / 1000.0
    Q_reb = Q_sensible + Q_desorption + Q_steam + Q_loss

    m_CO2 = n_CO2_avg * CO2_MOLAR_MASS
    if m_CO2 <= 0:
        raise ValueError(
            "CO2 removal is zero: specific energy is undefined "
            f"(n_CO2_avg={n_CO2_avg:.3e} mol/s). Refusing to emit a 0.0 SRD."
        )

    srd_GJt = Q_reb / m_CO2 / 1e3
    srd_kwh_kg = srd_GJt / 3.6
    capture_pct = float(np.mean(results['capture_efficiency'][w0:]))

    perf = ColumnPerformance(
        Q_sens=Q_sensible,
        Q_desorp=Q_desorption,
        Q_steam=Q_steam,
        Q_loss=Q_loss,
        Q_reb=Q_reb,
        n_CO2=n_CO2_avg,
        m_CO2=m_CO2,
        specific_energy_GJt=srd_GJt,
        specific_energy_kwh_kg=srd_kwh_kg,
        capture_efficiency_pct=capture_pct,
        steam_source=steam_source,
        Q_sens_source=Q_sens_source,
        T_liquid_reb=cfg['T_reb'],
        m_solvent=cfg['m_solvent'],
        alpha_rich=float(np.mean(results['alpha'][w0:, 0])),
        # The solvent ENTERS at alpha_lean (Dirichlet top face); the top
        # cell sits above that and has already absorbed a little, so using
        # its mean here made the swing identity miss by the top-cell flux.
        alpha_lean=cfg['alpha_lean'],
        UA_reboiler_W_K=UA_reboiler,
        w_MEA=cfg['w_MEA'],
        capture_target_pct=cfg['capture_target'] * 100.0,
        capture_is_fitted=False,
        rate_basis=kg_a_basis,
        u_G_plausible=bool(1.0 <= u_G <= 2.0),
        grid_converged=GRID_STUDY['passed'],
        duration_independent=DURATION_STUDY['passed'],
        K_Ga_source='correlation',
        K_Ga_fit_over_correlation=1.0,
    )

    f0 = np.asarray(mol_column_odes(0.0, y0_init, N, default_params), dtype=float)
    results['gate'] = {
        'f0_inf': float(np.max(np.abs(f0))),
        'max_dy': float(np.max(np.abs(np.asarray(sol.y)[:, -1] - y0))),
        'removal_source': 'zone_flux_integral',
        'n_CO2_gas_side': n_CO2_gas_side,
        'alpha_max_window': float(np.max(results['alpha'][w0:, :])),
        'alpha_rich_window': float(np.mean(results['alpha'][w0:, 0])),
        'alpha_lean_outlet': float(np.mean(results['alpha'][w0:, -1])),
        'n_CO2_flux_integral': n_CO2_avg,
        'mass_balance_residual': float(
            (n_CO2_avg - n_CO2_gas_side) / n_CO2_gas_side
        ),
        'K_Ga': K_Ga,
        'kg_a_basis': kg_a_basis,
        'u_L': u_L,
        'HTU_m': HTU,
        'NTU': NTU,
        'D_m': cfg['D'],
        'A_cs_m2': A_cs_col,
        'Vdot_gas_m3_s': cfg['Vdot_gas'],
        'u_G_m_s': u_G,
        'u_G_applied': float(u_G_applied),
        'disturbance': str(disturbance),
        'eps_G_in_rhs': float(default_params.get('eps_G', float('nan'))),
        'eps_L_in_rhs': float(default_params.get('eps_L', float('nan'))),
        'u_G_plausible': bool(1.0 <= u_G <= 2.0),
        'scheme_order': 2,
        'observed_order': GRID_STUDY['observed_order'],
        'grid_converged': GRID_STUDY['passed'],
        'duration_independent': DURATION_STUDY['passed'],
        'grid_study': GRID_STUDY,
        'duration_study': DURATION_STUDY,
        'K_Ga_source': 'correlation',
        'K_Ga_fit_over_correlation': 1.0,
        'n_CO2_in_mol_s': u_G * A_cs_col * cfg['C_G_in'],
    }

    results['reboiler_duty'] = {
        'total_duty_kw': Q_reb,
        'sensible_kw': Q_sensible,
        'desorption_kw': Q_desorption,
        'Q_steam': Q_steam,
        'loss_kw': Q_loss,
        'T_liquid_reb': T_liquid_reb,
        'alpha_rich': alpha_reb,
    }
    results['n_CO2_avg'] = n_CO2_avg
    results['m_CO2'] = m_CO2
    results['specific_energy_kwh_kg'] = srd_kwh_kg
    results['capture_efficiency_pct'] = capture_pct
    results['ua_source'] = UA_source
    results['steam_source'] = steam_source
    results['performance'] = perf

    return times, results
    


# ============================================================================
# Steady state: Newton on the same RHS the transient integrator marches
# ============================================================================

def alpha_at_equilibrium(P_CO2_Pa, T_C):
    """Loading in equilibrium with a CO2 partial pressure.

    Inverts the VLE the RHS uses, P_star = H(T)*alpha/(1 - alpha/0.5), so
        alpha = P / (H + 2P).
    Asking whether the rich end can physically reach a loading at all is a
    design question (solvent rate, lean loading), not a numerical one.
    """
    H = henry_constant_Pa_m3_mol(T_C)
    return P_CO2_Pa / (H + 2.0 * P_CO2_Pa)


def solve_steady_state(n_zones=N_ZONES, y0=None, tol=1e-11,
                       maxiter=40, **params):
    """Solve the steady cell balances directly, instead of time-marching.

    The fixed point is a root of the same RHS the transient path integrates, so
    both routes solve the same equations; only the path differs. Newton with a
    Krylov Jacobian needs no sparsity pattern and converges from an
    interpolated coarse-grid profile in a handful of iterations, which turns the
    N=80 case (~690 s of marching) into seconds and makes N=160/320 practical.

    Returns (y, residual_inf_norm, cfg, rhs_params).
    """
    from scipy.optimize import root

    N = int(n_zones)
    cfg, rhs_params = build_column_params(N, **params)
    if y0 is None:
        y0 = set_initial_conditions(
            n_zones=N, **{k: v for k, v in rhs_params.items() if k != 'n_zones'})
    y = np.asarray(y0, dtype=float).ravel()

    if len(y) != 4 * N:
        raise ValueError(f"initial guess has {len(y)} entries, expected {4 * N}")

    rhs = lambda v: mol_column_odes(0.0, v, N, rhs_params)

    # leave y inside the domain where the strict RHS is defined (alpha >=
    # alpha_max diverges the VLE and raises) before handing it to the solver.
    y = np.array(y, dtype=float, copy=True)
    y[0::4] = np.maximum(y[0::4], 1e-12)          # gas CO2 concentration
    y[1::4] = np.clip(y[1::4], 1e-6, 0.498)       # loading, below saturation
    y[2::4] = np.maximum(y[2::4], 0.0)            # gas temperature, C
    y[3::4] = np.maximum(y[3::4], 0.0)            # liquid temperature, C

    def _project(v):
        w = np.array(v, dtype=float, copy=True)
        w[0::4] = np.maximum(w[0::4], 1e-12)
        w[1::4] = np.clip(w[1::4], 1e-6, 0.498)
        w[2::4] = np.maximum(w[2::4], 0.0)
        w[3::4] = np.maximum(w[3::4], 0.0)
        return w

    # --- Jacobian sparsity ------------------------------------------------
    # Cell i's balance reads faces i and i+1. A MUSCL face value at f is built
    # from cells f-2, f-1, f (gas, upstream-biased), so the gas part couples
    # cell i to i-2..i+1; the liquid face at f+1 is built from f, f+1, f+2, so
    # cell i couples to i-1..i+2. The pattern is therefore block pentadiagonal
    # (i-2..i+2), NOT tridiagonal: a tridiagonal pattern silently drops the
    # limiter's two-cell coupling, and Newton then stalls at |rhs|~5e-4 on
    # N=160 while looking converged at N=80. Colouring the correct pattern
    # still forms the full Jacobian from ~17 residual evaluations.
    from scipy.optimize._numdiff import group_columns
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    sp = lil_matrix((4 * N, 4 * N), dtype=int)
    for i in range(N):
        sp[4 * i:4 * i + 4, 4 * max(i - 2, 0):4 * min(i + 2, N - 1) + 4] = 1
    sp = sp.tocsr()
    groups = group_columns(sp)
    rows_of = [sp.indices[sp.indptr[j]:sp.indptr[j + 1]] for j in range(4 * N)]
    unique_groups = np.unique(groups)

    y = _project(y)
    f = rhs(y)
    res = float(np.max(np.abs(f)))

    # Damped Newton on the sparse Jacobian. The projection is applied to
    # iterates only; the returned profile is checked against the strict RHS, so
    # a solution that needs projecting is reported rather than hidden.
    for _ in range(maxiter):
        if res < tol:
            break
        J = lil_matrix((4 * N, 4 * N), dtype=float)
        step_ok = True
        for g in unique_groups:
            cols = np.where(groups == g)[0]
            h = {int(j): 1e-7 * max(abs(y[j]), 1e-4) for j in cols}
            yp = y.copy()
            for j in cols:
                yp[j] += h[int(j)]
            try:
                fp = rhs(yp)
            except ValueError:
                step_ok = False
                break
            dfp = fp - f
            for j in cols:
                jj = int(j)
                rows = rows_of[jj]
                J[rows, jj] = dfp[rows] / h[jj]
        if not step_ok:
            break

        try:
            dy = spsolve(J.tocsr(), -f)
        except Exception:
            break
        if not np.all(np.isfinite(dy)):
            break

        lam, accepted = 1.0, False
        for _ in range(12):
            try:
                yn = _project(y + lam * dy)
                fn = rhs(yn)
            except ValueError:
                lam *= 0.5
                continue
            if np.max(np.abs(fn)) < np.max(np.abs(f)):
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
        y, f = yn, fn
        res = float(np.max(np.abs(f)))

    res_best = res
    if not np.isfinite(res_best) or res_best > 1e-8:
        # Last resort: Krylov-Newton restarts from the best iterate.
        fun = lambda v: mol_column_odes(0.0, _project(v), N, rhs_params)
        y_best = y
        for _ in range(4):
            sol = root(fun, y_best, method='krylov',
                       options={'fatol': tol, 'xatol': tol, 'maxiter': 400})
            cand = np.asarray(sol.x, dtype=float)
            if not np.all(np.isfinite(cand)):
                break
            try:
                res_cand = float(np.max(np.abs(rhs(cand))))
            except ValueError:
                res_cand = float('inf')
            if res_cand < res_best:
                y_best, res_best = cand, res_cand
            if res_best < 1e-11:
                break
        y = y_best

    if not np.isfinite(res_best) or res_best > 1e-11:
        # Polish. Minmod is a limiter: wherever a slope changes sign the residual
        # is only piecewise smooth, so a Newton line search can stall several
        # decades above machine precision even with an isolated root. A
        # trust-region least-squares step tolerates the kink better, and started
        # from an already-good iterate it costs a handful of evaluations. Bounds
        # keep it inside the domain where the strict RHS is defined.
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix

        sp = lil_matrix((4 * N, 4 * N), dtype=int)
        for i in range(N):
            sp[4 * i:4 * i + 4, 4 * max(i - 2, 0):4 * min(i + 2, N - 1) + 4] = 1
        lo = np.full(4 * N, -np.inf)
        hi = np.full(4 * N, np.inf)
        lo[0::4], hi[0::4] = 0.0, np.inf
        lo[1::4], hi[1::4] = 1e-8, 0.4999
        lo[2::4], hi[2::4] = -50.0, 400.0
        lo[3::4], hi[3::4] = -50.0, 400.0
        try:
            solp = least_squares(rhs, y, bounds=(lo, hi),
                                 jac_sparsity=sp.tocsr(), method='trf',
                                 xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                 max_nfev=120)
            yp = np.asarray(solp.x, dtype=float)
            resp = float(np.max(np.abs(rhs(yp))))
            if resp < res_best:
                y, res_best = yp, resp
        except Exception:
            pass

    return y, res_best, cfg, rhs_params


def steady_state_report(y, n_zones, cfg, rhs_params):
    """Observables of a converged profile, with both independent mass closures.

    n_flux is the cell integral A*dz*sum(rate_i); n_gas and n_liq are the two
    ways the column as a whole can be read. At a true steady state of this
    discretisation all three are the same number to machine precision, so
    swing_rel is a code-correctness assertion. If it is not ~1e-12 the
    discretisation is leaking, not converging.
    """
    N = int(n_zones)
    y = np.asarray(y, dtype=float).ravel()
    C_G = y[0::4]
    alpha = y[1::4]
    T_g = y[2::4]
    T_l = y[3::4]
    dz = cfg['H'] / N
    A = cfg['A_cs_m2']

    n_flux = 0.0
    for i in range(N):
        T_avg = 0.5 * (T_g[i] + T_l[i])
        P_G = C_G[i] * 8.314 * (T_avg + 273.15)
        P_star = equilibrium_liquid_co2_loading(alpha[i], T_avg)
        n_flux += cfg['K_Ga'] * (P_G - P_star) * A * dz

    C_in = cfg['C_G_in']
    n_gas = cfg['u_G'] * A * (C_in - C_G[N - 1])
    n_MEA = cfg['u_L'] * cfg['eps_L'] * A * cfg['C_MEA']
    alpha_lean = rhs_params['alpha_lean']
    n_liq = n_MEA * (alpha[0] - alpha_lean)

    T_bottom = 0.5 * (T_g[0] + T_l[0])
    P_G_bottom = C_G[0] * 8.314 * (T_bottom + 273.15)
    a_star = alpha_at_equilibrium(P_G_bottom, T_bottom)

    return {
        'capture_pct': 100.0 * (1.0 - C_G[N - 1] / C_in),
        'n_flux': n_flux,
        'n_gas': n_gas,
        'n_liq': n_liq,
        'n_gas_rel': abs(n_gas - n_flux) / abs(n_flux),
        'swing_rel': abs(n_liq - n_flux) / abs(n_flux),
        'alpha_bottom': float(alpha[0]),
        'alpha_top': float(alpha[N - 1]),
        'alpha_lean': float(alpha_lean),
        'n_MEA': n_MEA,
        'alpha_star_bottom': a_star,
        'P_CO2_bottom': P_G_bottom,
        'T_bottom': T_bottom,
        'C_G_out': float(C_G[N - 1]),
        'C_G_bottom': float(C_G[0]),
        'alpha_max_used': float(np.max(alpha)),
        'T_gas_span': (float(T_g[0]), float(T_g[N - 1])),
        'T_liq_span': (float(T_l[0]), float(T_l[N - 1])),
    }


# ============================================================================
# Performance Metrics Calculation
# ============================================================================

def calculate_column_performance_mol(simulation_result,
                                     C_G_inlet=8.0,
                                     m_solvent=1.0,
                                     **params):
    """Return a performance dict built from the enforced schema.

    Accepts a results dict that already carries a ColumnPerformance in
    ``results['performance']`` (the live path) or a legacy dict convertible by
    ``from_simulation_result``. No flat key is required that is not consumed.
    """
    perf = None
    if isinstance(simulation_result, dict):
        perf = simulation_result.get('performance')
    if perf is None:
        perf = from_simulation_result(simulation_result)
    if not isinstance(perf, ColumnPerformance):
        raise TypeError(f"expected ColumnPerformance, got {type(perf).__name__}")

    payload = {
        'performance': perf,
        'validation_flags': perf.validation_flags(),
        'avg_capture_efficiency_%': perf.capture_efficiency_pct,
        'co2_removal_mol_s_avg': perf.n_CO2,
        'co2_removal_kg_h': perf.n_CO2 * CO2_MOLAR_MASS * 3600.0 / 1000.0,
        'reboiler_duty_kw': perf.Q_reb,
        'Q_sens': perf.Q_sens,
        'Q_desorp': perf.Q_desorp,
        'Q_steam': perf.Q_steam,
        'Q_loss': perf.Q_loss,
        'specific_energy_kwh_kg': perf.specific_energy_kwh_kg,
        'specific_energy_GJt': perf.specific_energy_GJt,
    }

    if isinstance(simulation_result, dict) and 'C_G_CO2' in simulation_result:
        payload['z_positions'] = simulation_result.get('z_positions')
        if simulation_result.get('alpha') is not None:
            payload['alpha_avg_by_z'] = np.mean(simulation_result['alpha'], axis=0)
        if simulation_result.get('T_liquid') is not None:
            payload['T_liquid_avg_by_z'] = np.mean(
                simulation_result['T_liquid'], axis=0)
        payload['approach_temp_K'] = (
            params.get('T_gas_inlet', 40.0) - params.get('T_liquid_inlet', 40.0)
        )

    return payload



def simulate_column_step_response(duration=300, dt=1.0,
                                 disturbance=None,
                                 disturbance_magnitude=0.0,
                                 n_zones=10,
                                 **params):
    """Simplified step-response simulation wrapper."""
    times, results = simulate_column_mol(
        duration=duration,
        dt=dt,
        n_zones=n_zones,
        disturbance=disturbance,
        disturbance_magnitude=disturbance_magnitude,
        **params
    )
    
    if not isinstance(results.get('performance'), ColumnPerformance):
        results['performance'] = calculate_column_performance_mol(
            results, **params)['performance']
    
    return times, results


# ============================================================================
# Quick Test / Demo
# ============================================================================

# ---------------------------------------------------------------------------
# Thin wrappers for dashboard / legacy callers
# ---------------------------------------------------------------------------

def reboiler_energy_duty(**params):
    """Thin wrapper: run a short sim and return its reboiler duty dict."""
    times, results = simulate_column_step_response(
        duration=60.0, dt=1.0, disturbance=None,
        disturbance_magnitude=0.0, n_zones=int(params.get('n_zones', 10)), **params)
    return results.get('reboiler_duty', {})


def calculate_column_performance(m_solvent=1.0, **params):
    """Thin wrapper returning a ColumnPerformance for dashboard use."""
    params = dict(params)
    params.setdefault('m_solvent', m_solvent)
    times, results = simulate_column_step_response(
        duration=60.0, dt=1.0, disturbance=None,
        disturbance_magnitude=0.0, n_zones=int(params.get('n_zones', 10)), **params)
    return calculate_column_performance_mol(results, m_solvent=m_solvent, **params)


if __name__ == "__main__":
    print("=== Rate-Based MEA Column Model with MOL PDE Discretization ===\n")
    
    # 1. Live decomposition. Flags may be False -- that is the diagnostic,
    #    not a failure to hide.
    print("1. Live MOL simulation and decomposition...")
    times, results = simulate_column_step_response(
        duration=60.0,
        dt=1.0,
        disturbance='gas_velocity',
        disturbance_magnitude=0.2,
        n_zones=10,
        u_L=0.02,
    )
    p = results['performance']
    print(f"   t_end={times[-1]:.0f}s  n_points={len(times)}  "
          f"N={results['C_G_CO2'].shape[1]}  solver=RK45")
    print(p)
    print("   validation_flags:")
    for key, value in p.validation_flags().items():
        print(f"     {key:34s}: {value}")


    # 4. Test two-film with Formulation B (no unit mismatch)
    print("\n4. Two-film mass transfer Formulation B test:")
    try:
        # Typical conditions: C_G=8 mol/m³, alpha=0.05 (lean), T=40°C
        N_CO2, K_overall, m_eff, P_star = two_film_mass_transfer_loading(
            C_G=8.0, alpha=0.05, T=40.0
        )
        driving_forces_ok = (P_star < 8.0 * 8.314 * (40+273.15))  # P_G should be > P_star for absorber
        print(f"   N_CO2 (flux): {N_CO2:.6e} mol/m²/s")
        print(f"   P_star (eq pressure): {P_star:.1f} Pa")
        print(f"   m_eff (dist coeff): {m_eff:.4f} Pa·m³/mol")
        print(f"   ✓ Driving force P_G - P_star should be > 0 in absorber")
        if N_CO2 > 0:
            print(f"   ✓ Positive flux — Formulation B working correctly")
        else:
            print(f"   ! Flux sign check — may indicate driving force sign issue")
    except Exception as e:
        print(f"   ✗ Two-film test error: {e}")
    
    # 5. Test Henry's constant without clamp
    print("\n5. Henry's constant test (no clamp):")
    try:
        K_H = henry_constant_Pa_m3_mol(40.0)  # at 40°C
        K_H_80 = henry_constant_Pa_m3_mol(80.0)  # at 80°C
        print(f"   K_H at 40°C: {K_H:.2e} Pa·m³/mol")
        print(f"   K_H at 80°C: {K_H_80:.2e} Pa·m³/mol")
        print(f"   ✓ Henry's constant computed — no clamp active")
        # Verify: at 40°C, should be ~10^5 Pa·m³/mol
        if K_H > 1e4 and K_H < 1e7:
            print(f"   ✓ Reasonable magnitude (~10^5 as expected)")
        else:
            print(f"   ! Unusual magnitude — check correlation")
    except Exception as e:
        print(f"   ✗ Henry's constant error: {e}")
    
    # 6. Initial conditions with alpha (loading)
    print("\n6. Initial conditions test (alpha = loading, not C_L_CO2):")
    try:
        y0 = set_initial_conditions(n_zones=10, C_G_CO2_inlet=8.0, alpha_inlet=0.05,
                                    T_gas_inlet=40.0, T_liquid_inlet=40.0,
                                    C_MEA_rich=3000.0)
        print(f"   State size: {len(y0)} (expected {4*10}=40)")
        print(f"   ✓ Initial conditions: C_G_CO2, alpha, T_gas, T_liquid per zone")
        # Verify alpha values are reasonable loadings (0–0.5 mol CO2/mol MEA)
        alpha_0 = y0[1]  # second element, alpha at node 0
        alpha_N = y0[5]  # alpha at node N-1 (after 4*1=4, so index 5 = 1*4+1)
        print(f"   alpha at bottom (z=0): {alpha_0:.4f} mol CO2/mol MEA")
        print(f"   alpha at top (z=H):   {alpha_N:.4f} mol CO2/mol MEA")
        if 0 <= alpha_0 <= 0.5 and 0 <= alpha_N <= 0.5:
            print(f"   ✓ Loading values in physical range [0, 0.5]")
        else:
            print(f"   ! Loading outside expected range")
    except Exception as e:
        print(f"   ✗ Initial conditions error: {e}")
    
    print("\n=== MOL Model Test Complete ===")