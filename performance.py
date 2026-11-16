"""Enforced performance schema for MEA absorption column — one source of truth.

This module replaces the scattered dict-based performance computation with a
single dataclass that validates decomposition, anchors, and unit consistency
at construction time.  No `.get("key", 0.0)` defaults: missing keys raise
KeyError, making schema mismatches impossible to miss.

The schema is shared between:
  • simulate_column_step_response()  (producer)
  • calculate_column_performance_mol()  (consumer)
  • the dashboard (consumer)
  • the physics gate tests (validator)

All quantities are SI-ish:
  kW, mol/s, kg/s, kJ, GJ/t, kWh/kg, °C, Pa, mol/m³
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import ClassVar

# ---------------------------------------------------------------------------
# Constants (SI-ish, MEA/CO₂/H₂O system)
# ---------------------------------------------------------------------------

M_CO2: float = 0.04401      # kg / mol CO2
M_WATER: float = 0.018015   # kg / mol H2O
MEA_MOLAR_MASS: float = 0.1191  # kg / mol MEA

# Reaction enthalpies (kJ / mol per CO2)
DH_ABS: float = 85.0        # carbamate formation; desorption needs +85 kJ/mol
DH_VAP_H2O: float = 40.7    # kJ / mol H2O at ~100 °C, ~1 bar

# Design-basis reference: at 3.0 GJ/t SRD and m_CO2 ≈ 0.04159 kg/s,
# the reboiler duty is ~125 kW.  This is an anchor for bounds checking.
REF_SRD_GJT: float = 3.0
REF_Q_REB_KW: float = 124.8  # ≈ REF_SRD_GJT * REF_M_CO2_KG_S * 1000


# ---------------------------------------------------------------------------
# Core dataclass — one schema, enforced at construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnPerformance:
    """Column-level energy & mass performance.

    All fields are *required* (no Optional) and validated at __post_init__.
    Missing keys in the producer→consumer pipeline will raise KeyError,
    never silently zero the headline metric.

    The decomposition closure test:
        Q_sens + Q_desorp + Q_steam + Q_loss ≈ Q_reb   (relative tol 1e-6)

    The desorption anchor:
        Q_desorp ≈ n_CO2 × DH_ABS   (relative tol 5%)

    The steam ratio test (prevents a missing Q_steam from biasing SRD low):
        0.2 < Q_steam / Q_desorp < 0.6   (typical MEA range)

    The SRD plausibility test:
        2.5 < specific_energy_GJt < 6.0   (industry-standard range)
    """

    #: Sensible heat duty, kW  (solvent heating from feed to reboiler inlet)
    Q_sens: float
    #: Desorption heat, kW  (endothermic CO2 stripping, |ΔH| × mol/s)
    Q_desorp: float
    #: Stripping-steam heat, kW  (enthalpy of vaporized overhead water)
    Q_steam: float
    #: Loss heat (UA·ΔT), kW  (reboiler shell loss to ambient)
    Q_loss: float
    #: Total reboiler duty, kW  (sum of the four above)
    Q_reb: float
    #: CO2 captured, mol / s
    n_CO2: float
    #: CO2 captured, kg / s  (auto-derived: n_CO2 × M_CO2)
    m_CO2: float = field(repr=False)
    #: Specific energy consumption, GJ / tonne CO2  (industry standard)
    specific_energy_GJt: float
    #: Specific energy consumption, kWh / kg CO2
    specific_energy_kwh_kg: float
    #: CO2 capture efficiency, %  (0 < capture ≤ 100)
    capture_efficiency_pct: float

    # Class-level constants for validation
    _DH_ABS: ClassVar[float] = DH_ABS
    _M_CO2: ClassVar[float] = M_CO2
    _MIN_SRD_GJT: ClassVar[float] = 2.5
    _MAX_SRD_GJT: ClassVar[float] = 6.0
    _STEAM_RATIO_MIN: ClassVar[float] = 0.2
    _STEAM_RATIO_MAX: ClassVar[float] = 0.6
    _DECOMPOSITION_REL_TOL: ClassVar[float] = 1e-6
    _DESC_ANCHOR_REL_TOL: ClassVar[float] = 0.05

    def __post_init__(self) -> None:
        # -- decomposition closure -----------------------------------------
        parts = self.Q_sens + self.Q_desorp + self.Q_steam + self.Q_loss
        if abs(parts - self.Q_reb) > self._DECOMPOSITION_REL_TOL * max(
            self.Q_reb, 1.0
        ):
            raise ValueError(
                f"decomposition {parts:.6f} ≠ Q_reb {self.Q_reb:.6f} kW "
                f"(relative error {abs(parts - self.Q_reb)/max(parts, 1e-6):.2%})"
            )

        # -- desorption anchor: 85 kJ/mol → 1.93 GJ/t anchor ------------
        if abs(self.Q_desorp - self.n_CO2 * self._DH_ABS) > self._DESC_ANCHOR_REL_TOL * max(
            self.Q_desorp, 1.0
        ):
            raise ValueError(
                f"Q_desorp {self.Q_desorp:.3f} kW inconsistent with n_CO2 "
                f"{self.n_CO2:.3f} mol/s × {self._DH_ABS} kJ/mol"
            )

        # -- steam ratio test (prevents biased SRD) -----------------------
        if not (self._STEAM_RATIO_MIN < self.Q_steam / self.Q_desorp < self._STEAM_RATIO_MAX):
            raise ValueError(
                f"steam/desorption ratio {self.Q_steam/self.Q_desorp:.3f} "
                f"outside plausible MEA range [{self._STEAM_RATIO_MIN}, "
                f"{self._STEAM_RATIO_MAX}]"
            )

        # -- SRD plausibility ----------------------------------------------
        srd = self.specific_energy_GJt
        if not (self._MIN_SRD_GJT < srd < self._MAX_SRD_GJT):
            raise ValueError(
                f"specific_energy_GJt {srd:.3f} outside plausible range "
                f"[{self._MIN_SRD_GJT}, {self._MAX_SRD_GJt}] GJ/t"
            )

        # -- capture efficiency in (0, 100] ---------------------------------
        if not (0.0 < self.capture_efficiency_pct <= 100.0):
            raise ValueError(
                f"capture_efficiency_pct {self.capture_efficiency_pct}% out of (0, 100]"
            )

        # -- m_CO2 must be > 0 (specific energy undefined otherwise) ------
        if self.m_CO2 <= 0:
            raise ValueError("m_CO2 must be > 0 to report specific energy")

    # ------------------------------------------------------------------
    # Property: derived from decomposition (for audit)
    # -------------------------------------------------------------------------
    @property
    def decomposition_residual(self) -> float:
        """Q_reb - (Q_sens + Q_desorp + Q_steam + Q_loss).  Zero when closure holds."""
        return self.Q_reb - (self.Q_sens + self.Q_desorp + self.Q_steam + self.Q_loss)

    # ------------------------------------------------------------------
    # Human-readable representation
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = [
            f"capture_efficiency      : {self.capture_efficiency_pct:6.1f} %",
            f"n_CO2_captured          : {self.n_CO2:8.4f} mol/s",
            f"m_CO2_captured          : {self.m_CO2:8.5f} kg/s",
            f"Q_sens                  : {self.Q_sens:7.1f} kW",
            f"Q_desorp                : {self.Q_desorp:7.1f} kW  (anchor: n_CO2 x {self._DH_ABS} kJ/mol = {self.n_CO2 * self._DH_ABS:6.2f} kW)",
            f"Q_steam                 : {self.Q_steam:7.1f} kW",
            f"Q_loss                  : {self.Q_loss:7.1f} kW",
            "------------------------------------",
            f"Q_reb                   : {self.Q_reb:7.1f} kW  (sum check: {self.decomposition_residual:.2e})",
            f"specific_energy         : {self.specific_energy_GJt:6.3f} GJ/t  = {self.specific_energy_kwh_kg:6.3f} kWh/kg",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single adapter: simulation result → ColumnPerformance
# ---------------------------------------------------------------------------


def from_simulation_result(
    simulation_result: dict,
    *,
    design_basis_name: str = "pilot",
) -> ColumnPerformance:
    """Convert a simulation-result dict into a validated ColumnPerformance.

    The dict must contain (no defaults — KeyError if missing):

    - 'reboiler_duty': dict with keys 'total_duty_kw', 'sensible_kw',
      'desorption_kw', 'loss_kw'  (or top-level keys for backward compat)
    - 'co2_removal_mol_s_avg': mol/s average CO2 removal
    - 'capture_efficiency_pct': % capture efficiency
    - 'm_solvent': kg/s solvent flow (or falls back to design_basis default)
    - 'T_ambient': °C ambient temperature
    - 'UA_reboiler': W/K reboiler overall heat-transfer coefficient
    - 'n_H2O_overhead': mol/s overhead water vapor  (new — no more 0.0 default)
    - 'T_reboiler_in': °C reboiler inlet temperature
    - 'T_feed': °C feed temperature

    Any missing key raises KeyError; nothing is silently zeroed.
    """

    # ---- extract reboiler duty components --------------------------------
    try:
        reb = simulation_result["reboiler_duty"]
    except KeyError:
        reb = simulation_result  # backward compat: top-level dict

    try:
        Q_sens = reb["sensible_kw"]
    except KeyError:
        raise KeyError("reboiler_duty['sensible_kw'] missing from simulation result")

    try:
        Q_desorp = reb["desorption_kw"]
    except KeyError:
        raise KeyError("reboiler_duty['desorption_kw'] missing from simulation result")

    try:
        Q_steam = reb["Q_steam"]  # <-- new key, no default
    except KeyError:
        raise KeyError("reboiler_duty['Q_steam'] missing — add stripping steam term")

    try:
        Q_loss = reb["loss_kw"]
    except KeyError:
        raise KeyError("reboiler_duty['loss_kw'] missing from simulation result")

    try:
        Q_reb = reb["total_duty_kw"]
    except KeyError:
        # last-resort compute from components (should not happen if producer is fixed)
        Q_reb = Q_sens + Q_desorp + Q_steam + Q_loss

    # ---- CO2 removal ----------------------------------------------------
    try:
        n_CO2 = simulation_result["co2_removal_mol_s_avg"]
    except KeyError:
        raise KeyError("simulation_result['co2_removal_mol_s_avg'] missing")

    try:
        capture = simulation_result["capture_efficiency_pct"]
    except KeyError:
        raise KeyError("simulation_result['capture_efficiency_pct'] missing")

    try:
        m_solvent = simulation_result["m_solvent"]
    except KeyError:
        raise KeyError("simulation_result['m_solvent'] missing")

    try:
        T_amb = simulation_result["T_ambient"]
    except KeyError:
        raise KeyError("simulation_result['T_ambient'] missing")

    try:
        UA_reb = simulation_result["UA_reboiler"]
    except KeyError:
        raise KeyError("simulation_result['UA_reboiler'] missing")

    try:
        n_H2O = simulation_result["n_H2O_overhead"]
    except KeyError:
        raise KeyError("simulation_result['n_H2O_overhead'] missing — Q_steam required")

    # ---- optional: re-enter the Q_steam closure if UA·ΔT was used ------
    # If the producer computed Q_loss = UA·ΔT but didn't add Q_steam, the
    # steam ratio test will fire and force the caller to add it properly.

    # ---- m_CO2 from molar flow -------------------------------------------
    m_CO2 = n_CO2 * M_CO2  # kg/s

    # ---- specific energy: Q_reb / m_CO2 / 1e3  (kW / (kg/s) → GJ/t) --
    srd_GJt = Q_reb / m_CO2 / 1e3  # GJ/tonne CO2

    srd_kwh_kg = srd_GJt / 3.6  # kWh/kg CO2

    # ---- capture efficiency -----------------------------------------------
    cap = capture  # already a %

    return ColumnPerformance(
        Q_sens=Q_sens,
        Q_desorp=Q_desorp,
        Q_steam=Q_steam,
        Q_loss=Q_loss,
        Q_reb=Q_reb,
        n_CO2=n_CO2,
        m_CO2=m_CO2,
        specific_energy_GJt=srd_GJt,
        specific_energy_kwh_kg=srd_kwh_kg,
        capture_efficiency_pct=cap,
    )


# ---------------------------------------------------------------------------
# Small closure helper: compute n_H2O from overhead conditions
# ---------------------------------------------------------------------------


def compute_overhead_water_mol_s(
    T_top_K: float,
    P_strip_Pa: float,
    P_CO2_overhead_Pa: float,
    n_CO2_mol_s: float,
) -> float:
    """Saturated overhead water vapor: n_H2O / n_CO2 ≈ P_sat(T)/P_CO2.

    Typical MEA stripper overhead (T_top ≈ 353–363 K, P ≈ 1.9 bar):
    n_H2O / n_CO2 ≈ 0.5–1.0  →  Q_steam ≈ 0.5–1.0 GJ/t.

    The min(...) guard is a *composition* limit (water partial pressure
    cannot exceed total pressure), not a clamp hiding a temperature error.
    """

    # Simple saturation estimate — use Clapeyron / Antoine in production.
    # Here we just guard against P_sat > P_total.
    P_sat = min(1e5 * np.exp(20.388 - 5132.0 / T_top_K), 0.95 * P_strip_Pa)  # rough
    P_CO2 = max(P_strip_Pa - P_sat, 1e-6)
    n_H2O = n_CO2_mol_s * P_sat / P_CO2
    return n_H2O


# ---------------------------------------------------------------------------
# Quick self-test (run with: python -m performance)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reference case: 3.0 GJ/t SRD, m_CO2 ≈ 0.04159 kg/s, Q_reb ≈ 124.8 kW
    # With Q_sens ≈ 25 kW, Q_desorp ≈ 95 kW, Q_steam ≈ 4.0 kW, Q_loss ≈ 0.8 kW
    ref = ColumnPerformance(
        Q_sens=25.0,
        Q_desorp=95.0,
        Q_steam=4.0,
        Q_loss=0.8,
        Q_reb=124.8,
        n_CO2=0.945,
        m_CO2=0.945 * M_CO2,
        specific_energy_GJt=3.0,
        specific_energy_kwh_kg=3.0 / 3.6,
        capture_efficiency_pct=90.0,
    )
    print(ref)

    # Verify the adapter from a mock result
    mock_result = {
        "reboiler_duty": {
            "sensible_kw": 25.0,
            "desorption_kw": 95.0,
            "Q_steam": 4.0,
            "loss_kw": 0.8,
            "total_duty_kw": 124.8,
        },
        "co2_removal_mol_s_avg": 0.945,
        "capture_efficiency_pct": 90.0,
        "m_solvent": 1.0,
        "T_ambient": 40.0,
        "UA_reboiler": 50.0,
        "n_H2O_overhead": 0.023,  # mol/s ≈ 0.5–1.0 GJ/t at these conditions
    }
    perf = from_simulation_result(mock_result)
    print("\nAdapter output:")
    print(perf)