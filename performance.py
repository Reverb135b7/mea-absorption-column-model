"""Enforced performance schema for MEA absorption column — one source of truth.

This module replaces the scattered dict-based performance computation with a
single dataclass.  It draws a hard line between two kinds of check:

  * mathematical consistency  -> enforced in ``__post_init__`` (raises)
      - energy decomposition closes
      - desorption anchor consistent with the reported CO2 flow
      - m_CO2 > 0 (specific energy is undefined otherwise)
      - every field finite

  * physical plausibility     -> reported by ``validation_flags()`` (never raises)
      - SRD inside the industry range
      - steam/desorption ratio inside the MEA-typical range
      - capture efficiency inside (0, 100]
      - reboiler loss fraction inside 1-6 % of duty
      - solvent flow x loading swing consistent with the CO2 flow
      - provenance of Q_steam and Q_sens

Why the split: an implausible-but-computable live run must be *reportable*.
Rejecting it at construction hides the decomposition, which is exactly the
failure mode this schema exists to prevent.

All quantities are SI-ish:
  kW, mol/s, kg/s, kJ, GJ/t, kWh/kg, degC, Pa, mol/m3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

# ---------------------------------------------------------------------------
# Constants (SI-ish, MEA/CO2/H2O system)
# ---------------------------------------------------------------------------

M_CO2: float = 0.04401      # kg / mol CO2
M_WATER: float = 0.018015   # kg / mol H2O
MEA_MOLAR_MASS: float = 0.06108  # kg / mol MEA (C2H7NO)

# Reaction enthalpies (kJ / mol per CO2)
DH_ABS: float = 85.0        # carbamate formation; desorption needs +85 kJ/mol
DH_VAP_H2O: float = 40.7    # kJ / mol H2O at ~100 degC, ~1 bar

# Design-basis reference: at 3.0 GJ/t SRD and m_CO2 ~ 0.04159 kg/s,
# the reboiler duty is ~125 kW.  Anchor for bounds checking only.
REF_SRD_GJT: float = 3.0
REF_Q_REB_KW: float = 124.8

#: Default MEA mass fraction of the aqueous solvent (30 wt%).
W_MEA_DEFAULT: float = 0.30


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnPerformance:
    """Column-level energy & mass performance.

    Construction only enforces mathematics:

        Q_sens + Q_desorp + Q_steam + Q_loss == Q_reb   (relative tol 1e-6)
        Q_desorp == n_CO2 * DH_ABS                      (relative tol 5%)
        m_CO2 > 0
        all fields finite

    Physical plausibility is NOT a construction precondition -- call
    ``validation_flags()`` and read the flags.  A run that lands at 1.2 GJ/t
    is a diagnostic result, not an exception.
    """

    #: Sensible heat duty, kW  (solvent heating from feed to reboiler inlet)
    Q_sens: float
    #: Desorption heat, kW  (endothermic CO2 stripping, |dH| * mol/s)
    Q_desorp: float
    #: Stripping-steam heat, kW  (enthalpy of vaporized overhead water)
    Q_steam: float
    #: Loss heat (UA * dT), kW  (reboiler shell loss to ambient)
    Q_loss: float
    #: Total reboiler duty, kW  (sum of the four above)
    Q_reb: float
    #: CO2 captured, mol / s
    n_CO2: float
    #: CO2 captured, kg / s  (auto-derived: n_CO2 * M_CO2)
    m_CO2: float = field(repr=False)
    #: Specific energy consumption, GJ / tonne CO2  (industry standard)
    specific_energy_GJt: float
    #: Specific energy consumption, kWh / kg CO2
    specific_energy_kwh_kg: float
    #: CO2 capture efficiency, %  (plausible range checked by flags)
    capture_efficiency_pct: float

    # -- provenance / optional context (feeds validation_flags) -------------
    #: How Q_steam was obtained: "constant:...", "assumed_overhead:...", "state:..."
    steam_source: str = "unknown"
    #: How Q_sens was obtained: "hx_approach_placeholder", "unconfigured:..."
    Q_sens_source: str = "unknown"
    #: Mean liquid temperature at the reboiler node, degC
    T_liquid_reb: float | None = None
    #: Solvent mass flow, kg/s
    m_solvent: float | None = None
    #: Rich / lean CO2 loading, mol CO2 / mol MEA
    alpha_rich: float | None = None
    alpha_lean: float | None = None
    #: Reboiler UA used for Q_loss, W/K
    UA_reboiler_W_K: float | None = None
    #: MEA mass fraction used for loading-swing closure
    w_MEA: float = W_MEA_DEFAULT
    #: Design capture target, %
    capture_target_pct: float = 90.0

    # Class-level constants
    _DH_ABS: ClassVar[float] = DH_ABS
    _M_CO2: ClassVar[float] = M_CO2
    _MIN_SRD_GJT: ClassVar[float] = 2.5
    _MAX_SRD_GJT: ClassVar[float] = 6.0
    _STEAM_RATIO_MIN: ClassVar[float] = 0.2
    _STEAM_RATIO_MAX: ClassVar[float] = 0.6
    _LOSS_FRACTION_MIN: ClassVar[float] = 0.01
    _LOSS_FRACTION_MAX: ClassVar[float] = 0.06
    _DECOMPOSITION_REL_TOL: ClassVar[float] = 1e-6
    _DESC_ANCHOR_REL_TOL: ClassVar[float] = 0.05

    def __post_init__(self) -> None:
        # -- every field finite (a NaN run must not pass silently) ----------
        numeric = {
            "Q_sens": self.Q_sens,
            "Q_desorp": self.Q_desorp,
            "Q_steam": self.Q_steam,
            "Q_loss": self.Q_loss,
            "Q_reb": self.Q_reb,
            "n_CO2": self.n_CO2,
            "m_CO2": self.m_CO2,
            "specific_energy_GJt": self.specific_energy_GJt,
            "specific_energy_kwh_kg": self.specific_energy_kwh_kg,
            "capture_efficiency_pct": self.capture_efficiency_pct,
        }
        for name, value in numeric.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} is not finite ({value!r})")

        # -- decomposition closure -----------------------------------------
        parts = self.Q_sens + self.Q_desorp + self.Q_steam + self.Q_loss
        if abs(parts - self.Q_reb) > self._DECOMPOSITION_REL_TOL * max(
            self.Q_reb, 1.0
        ):
            raise ValueError(
                f"decomposition {parts:.6f} != Q_reb {self.Q_reb:.6f} kW "
                f"(relative error {abs(parts - self.Q_reb)/max(parts, 1e-6):.2%})"
            )

        # -- desorption anchor: 85 kJ/mol ----------------------------------
        if abs(self.Q_desorp - self.n_CO2 * self._DH_ABS) > self._DESC_ANCHOR_REL_TOL * max(
            self.Q_desorp, 1.0
        ):
            raise ValueError(
                f"Q_desorp {self.Q_desorp:.3f} kW inconsistent with n_CO2 "
                f"{self.n_CO2:.3f} mol/s * {self._DH_ABS} kJ/mol"
            )

        # -- m_CO2 > 0 ------------------------------------------------------
        if self.m_CO2 <= 0:
            raise ValueError("m_CO2 must be > 0 to report specific energy")

    # ------------------------------------------------------------------
    # Plausibility report -- never raises
    # ------------------------------------------------------------------
    def validation_flags(self) -> dict:
        """Physical plausibility of this result.  All checks are advisory.

        A False flag is a finding about the model or the configuration, not a
        reason to suppress the numbers.
        """
        ratio = (
            self.Q_steam / self.Q_desorp if self.Q_desorp > 0 else float("inf")
        )
        loss_fraction = self.Q_loss / self.Q_reb if self.Q_reb > 0 else float("inf")
        anchor_ratio = (
            self.Q_desorp / (self.n_CO2 * self._DH_ABS)
            if self.n_CO2 > 0
            else float("inf")
        )

        flags = {
            "srd_GJt": self.specific_energy_GJt,
            "srd_plausible": bool(
                self._MIN_SRD_GJT < self.specific_energy_GJt < self._MAX_SRD_GJT
            ),
            "steam_ratio": ratio,
            "steam_ratio_plausible": bool(
                self._STEAM_RATIO_MIN < ratio < self._STEAM_RATIO_MAX
            ),
            "steam_source": self.steam_source,
            "q_sens_source": self.Q_sens_source,
            "capture_efficiency_pct": self.capture_efficiency_pct,
            "capture_efficiency_plausible": bool(
                0.0 < self.capture_efficiency_pct <= 100.0
            ),
            "capture_target_pct": self.capture_target_pct,
            "capture_target_met": bool(
                abs(self.capture_efficiency_pct - self.capture_target_pct) <= 2.0
            ),
            "loss_fraction_of_duty": loss_fraction,
            "ua_reboiler_plausible": bool(
                self._LOSS_FRACTION_MIN < loss_fraction < self._LOSS_FRACTION_MAX
            ),
            "decomposition_residual_kW": self.decomposition_residual,
            "desorption_anchor_ratio": anchor_ratio,
            "loading_swing_closure": None,
            "loading_swing_n_CO2_mol_s": None,
            "loading_swing_reason": "not evaluable: m_solvent/loadings missing",
        }

        if (
            self.m_solvent is not None
            and self.alpha_rich is not None
            and self.alpha_lean is not None
            and self.m_solvent > 0
        ):
            n_MEA = self.m_solvent * self.w_MEA / MEA_MOLAR_MASS
            n_from_swing = n_MEA * (self.alpha_rich - self.alpha_lean)
            flags["loading_swing_n_CO2_mol_s"] = n_from_swing
            if self.n_CO2 > 0:
                rel = abs(n_from_swing - self.n_CO2) / self.n_CO2
                flags["loading_swing_closure"] = bool(rel < 0.02)
                flags["loading_swing_reason"] = (
                    "consistent within 2%" if rel < 0.02
                    else f"solvent flow x loading swing implies {n_from_swing:.3f} mol/s, "
                         f"model reports {self.n_CO2:.3f} mol/s ({rel:.1%} off)"
                )
            else:
                flags["loading_swing_closure"] = False
                flags["loading_swing_reason"] = "n_CO2 is zero"

        return flags

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def decomposition_residual(self) -> float:
        """Q_reb - (Q_sens + Q_desorp + Q_steam + Q_loss).  Zero when closure holds."""
        return self.Q_reb - (self.Q_sens + self.Q_desorp + self.Q_steam + self.Q_loss)

    # ------------------------------------------------------------------
    # Human-readable representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = [
            f"capture_efficiency      : {self.capture_efficiency_pct:6.1f} %",
            f"n_CO2_captured          : {self.n_CO2:8.4f} mol/s",
            f"m_CO2_captured          : {self.m_CO2:8.5f} kg/s",
            f"Q_sens                  : {self.Q_sens:7.1f} kW  [{self.Q_sens_source}]",
            f"Q_desorp                : {self.Q_desorp:7.1f} kW  (anchor: n_CO2 x {self._DH_ABS} kJ/mol = {self.n_CO2 * self._DH_ABS:6.2f} kW)",
            f"Q_steam                 : {self.Q_steam:7.1f} kW  [{self.steam_source}]",
            f"Q_loss                  : {self.Q_loss:7.1f} kW",
            "------------------------------------",
            f"Q_reb                   : {self.Q_reb:7.1f} kW  (sum check: {self.decomposition_residual:.2e})",
            f"specific_energy         : {self.specific_energy_GJt:6.3f} GJ/t  = {self.specific_energy_kwh_kg:6.3f} kWh/kg",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy adapter: simulation dict -> ColumnPerformance
# ---------------------------------------------------------------------------


def from_simulation_result(
    simulation_result: dict,
    *,
    design_basis_name: str = "pilot",
) -> ColumnPerformance:
    """Convert a legacy simulation-result dict into a ColumnPerformance.

    Only keys that are actually consumed are required:

    - 'reboiler_duty': 'total_duty_kw', 'sensible_kw', 'desorption_kw',
      'Q_steam', 'loss_kw'
    - 'co2_removal_mol_s_avg': mol/s
    - 'capture_efficiency_pct': %

    Everything else (m_solvent, T_ambient, UA_reboiler, n_H2O_overhead,
    T_reboiler_in, T_feed) is no longer demanded: requiring a key that the
    computation never reads is schema theatre.  The live producer builds
    ColumnPerformance directly and does not use this adapter.

    ``design_basis_name`` is retained for call-site compatibility.
    """

    try:
        reb = simulation_result["reboiler_duty"]
    except KeyError:
        reb = simulation_result  # backward compat: top-level dict

    def _require(key: str, where: str = "reboiler_duty") -> float:
        try:
            return float(reb[key])
        except KeyError:
            raise KeyError(f"{where}['{key}'] missing from simulation result")

    Q_sens = _require("sensible_kw")
    Q_desorp = _require("desorption_kw")
    Q_steam = _require("Q_steam")
    Q_loss = _require("loss_kw")
    Q_reb = float(reb.get("total_duty_kw", Q_sens + Q_desorp + Q_steam + Q_loss))

    try:
        n_CO2 = float(simulation_result["co2_removal_mol_s_avg"])
    except KeyError:
        raise KeyError("simulation_result['co2_removal_mol_s_avg'] missing")

    try:
        capture = float(simulation_result["capture_efficiency_pct"])
    except KeyError:
        raise KeyError("simulation_result['capture_efficiency_pct'] missing")

    m_CO2 = n_CO2 * M_CO2                      # kg/s
    srd_GJt = Q_reb / m_CO2 / 1e3              # kW / (kg/s) -> GJ/t
    srd_kwh_kg = srd_GJt / 3.6

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
        capture_efficiency_pct=capture,
        steam_source="legacy_adapter:unlabelled",
        Q_sens_source="legacy_adapter:unlabelled",
    )


# ---------------------------------------------------------------------------
# Overhead-water helper (used only when a caller wants Q_steam coupled)
# ---------------------------------------------------------------------------


def compute_overhead_water_mol_s(
    T_top_K: float,
    P_strip_Pa: float,
    P_CO2_overhead_Pa: float,
    n_CO2_mol_s: float,
) -> float:
    """Saturated overhead water vapor estimate: n_H2O/n_CO2 ~ P_sat(T)/P_CO2.

    This is an explicit assumption-based estimate, NOT a state of the reduced
    model.  Any Q_steam derived from it must be labelled
    ``steam_source="assumed_overhead"``.
    """

    if not (T_top_K > 0 and P_strip_Pa > 0):
        raise ValueError(
            f"invalid overhead conditions: T_top_K={T_top_K}, P_strip_Pa={P_strip_Pa}"
        )
    import numpy as np

    P_sat = min(1e5 * float(np.exp(20.388 - 5132.0 / T_top_K)), 0.95 * P_strip_Pa)
    P_CO2 = max(P_strip_Pa - P_sat, 1e-6)
    return n_CO2_mol_s * P_sat / P_CO2


# ---------------------------------------------------------------------------
# Self-test: python -m performance
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reference case must satisfy every constructor invariant.
    # Q_desorp = n_CO2 * 85 = 0.945 * 85 = 80.325 kW  (was 95.0 -- 15% off)
    n_ref = 0.945
    Q_desorp_ref = n_ref * DH_ABS          # 80.325 kW
    Q_steam_ref = 20.0
    Q_loss_ref = 4.75                      # 50 W/K * 95 K, ~3.8 % of duty
    Q_reb_ref = REF_Q_REB_KW               # 124.8 kW
    Q_sens_ref = Q_reb_ref - Q_desorp_ref - Q_steam_ref - Q_loss_ref

    ref = ColumnPerformance(
        Q_sens=Q_sens_ref,
        Q_desorp=Q_desorp_ref,
        Q_steam=Q_steam_ref,
        Q_loss=Q_loss_ref,
        Q_reb=Q_reb_ref,
        n_CO2=n_ref,
        m_CO2=n_ref * M_CO2,
        specific_energy_GJt=Q_reb_ref / (n_ref * M_CO2) / 1e3,
        specific_energy_kwh_kg=Q_reb_ref / (n_ref * M_CO2) / 1e3 / 3.6,
        capture_efficiency_pct=90.0,
        steam_source="constant:reference_case",
        Q_sens_source="hx_approach_placeholder",
    )
    print(ref)
    print()
    for key, value in ref.validation_flags().items():
        print(f"{key:34s}: {value}")

    # Adapter round-trip
    mock_result = {
        "reboiler_duty": {
            "sensible_kw": Q_sens_ref,
            "desorption_kw": Q_desorp_ref,
            "Q_steam": Q_steam_ref,
            "loss_kw": Q_loss_ref,
            "total_duty_kw": Q_reb_ref,
        },
        "co2_removal_mol_s_avg": n_ref,
        "capture_efficiency_pct": 90.0,
    }
    perf = from_simulation_result(mock_result)
    print("\nAdapter output:")
    print(perf)
    assert abs(perf.specific_energy_GJt - ref.specific_energy_GJt) < 1e-9
    print("\nself-test OK")
