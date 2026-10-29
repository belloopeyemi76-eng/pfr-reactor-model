"""
Non-isothermal plug flow reactor (PFR) model for a first-order,
irreversible, exothermic reaction A -> B, with optional jacket cooling.

Governing equations (steady state, constant volumetric flow v0,
constant mixture heat capacity flow FCp = F_A0 * Cp_mix):

    dX/dV = k(T) * C_A0 * (1 - X) / F_A0

    dT/dV = [ (-dH_rxn) * k(T) * C_A0 * (1 - X) + Ua * (Ta - T) ] / FCp

where
    X       conversion of A                         [-]
    T       reactor temperature                      [K]
    V       reactor volume (independent variable)    [L]
    k(T)    Arrhenius rate constant = k0 * exp(-Ea/(R*T))   [1/s]
    C_A0    inlet concentration of A                 [mol/L]
    F_A0    molar feed rate of A = C_A0 * v0          [mol/s]
    dH_rxn  heat of reaction (negative, exothermic)   [J/mol]
    Ua      heat transfer coefficient x area per volume [J/(s*K*L)]
    Ta      coolant/jacket temperature                [K]
    FCp     thermal capacity flow rate                [J/(s*K)]

This mirrors the standard energy-balance form used for non-isothermal
PFR design (Fogler, "Elements of Chemical Reaction Engineering").

Three cases are implemented:
  1. Isothermal   -> has a closed-form analytical solution, used to
                     validate the numerical integrator.
  2. Adiabatic     -> Ua = 0. The outlet temperature rise at full
                     conversion must match the adiabatic temperature
                     rise formula, used as a second validation check.
  3. Cooled        -> the general, non-isothermal case with jacket
                     cooling (Ua > 0). This is the main design
                     problem: find the reactor volume needed to reach
                     a target conversion.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314  # J/(mol*K)


@dataclass
class ReactorParams:
    C_A0: float       # mol/L, inlet concentration of A
    v0: float          # L/s, volumetric flow rate
    k0: float          # 1/s, Arrhenius pre-exponential factor
    Ea: float          # J/mol, activation energy
    dH_rxn: float      # J/mol, heat of reaction (negative = exothermic)
    FCp: float         # J/(s*K), thermal capacity flow rate (F_A0 * Cp_mix)
    Ua: float = 0.0    # J/(s*K*L), heat transfer coefficient x area per volume
    Ta: float = 300.0  # K, coolant temperature (ignored if Ua = 0)

    @property
    def F_A0(self) -> float:
        return self.C_A0 * self.v0


def rate_constant(T: float, params: ReactorParams) -> float:
    """Arrhenius rate constant k(T)."""
    return params.k0 * np.exp(-params.Ea / (R_GAS * T))


def pfr_odes(V, state, params: ReactorParams, isothermal_T=None):
    """
    Right-hand side of the PFR ODE system.

    state = [X, T]

    If isothermal_T is given, T is held fixed at that value and the
    energy balance is skipped (used for the isothermal case / the
    analytical-solution check).
    """
    X, T = state
    X = min(max(X, 0.0), 0.999999)  # keep conversion in a sane range

    if isothermal_T is not None:
        T = isothermal_T

    k = rate_constant(T, params)
    rA = k * params.C_A0 * (1.0 - X)  # mol/(L*s), -r_A

    dXdV = rA / params.F_A0

    if isothermal_T is not None:
        dTdV = 0.0
    else:
        dTdV = (
            (-params.dH_rxn) * rA + params.Ua * (params.Ta - T)
        ) / params.FCp

    return [dXdV, dTdV]


def simulate(
    params: ReactorParams,
    V_max: float,
    T0: float,
    isothermal_T: float = None,
    n_points: int = 400,
):
    """
    Integrate the PFR from V = 0 to V = V_max and return the
    conversion and temperature profiles.

    Returns a dict with keys 'V', 'X', 'T'.
    """
    V_eval = np.linspace(0.0, V_max, n_points)
    sol = solve_ivp(
        fun=lambda V, y: pfr_odes(V, y, params, isothermal_T=isothermal_T),
        t_span=(0.0, V_max),
        y0=[0.0, T0],
        t_eval=V_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return {"V": sol.t, "X": sol.y[0], "T": sol.y[1]}


def analytical_isothermal_conversion(V, params: ReactorParams, T: float):
    """
    Closed-form conversion for an isothermal first-order PFR:
        X(V) = 1 - exp(-k(T) * V / v0)
    """
    k = rate_constant(T, params)
    return 1.0 - np.exp(-k * V / params.v0)


def find_volume_for_conversion(
    params: ReactorParams,
    X_target: float,
    T0: float,
    V_search_max: float = 5000.0,
):
    """
    Main design problem: find the reactor volume at which the
    cooled, non-isothermal PFR first reaches X_target.

    Uses an event function so the integrator stops exactly at the
    target conversion rather than relying on a fixed grid.
    """

    def hit_target(V, y):
        return y[0] - X_target

    hit_target.terminal = True
    hit_target.direction = 1

    sol = solve_ivp(
        fun=lambda V, y: pfr_odes(V, y, params, isothermal_T=None),
        t_span=(0.0, V_search_max),
        y0=[0.0, T0],
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        events=hit_target,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    if len(sol.t_events[0]) == 0:
        raise ValueError(
            f"Target conversion {X_target} was not reached within "
            f"V_search_max = {V_search_max} L. Increase V_search_max "
            f"or check reactor parameters."
        )

    V_required = sol.t_events[0][0]
    X_final, T_final = sol.y_events[0][0]
    return {"V_required": V_required, "X_final": X_final, "T_final": T_final}


def adiabatic_temperature_rise(params: ReactorParams, X: float = 1.0) -> float:
    """
    Theoretical adiabatic temperature rise at conversion X:
        dT_ad(X) = (-dH_rxn) * C_A0 * X / Cp_mix_flow_per_unit_feed

    Expressed consistently with FCp (thermal capacity flow rate) and
    F_A0 (molar feed rate of A):
        dT_ad(X) = (-dH_rxn) * F_A0 * X / FCp
    """
    return (-params.dH_rxn) * params.F_A0 * X / params.FCp


def rk4_integrate(params: ReactorParams, V_end: float, T0: float,
                  n_steps: int, isothermal_T=None):
    """
    Fixed-step classical fourth-order Runge-Kutta integrator written
    from scratch, used to check convergence order independently of
    SciPy.

    Returns the final state [X, T] at V = V_end.
    """
    h = V_end / n_steps
    y = np.array([0.0, T0], dtype=float)
    V = 0.0
    for _ in range(n_steps):
        k1 = np.array(pfr_odes(V, y, params, isothermal_T))
        k2 = np.array(pfr_odes(V + h / 2, y + h / 2 * k1, params, isothermal_T))
        k3 = np.array(pfr_odes(V + h / 2, y + h / 2 * k2, params, isothermal_T))
        k4 = np.array(pfr_odes(V + h, y + h * k3, params, isothermal_T))
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        V += h
    return y


if __name__ == "__main__":
    # Example design case: exothermic reaction, jacket-cooled PFR.
    params = ReactorParams(
        C_A0=2.0,          # mol/L
        v0=5.0,            # L/s
        k0=7.2e10,         # 1/s
        Ea=72_750.0,       # J/mol
        dH_rxn=-50_000.0,  # J/mol (exothermic)
        FCp=1.0e4,         # J/(s*K)
        Ua=200.0,          # J/(s*K*L)
        Ta=320.0,          # K
    )

    result = find_volume_for_conversion(params, X_target=0.85, T0=330.0)
    print("Cooled PFR design result:")
    print(f"  Volume required for X=0.85: {result['V_required']:.2f} L")
    print(f"  Final temperature: {result['T_final']:.2f} K")

    dT_ad = adiabatic_temperature_rise(params, X=1.0)
    print(f"\nTheoretical adiabatic temperature rise at full conversion: {dT_ad:.2f} K")
