"""
Validation tests for pfr_reactor_model.py.

Two independent analytical checks are used:
  1. Isothermal case: the numerical conversion profile must match the
     closed-form X(V) = 1 - exp(-k*V/v0) solution.
  2. Adiabatic case: the temperature rise at full conversion must
     match the theoretical adiabatic temperature rise formula.

Plus basic sanity checks (monotonic conversion, volume-search
consistency).
"""

import numpy as np
from pfr_reactor_model import (
    ReactorParams,
    simulate,
    analytical_isothermal_conversion,
    find_volume_for_conversion,
    adiabatic_temperature_rise,
)


def make_base_params(Ua=0.0, Ta=320.0):
    return ReactorParams(
        C_A0=2.0,
        v0=5.0,
        k0=7.2e10,
        Ea=72_750.0,
        dH_rxn=-50_000.0,
        FCp=1.0e4,
        Ua=Ua,
        Ta=Ta,
    )


def test_isothermal_matches_analytical():
    params = make_base_params()
    T_fixed = 350.0
    V_max = 200.0

    numerical = simulate(params, V_max=V_max, T0=T_fixed, isothermal_T=T_fixed)
    analytical_X = analytical_isothermal_conversion(numerical["V"], params, T_fixed)

    max_error = np.max(np.abs(numerical["X"] - analytical_X))
    assert max_error < 1e-4, (
        f"Isothermal numerical solution deviates from analytical "
        f"solution by {max_error:.2e}, expected < 1e-4"
    )


def test_adiabatic_temperature_rise_matches_theory():
    params = make_base_params(Ua=0.0)  # adiabatic: no jacket heat exchange
    T0 = 330.0
    V_max = 300.0

    profile = simulate(params, V_max=V_max, T0=T0, isothermal_T=None)

    X_final = profile["X"][-1]
    T_final = profile["T"][-1]

    assert X_final > 0.999, (
        f"Reactor did not reach ~full conversion by V_max={V_max} L "
        f"(X_final={X_final:.4f}); increase V_max for this test"
    )

    expected_rise = adiabatic_temperature_rise(params, X=X_final)
    actual_rise = T_final - T0

    rel_error = abs(actual_rise - expected_rise) / expected_rise
    assert rel_error < 1e-3, (
        f"Adiabatic temperature rise mismatch: actual={actual_rise:.3f} K, "
        f"theoretical={expected_rise:.3f} K, rel_error={rel_error:.2e}"
    )


def test_conversion_is_monotonic_increasing():
    params = make_base_params(Ua=200.0, Ta=320.0)
    profile = simulate(params, V_max=150.0, T0=330.0)
    diffs = np.diff(profile["X"])
    assert np.all(diffs >= -1e-9), "Conversion must be non-decreasing along the PFR"


def test_volume_search_reaches_target_conversion():
    params = make_base_params(Ua=200.0, Ta=320.0)
    X_target = 0.85

    result = find_volume_for_conversion(params, X_target=X_target, T0=330.0)

    assert abs(result["X_final"] - X_target) < 1e-6
    assert result["V_required"] > 0.0

    # Cross-check: simulating up to the found volume should reproduce
    # the same conversion.
    profile = simulate(params, V_max=result["V_required"], T0=330.0, n_points=500)
    assert abs(profile["X"][-1] - X_target) < 1e-3


def test_cooled_reactor_runs_cooler_than_adiabatic_at_same_volume():
    # Physical sanity check: with active jacket cooling (Ta well below
    # the adiabatic hot-spot temperature), the reactor temperature at
    # a given volume should be lower than the adiabatic case.
    V_check = 50.0
    T0 = 330.0

    adiabatic_params = make_base_params(Ua=0.0)
    cooled_params = make_base_params(Ua=200.0, Ta=320.0)

    adiabatic_profile = simulate(adiabatic_params, V_max=V_check, T0=T0)
    cooled_profile = simulate(cooled_params, V_max=V_check, T0=T0)

    assert cooled_profile["T"][-1] < adiabatic_profile["T"][-1]


if __name__ == "__main__":
    test_isothermal_matches_analytical()
    test_adiabatic_temperature_rise_matches_theory()
    test_conversion_is_monotonic_increasing()
    test_volume_search_reaches_target_conversion()
    test_cooled_reactor_runs_cooler_than_adiabatic_at_same_volume()
    print("All tests passed.")
