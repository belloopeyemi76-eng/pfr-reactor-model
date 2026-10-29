# Non-Isothermal Plug Flow Reactor (PFR) Model

A Python model of a plug flow reactor running a first-order, irreversible,
exothermic reaction (A -> B), with optional jacket cooling.

## What it does

Solves the coupled conversion and energy balance ODEs along the reactor
volume:

```
dX/dV = k(T) * C_A0 * (1 - X) / F_A0
dT/dV = [ (-dH_rxn) * k(T) * C_A0 * (1 - X) + Ua * (Ta - T) ] / FCp
```

Three cases are supported:

1. **Isothermal** - reactor held at a fixed temperature. Has a closed-form
   analytical solution, `X(V) = 1 - exp(-k*V/v0)`, used to validate the
   numerical integrator.
2. **Adiabatic** - no heat exchange with a jacket (`Ua = 0`). The
   temperature rise at full conversion must match the theoretical
   adiabatic temperature rise formula, used as a second, independent
   validation check.
3. **Cooled (general case)** - jacket cooling active (`Ua > 0`). This is
   the main design problem: given reaction kinetics, feed conditions,
   and cooling parameters, find the reactor volume required to reach a
   target conversion.

## Numerical method

The model is integrated with SciPy's adaptive RK45 solver. A classical
fourth-order Runge-Kutta integrator is also written from scratch
(`rk4_integrate`) and tested against the exact isothermal solution. Halving
the step size cuts the error by about a factor of 16, which confirms fourth-order
convergence (observed order about 4.0 to 4.1).

## Files

- `pfr_reactor_model.py` - the model itself: ODE system, integrator,
  analytical isothermal solution, adiabatic temperature rise formula,
  and the volume-search routine for the main design problem.
- `test_pfr_reactor_model.py` - validation tests. Checks the numerical
  solution against the analytical isothermal solution, checks the
  adiabatic temperature rise against theory, confirms that the
  hand-written RK4 solver converges at fourth order, and includes basic
  physical sanity checks (monotonic conversion, cooled vs. adiabatic
  comparison).

## Running it

```bash
pip install -r requirements.txt
python pfr_reactor_model.py        # runs the example design case
python test_pfr_reactor_model.py   # runs the validation tests
```

## Example output

```
Cooled PFR design result:
  Volume required for X=0.85: 9.66 L
  Final temperature: 367.58 K

Theoretical adiabatic temperature rise at full conversion: 50.00 K
```

## Reference

Reaction engineering formulation follows the standard non-isothermal
PFR energy balance presented in Fogler, *Elements of Chemical Reaction
Engineering*.
