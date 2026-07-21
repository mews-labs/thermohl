# SPDX-FileCopyrightText: 2026 Mews Labs (https://www.mews-labs.com)
# SPDX-FileCopyrightText: 2026 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import itertools
import os.path
import yaml

import numpy as np
import pandas as pd
import pytest

from thermohl import solver

_SCENARIO_FILE = os.path.join("test", "nonreg", "scenario_steady.yaml")

# target temperature for the ampacity computation (deg C)
_TMAX = 80.0


def cable_data(s: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if s in df["conductor"].values:
        return df[df["conductor"] == s].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {s} not found in file {f}.")

def _get_scenario(name: str) -> dict:
    """Build scalar solver inputs for a steady scenario."""
    dp = cable_data(name)
    dct = dp | dict(
        lat=46.0,
        lon=0.0,
        alt=1.0,
        azm=90.0,
        month=6,
        day=21,
        hour=12.0,
        Ta=20.0,
        Pa=1.0e05,
        rh=0.8,
        pr=0.0,
        ws=3.0,
        wa=0.0,
        al=0.8,
        tb=0.1,
        srad=float("nan"),
        I=555.0,
        alpha=0.9,
        epsilon=0.8,
        RDCHigh=3.05e-05,
        RDCLow=2.66e-05,
        THigh=60.0,
        TLow=20.0,
    )
    return dct

# physical [min, max] ranges for the fully-random scenario, one entry per dct
# field. Bounds are taken from src/thermohl/default_uncertainties.yaml where that
# file provides them; fields absent from it (or given as a distribution without
# min/max) use hand-picked ranges. month/day are drawn as integers (they index
# the solar-declination table), every other field is uniform float.
_RANDOM_SIZE = 128
_INTEGER_FIELDS = ("month", "day")
_RANDOM_RANGES = {
    # conductor geometry / physical params
    "D": (0.005, 0.1),
    "d": (0.0, 0.05),
    "A": (2.0e-05, 2.0e-03),
    "a": (0.0, 5.0e-04),
    "RDC20": (1.0e-05, 1.0e-03),
    "m": (0.1, 5.0),
    "c": (500.0, 1000.0),
    "R": (0.02, 0.20),
    "l": (0.7, 1.5),
    "kl": (3.5e-03, 4.1e-03),
    "kq": (5.0e-08, 1.5e-07),
    "km": (0.9, 1.1),
    "ki": (0.0, 1.6E-02),
    # position / geometry
    "lat": (35.0, 55.0),
    "lon": (-4.9, 8.3),
    "alt": (0.0, 3000.0),
    "azm": (0.0, 360.0),
    # calendar (integers)
    "month": (1, 12),
    "day": (1, 31),
    "hour": (0.0, 24.0),
    # weather
    "Ta": (-40.0, 50.0),
    "Pa": (0.870e05, 1.050e05),
    "rh": (0.0, 1.0),
    "pr": (0.0, 0.05),
    "ws": (0.0, 100.0),
    "wa": (0.0, 360.0),
    "al": (0.0, 1.0),
    "tb": (0.0, 1.0),
    # "srad": (0.0, 0.0),
    # load / material
    "I": (0.0, 999.0),
    "alpha": (0.23, 0.93),
    "epsilon": (0.13, 0.83),
    "RDCHigh": (2.0e-05, 8.0e-05),
    "RDCLow": (2.0e-05, 8.0e-05),
    "THigh": (55.0, 65.0),
    "TLow": (15., 25.0),
}

def _get_scenario_random(seed: int) -> dict:
    """Build a fully-random batch of solver inputs (uniform per field, fixed seed)."""
    rng = np.random.default_rng(seed)
    dct = {}
    for field, (lo, hi) in _RANDOM_RANGES.items():
        if field in _INTEGER_FIELDS:
            dct[field] = rng.integers(lo, hi + 1, size=_RANDOM_SIZE)
        else:
            dct[field] = rng.uniform(lo, hi, size=_RANDOM_SIZE)
    return dct

def _make_scenarios() -> dict:
    """Build the scenario specs (heat equation, model, conductor)."""
    models = ["cigre", "ieee", "rte", "olla"]
    conductors = ["ASTER600", "CROCUS400"]
    heat_equations = ["1t", "3t"]

    scenario = {}
    for heat_equation, model, conductor in itertools.product(heat_equations, models, conductors):
        scenario[f"{heat_equation}-{model}-{conductor}"] = {
            "heat_equation": heat_equation,
            "model": model,
            "conductor": conductor,
        }

    # fully-random array batch, one per heat equation x model (fixed seed each)
    for seed, (heat_equation, model) in enumerate(itertools.product(heat_equations, models)):
        scenario[f"{heat_equation}-{model}-random"] = {
            "heat_equation": heat_equation,
            "model": model,
            "seed": seed,
        }

    return scenario

def _run_scenario(s: dict):
    """Build the solver from a scenario spec, run steady temperature + ampacity.

    Shared by the reference generator and the non-reg test so both use the
    exact same setup. Returns ``(res_temperature, res_intensity)``.
    """
    dc = _get_scenario_random(s["seed"]) if "seed" in s else _get_scenario(s["conductor"])
    slv = solver._factory(dc, heateq=s["heat_equation"], model=s["model"])

    if s["heat_equation"] == "3t":
        # the 1t solve is robust; use it as initial guess for the 3t surface and
        # core temperatures to help the quasi-Newton solver converge
        guess = (
            solver._factory(dc, heateq="1t", model=s["model"])
            .steady_temperature(return_power=False)[solver.Solver.Names.temp]
            .to_numpy()
        )
        res_temperature = slv.steady_temperature(Tsg=guess, Tcg=guess, return_power=False)
    else:
        res_temperature = slv.steady_temperature(return_power=False)

    res_intensity = slv.steady_intensity(T=_TMAX, return_power=False)

    return res_temperature, res_intensity

# yaml field names for the stored reference temperatures, per heat equation
_TEMP_FIELDS = {
    "1t": {"temperature": solver.Solver.Names.temp},
    "3t": {
        "surface_temperature": solver.Solver.Names.tsurf,
        "average_temperature": solver.Solver.Names.tavg,
        "core_temperature": solver.Solver.Names.tcore,
    },
}

def _skip_ampacity(s: dict) -> bool:
    """The 3t ampacity solver is multistable on the fully-random inputs: it
    converges to different roots depending on platform-specific float behaviour,
    so its result is not reproducible across OSes. Skip it for the random 3t
    scenarios only; every other scenario has a single, portable ampacity solution.
    """
    return "seed" in s and s["heat_equation"] == "3t"

def _gen_scenario_steady():
    """Generate scenarios, compute results and write the yaml non-reg reference."""
    scenario = _make_scenarios()

    for s in scenario.values():
        res_temperature, res_intensity = _run_scenario(s)
        for field, key in _TEMP_FIELDS[s["heat_equation"]].items():
            s[field] = res_temperature[key].tolist()
        if not _skip_ampacity(s):
            s["max_intensity"] = res_intensity[solver.Solver.Names.transit].tolist()

    yaml.dump(scenario, open(_SCENARIO_FILE, "w"))

_REFERENCE = yaml.safe_load(open(_SCENARIO_FILE, "r")) if os.path.exists(_SCENARIO_FILE) else {}

@pytest.mark.parametrize("sid", list(_REFERENCE), ids=list(_REFERENCE))
def test_scenario_steady(sid):
    """Non-reg test for steady computations, one case per scenario."""
    atol = 1.0e-06
    s = _REFERENCE[sid]

    res_temperature, res_intensity = _run_scenario(s)

    for field, key in _TEMP_FIELDS[s["heat_equation"]].items():
        assert np.allclose(res_temperature[key], s[field], atol=atol)
    if not _skip_ampacity(s):
        assert np.allclose(res_intensity[solver.Solver.Names.transit], s["max_intensity"], atol=atol)