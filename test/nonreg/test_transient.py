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

_SCENARIO_FILE = os.path.join("test", "nonreg", "scenario_transient.yaml")


def cable_data(s: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if s in df["conductor"].values:
        return df[df["conductor"] == s].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {s} not found in file {f}.")

def _get_scenario_default(
        name:str,
        I0: float,
        If: float,
        u0: float,
        uf: float,
        t0: float,
        tf: float,
        nt: int
):
    # conductor data
    dp = cable_data(name)

    # time vector
    t = np.linspace(t0, tf, nt)

    # transit
    I = np.zeros_like(t)
    I[t < 0.0] = I0
    I[t >= 0.0] = If

    # wind speed
    u = np.zeros_like(t)
    u[t < 0.0] = u0
    u[t >= 0.0] = uf

    # solver input
    # dct = dp | dict(
    #     latitude=46.0,
    #     longitude=0.0,
    #     altitude=1.0,
    #     cable_azimuth=90.0,
    #     datetime_utc=np.datetime64("2025-06-21T00:00:00"),
    #     measured_global_radiation=np.nan,
    #     solar_irradiance=np.nan,
    #     ambient_temperature=20.0,
    #     ambient_pressure=1.0e5,
    #     relative_humidity=0.8,
    #     precipitation_rate=0.0,
    #     wind_speed=3.0,
    #     wind_azimuth=0.0,
    #     nebulosity=np.nan,
    #     albedo=0.8,
    #     turbidity=0.1,
    #     transit=555.0,
    #     solar_absorptivity=0.9,
    #     emissivity=0.8,
    # )

    dct = dp | dict(
        lat=46.0,
        lon=0.0,
        alt=1.0,
        azm=90.0,
        month=6,
        day=21,
        hour=0.0,
        Ta=20.0,
        Pa = 1.0e05,
        rh = 0.8,
        pr = 0.0,
        ws=3.0,
        wa=0,
        al = 0.8,
        tb = 0.1,
        srad = float("nan"),
        I=555.0,
        alpha=0.9,
        epsilon=0.8,
        RDCHigh = 3.05e-05,
        RDCLow = 2.66e-05,
        THigh = 60.0,
        TLow = 20.0,
    )

    Ta = None
    wa = None

    return dct, t, I, Ta, u, wa

def _get_scenario_enhanced(
        name:str,
        key
):
    """Generate transient scenarios with key in ['A'-'H']."""
    I0 = 222.0
    If = 888.0
    u0 = 1.0
    uf = 3.0
    t0 = -300
    tf = 2700.0
    nt = 301

    dct, t, I, Ta, u, wa = _get_scenario_default(name, I0, If, u0, uf, t0, tf, nt)

    ix = t >= 0.0
    ft = 2 * np.pi / 700.0

    if key == "A":
        # step transit
        u = np.ones_like(t) * u0

    elif key == "B":
        # step wind speed
        I = np.ones_like(t) * I0

    elif key == "C":
        # step transit and wind speed
        pass

    elif key == "D":
        # step transit and wind speed and sun
        dct["hour"] = 12.0

    elif key == "E":
        # osc transit
        I[ix] = I0 + 0.5 * (If - I0) * np.sin(ft * t[ix])
        u = np.ones_like(t) * u0

    elif key == "F":
        # osc wind speed + sun
        I = np.ones_like(t) * I0
        u[ix] = u0 + 0.3 * (uf - u0) * np.sin(ft * t[ix])
        dct["hour"] = 12.0

    elif key == "G":
        # osc wind angle
        I = np.ones_like(t) * I0
        u = np.ones_like(t) * u0
        wa = np.ones_like(t) * dct["wa"]
        wa[ix] += 30.0 * np.sin(ft * t[ix])

    elif key == "H":
        # osc transit and wind speed
        I[ix] = I0 + 0.5 * (If - I0) * np.sin(ft * t[ix])
        u[ix] = u0 + 0.3 * (uf - u0) * np.sin(ft * t[ix])

    else:
        raise ValueError

    dynamic = {"I": I, "ws": u}
    if Ta is not None:
        dynamic["Ta"] = Ta
    if wa is not None:
        dynamic["wa"] = wa

    return dct, t, dynamic

def _make_scenarios() -> dict:
    """Build the scenario specs (heat equation, model, conductor(s), key(s))."""
    models = ["cigre", "ieee", "rte", "olla"]
    conductors = ["ASTER600", "CROCUS400"]
    keys = ["A", "B", "C", "D", "E", "F", "G", "H"]
    heat_equations = ["1t", "3t"]

    scenario = {}

    # group 01: single-block, 1t, every model x conductor x key combo
    for model, conductor, key in itertools.product(models, conductors, keys):
        scenario[f"01-{model}-{conductor}-{key}"] = {
            "heat_equation": "1t",
            "model": model,
            "conductor": conductor,
            "key": key,
        }

    # group 02: multi-block, every heat equation x model combo
    for heat_equation, model in itertools.product(heat_equations, models):
        scenario[f"02-{heat_equation}-{model}"] = {
            "heat_equation": heat_equation,
            "model": model,
            "conductor": len(keys) * ["ASTER600"] + len(keys) * ["CROCUS400"],
            "key": keys + keys,
        }

    return scenario

def _build_inputs(s: dict):
    """Build solver inputs (dc, t, dynamic) for a scenario spec.

    A scenario is a single block when ``conductor`` is a scalar, or several
    blocks (stacked along a second axis) when it is a list.
    """
    block = isinstance(s["conductor"], list)

    if block:
        parts = [_get_scenario_enhanced(c, k) for c, k in zip(s["conductor"], s["key"])]
        dc = {k: [p[0][k] for p in parts] for k in parts[0][0]}
        t = parts[0][1]
        dynamic = {k: np.stack([p[2][k] for p in parts]).T for k in parts[0][2]}
    else:
        dc, t, dynamic = _get_scenario_enhanced(s["conductor"], key=s["key"])

    return dc, t, dynamic, block

def _run_scenario(s: dict):
    """Build the solver from a scenario spec, run steady + transient.

    Shared by the reference generator and the non-reg test so both use the
    exact same setup. Returns ``(t, res_transient)``.
    """
    dc, t, dynamic, block = _build_inputs(s)
    slv = solver._factory(dc, heateq=s["heat_equation"], model=s["model"])

    # initial conditions: first time-step of each dynamic input
    idx = (0, slice(None)) if block else 0
    for name in ("I", "ws", "Ta", "wa"):
        if name in dynamic:
            slv.args[name] = dynamic[name][idx]

    slv.update()
    res_steady = slv.steady_temperature()

    if s["heat_equation"] == "1t":
        res_transient = slv.transient_temperature(
            t,
            T0=res_steady["t"],
            dynamic=dynamic,
            return_power=False,
        )
    elif s["heat_equation"] == "3t":
        res_transient = slv.transient_temperature(
            t,
            Ts0=res_steady["t_surf"],
            Tc0=res_steady["t_core"],
            dynamic=dynamic,
            return_power=False,
        )
    else:
        raise ValueError(f"unknown heat_equation {s['heat_equation']!r}")

    return t, res_transient

# yaml field names for the stored reference temperatures, per heat equation
_TEMP_FIELDS = {
    "1t": {"temperature": solver.Solver.Names.temp},
    "3t": {
        "surface_temperature": solver.Solver.Names.tsurf,
        "average_temperature": solver.Solver.Names.tavg,
        "core_temperature": solver.Solver.Names.tcore,
    },
}

def _gen_scenario_transient():
    """Generate scenarios, compute results and write the yaml non-reg reference."""
    scenario = _make_scenarios()

    for s in scenario.values():
        t, res = _run_scenario(s)
        s["time"] = t[::10].tolist()
        for field, key in _TEMP_FIELDS[s["heat_equation"]].items():
            s[field] = res[key][::10].tolist()

    yaml.dump(scenario, open(_SCENARIO_FILE, "w"))

_REFERENCE = yaml.safe_load(open(_SCENARIO_FILE, "r"))

@pytest.mark.parametrize("sid", list(_REFERENCE), ids=list(_REFERENCE))
def test_scenario_transient(sid):
    """Non-reg test for transient computations, one case per scenario."""
    atol = 1.0e-06
    s = _REFERENCE[sid]

    _, res = _run_scenario(s)

    for field, key in _TEMP_FIELDS[s["heat_equation"]].items():
        assert np.allclose(res[key][::10], s[field], atol=atol)
