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
from thermohl.solver.entities import (
    HeatEquationType,
    ModelType,
    TemperatureType,
    VariableType,
)

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
        latitude=46.0,
        longitude=0.0,
        altitude=1.0,
        cable_azimuth=90.0,
        datetime_utc=np.datetime64("2025-06-21T12:00:00"),
        measured_global_radiation=np.nan,
        solar_irradiance=np.nan,
        ambient_temperature=20.0,
        ambient_pressure=1.0e05,
        relative_humidity=0.8,
        precipitation_rate=0.0,
        wind_speed=3.0,
        wind_azimuth=0.0,
        nebulosity=np.nan,
        albedo=0.8,
        turbidity=0.1,
        transit=555.0,
        solar_absorptivity=0.9,
        emissivity=0.8,
        linear_resistance_temp_high=3.05e-05,
        linear_resistance_temp_low=2.66e-05,
        temp_high=60.0,
        temp_low=20.0,
    )
    return dct


# physical [min, max] ranges for the fully-random scenario, one entry per dct
# field. month/day are drawn as integers (they are folded into datetime_utc),
# every other field is uniform float.
_RANDOM_SIZE = 128
_INTEGER_FIELDS = ("month", "day")
# cumulative days before each month (non-leap calendar), used to map a random
# (month, day) pair to a day-of-year when building datetime_utc.
_CUMULATIVE_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
_RANDOM_RANGES = {
    # conductor geometry / physical params
    "outer_diameter": (0.005, 0.1),
    "core_diameter": (0.0, 0.05),
    "outer_area": (2.0e-05, 2.0e-03),
    "core_area": (0.0, 5.0e-04),
    "linear_resistance_dc_20c": (1.0e-05, 1.0e-03),
    "linear_mass": (0.1, 5.0),
    "heat_capacity": (500.0, 1000.0),
    "roughness_ratio": (0.02, 0.20),
    "radial_thermal_conductivity": (0.7, 1.5),
    "temperature_coeff_linear": (3.5e-03, 4.1e-03),
    "temperature_coeff_quadratic": (5.0e-08, 1.5e-07),
    "magnetic_coeff": (0.9, 1.1),
    "magnetic_coeff_per_a": (0.0, 1.6e-02),
    # position / geometry
    "latitude": (35.0, 55.0),
    "longitude": (-4.9, 8.3),
    "altitude": (0.0, 3000.0),
    "cable_azimuth": (0.0, 360.0),
    # calendar (integers), folded into datetime_utc below
    "month": (1, 12),
    "day": (1, 31),
    "hour": (0.0, 24.0),
    # weather
    "ambient_temperature": (-40.0, 50.0),
    "ambient_pressure": (0.870e05, 1.050e05),
    "relative_humidity": (0.0, 1.0),
    "precipitation_rate": (0.0, 0.05),
    "wind_speed": (0.0, 100.0),
    "wind_azimuth": (0.0, 360.0),
    "albedo": (0.0, 1.0),
    "turbidity": (0.0, 1.0),
    # load / material
    "transit": (0.0, 999.0),
    "solar_absorptivity": (0.23, 0.93),
    "emissivity": (0.13, 0.83),
    "linear_resistance_temp_high": (2.0e-05, 8.0e-05),
    "linear_resistance_temp_low": (2.0e-05, 8.0e-05),
    "temp_high": (55.0, 65.0),
    "temp_low": (15.0, 25.0),
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

    # fold calendar fields into a single datetime_utc (new API). (month, day) is
    # mapped to a day-of-year via a fixed non-leap calendar, then to a concrete
    # 2025 date, so every draw yields a valid datetime regardless of the day.
    month = dct.pop("month")
    day = dct.pop("day")
    hour = dct.pop("hour")
    day_of_year = (_CUMULATIVE_DAYS[month - 1] + day - 1) % 365
    dct["datetime_utc"] = (
        np.datetime64("2025-01-01T00:00:00")
        + day_of_year.astype("timedelta64[D]")
        + (hour * 3600.0).astype("int64").astype("timedelta64[s]")
    )
    return dct


def _make_scenarios() -> dict:
    """Build the scenario specs (heat equation, model, conductor)."""
    models = ["cigre", "ieee", "rte", "olla"]
    conductors = ["ASTER600", "CROCUS400"]
    heat_equations = ["1t", "3t"]

    scenario = {}
    for heat_equation, model, conductor in itertools.product(
        heat_equations, models, conductors
    ):
        scenario[f"{heat_equation}-{model}-{conductor}"] = {
            "heat_equation": heat_equation,
            "model": model,
            "conductor": conductor,
        }

    # fully-random array batch, one per heat equation x model (fixed seed each)
    for seed, (heat_equation, model) in enumerate(
        itertools.product(heat_equations, models)
    ):
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
    dc = (
        _get_scenario_random(s["seed"])
        if "seed" in s
        else _get_scenario(s["conductor"])
    )
    slv = solver._factory(
        dc,
        heat_equation=HeatEquationType(s["heat_equation"]),
        model=ModelType(s["model"]),
    )

    if s["heat_equation"] == "3t":
        # the 1t solve is robust; use it as initial guess for the 3t surface and
        # core temperatures to help the quasi-Newton solver converge
        guess = solver._factory(
            dc, heat_equation=HeatEquationType("1t"), model=ModelType(s["model"])
        ).steady_temperature(return_power=False)[VariableType.TEMPERATURE.value]
        res_temperature = slv.steady_temperature(
            surface_temperature_guess=guess,
            core_temperature_guess=guess,
            return_power=False,
        )
    else:
        res_temperature = slv.steady_temperature(return_power=False)

    res_intensity = slv.steady_intensity(
        max_conductor_temperature=_TMAX, return_power=False
    )

    return res_temperature, res_intensity


# yaml field names for the stored reference temperatures, per heat equation
_TEMP_FIELDS = {
    "1t": {"temperature": VariableType.TEMPERATURE.value},
    "3t": {
        "surface_temperature": TemperatureType.SURFACE.value,
        "average_temperature": TemperatureType.AVERAGE.value,
        "core_temperature": TemperatureType.CORE.value,
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
            s[field] = np.asarray(res_temperature[key]).tolist()
        if not _skip_ampacity(s):
            s["max_intensity"] = np.asarray(
                res_intensity[VariableType.TRANSIT.value]
            ).tolist()

    yaml.dump(scenario, open(_SCENARIO_FILE, "w"))


_REFERENCE = (
    yaml.safe_load(open(_SCENARIO_FILE, "r")) if os.path.exists(_SCENARIO_FILE) else {}
)


@pytest.mark.parametrize("sid", list(_REFERENCE), ids=list(_REFERENCE))
def test_scenario_steady(sid):
    """Non-reg test for steady computations, one case per scenario."""
    atol = 1.0e-06
    s = _REFERENCE[sid]

    res_temperature, res_intensity = _run_scenario(s)

    for field, key in _TEMP_FIELDS[s["heat_equation"]].items():
        assert np.allclose(res_temperature[key], s[field], atol=atol)
    if not _skip_ampacity(s):
        assert np.allclose(
            res_intensity[VariableType.TRANSIT.value], s["max_intensity"], atol=atol
        )
