# SPDX-FileCopyrightText: 2026 Mews Labs (https://www.mews-labs.com)
# SPDX-FileCopyrightText: 2026 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os.path
import yaml

import numpy as np
import pandas as pd

from thermohl import solver
from thermohl.solver.entities import (
    HeatEquationType,
    ModelType,
    TemperatureType,
    VariableType,
)


def cable_data(s: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if s in df["conductor"].values:
        return df[df["conductor"] == s].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {s} not found in file {f}.")


def _get_scenario_default(
    name: str, I0: float, If: float, u0: float, uf: float, t0: float, tf: float, nt: int
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
    dct = dp | dict(
        latitude=46.0,
        altitude=1.0,
        cable_azimuth=90.0,
        datetime_utc=np.datetime64("2024-06-21T00:00:00"),
        ambient_temperature=20.0,
        wind_speed=3.0,
        wind_azimuth=0,
        transit=555.0,
        solar_absorptivity=0.9,
        emissivity=0.8,
    )

    Ta = None
    wa = None

    return dct, t, I, Ta, u, wa


def _get_scenario_enhanced(name: str, key):
    """Generate transient scenarios with key in ['A'-'H']."""
    I0 = 222.0
    If = 888.0
    u0 = 1.0
    uf = 3.0
    t0 = -300
    tf = 2700.0
    nt = 3001

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
        wa = np.ones_like(t) * dct["wind_angle"]
        wa[ix] += 30.0 * np.sin(ft * t[ix])

    elif key == "H":
        # osc transit and wind speed
        I[ix] = I0 + 0.5 * (If - I0) * np.sin(ft * t[ix])
        u[ix] = u0 + 0.3 * (uf - u0) * np.sin(ft * t[ix])

    else:
        raise ValueError

    dynamic = {"transit": I, "wind_speed": u}
    if Ta is not None:
        dynamic["ambient_temperature"] = Ta
    if wa is not None:
        dynamic["wind_angle"] = wa

    return dct, t, dynamic


def _gen_scenario_transient():
    """Generate a list of scenario, compute results and write yaml file for non-reg."""

    scenario =  {
        "01a": {
            "heateq": "1t",
            "model": "cigre",
            "conductor": "ASTER600",
            "key": "A",
        },
        "01b": {
            "heateq": "1t",
            "model": "cigre",
            "conductor": "CROCUS400",
            "key": "A",
        },
        "01c": {
            "heateq": "1t",
            "model": "ieee",
            "conductor": "ASTER600",
            "key": "A",
        },
        "01d": {
            "heateq": "1t",
            "model": "ieee",
            "conductor": "CROCUS400",
            "key": "A",
        },
        "01e": {
            "heateq": "1t",
            "model": "rte",
            "conductor": "ASTER600",
            "key": "A",
        },
        "01f": {
            "heateq": "1t",
            "model": "rte",
            "conductor": "CROCUS400",
            "key": "A",
        },
        "01g": {
            "heateq": "1t",
            "model": "olla",
            "conductor": "ASTER600",
            "key": "A",
        },
        "01h": {
            "heateq": "1t",
            "model": "olla",
            "conductor": "CROCUS400",
            "key": "A",
        },

        "02a": {
            "heateq": "1t",
            "model": "cigre",
            "conductor": "ASTER600",
            "key": "A",
        },
        "02b": {
            "heateq": "1t",
            "model": "cigre",
            "conductor": "CROCUS400",
            "key": "A",
        },
        "02c": {
            "heateq": "1t",
            "model": "ieee",
            "conductor": "ASTER600",
            "key": "A",
        },
        "02d": {
            "heateq": "1t",
            "model": "ieee",
            "conductor": "CROCUS400",
            "key": "A",
        },
        "02e": {
            "heateq": "1t",
            "model": "rte",
            "conductor": "ASTER600",
            "key": "A",
        },
        "02f": {
            "heateq": "1t",
            "model": "rte",
            "conductor": "CROCUS400",
            "key": "A",
        },
        "02g": {
            "heateq": "1t",
            "model": "olla",
            "conductor": "ASTER600",
            "key": "A",
        },
        "02h": {
            "heateq": "1t",
            "model": "olla",
            "conductor": "CROCUS400",
            "key": "A",
        },

        "03a": {
            "heateq": "1t",
            "model": "cigre",
            "conductor": ["ASTER600", "CROCUS400", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E", "F"],
        },
        "03b": {
            "heateq": "1t",
            "model": "ieee",
            "conductor": ["ASTER600", "CROCUS400", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E", "F"],
        },
        "03c": {
            "heateq": "1t",
            "model": "rte",
            "conductor": ["ASTER600", "CROCUS400", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E", "F"],
        },
        "03d": {
            "heateq": "1t",
            "model": "olla",
            "conductor": ["ASTER600", "CROCUS400", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E", "F"],
        },
        "04a": {
            "heateq": "3t",
            "model": "cigre",
            "conductor": ["ASTER600", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E"],
        },
        "04b": {
            "heateq": "3t",
            "model": "ieee",
            "conductor": ["ASTER600", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E"],
        },
        "04c": {
            "heateq": "3t",
            "model": "rte",
            "conductor": ["ASTER600", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E"],
        },
        "04d": {
            "heateq": "3t",
            "model": "olla",
            "conductor": ["ASTER600", "ASTER600", "CROCUS400"],
            "key": ["A", "B", "E"],
        }
    }

    for i, s in scenario.items():
        nblocks = len(s["conductor"]) if isinstance(s["conductor"], list) else 1
        block = nblocks > 1

        if block:
            args = [
                _get_scenario_enhanced(s["conductor"][j], s["key"][j])
                for j in range(nblocks)
            ]
            dc = {}
            for k in args[0][0].keys():
                dc[k] = np.array([args[j][0][k] for j in range(nblocks)])
            t = args[0][1]
            dynamic = {}
            for k in args[0][2].keys():
                dynamic[k] = np.stack([args[j][2][k] for j in range(nblocks)]).T
        else:
            dc, t, dynamic = _get_scenario_enhanced(s["conductor"], key=s["key"])

        slv = solver._factory(dc, heat_equation=HeatEquationType(s["heat_equation"]), model=ModelType(s["model"]))

        if block:
            slv.args["transit"] = dynamic["transit"][0, :]
            slv.args["wind_speed"] = dynamic["wind_speed"][0, :]
            if "ambient_temperature" in dynamic:
                slv.args["ambient_temperature"] = dynamic["ambient_temperature"][0, :]
            if "wind_angle" in dynamic:
                slv.args["wind_angle"] = dynamic["wind_angle"][0, :]
        else:
            slv.args["transit"] = dynamic["transit"][0]
            slv.args["wind_speed"] = dynamic["wind_speed"][0]
            if "ambient_temperature" in dynamic:
                slv.args["ambient_temperature"] = dynamic["ambient_temperature"][0]
            if "wind_angle" in dynamic:
                slv.args["wind_angle"] = dynamic["wind_angle"][0]
        slv.update()
        res_steady = slv.steady_temperature(return_power=False)

        s["time"] = t[::100].tolist()

        if s["heat_equation"] == "1t":
            res_transient = slv.transient_temperature(
                t,
                T0=res_steady[VariableType.TEMPERATURE.value],
                dynamic=dynamic,
                return_power=False,
            )
            s[VariableType.TEMPERATURE.value] = res_transient[VariableType.TEMPERATURE.value][
                ::100
            ].tolist()
        elif s["heat_equation"] == "3t":
            res_transient = slv.transient_temperature(
                t,
                surface_temperature_0=res_steady[TemperatureType.SURFACE.value],
                core_temperature_0=res_steady[TemperatureType.CORE.value],
                dynamic=dynamic,
                return_power=False,
            )
            for k in (
                    TemperatureType.SURFACE.value,
                    TemperatureType.AVERAGE.value,
                    TemperatureType.CORE.value,
            ):
                s[k] = res_transient[k][::100].tolist()
        else:
            raise ValueError

    yaml.dump(
        scenario,
        open(os.path.join("test", "functional_test", "scenario_transient_new.yaml"), "w"),
    )


def test_scenario_transient():
    """Test for non-reg in transcient computations."""

    # np.random.seed(_nprs)
    # for c in ["ASTER600", "CROCUS400"]
    # for m in ["rte", "cigre", "ieee", "olla"]
    # for k in ["A", "B", "C", "D", "E", "F", "G", "H", "J"]

    atol = 1.0e-06

    scenario = yaml.safe_load(
        open(os.path.join("test", "functional_test", "scenario_transient.yaml"), "r")
    )

    for i, s in scenario.items():
        nblocks = len(s["conductor"]) if isinstance(s["conductor"], list) else 1
        block = nblocks > 1

        if block:
            args = [
                _get_scenario_enhanced(s["conductor"][j], s["key"][j])
                for j in range(nblocks)
            ]
            dc = {}
            for k in args[0][0].keys():
                dc[k] = np.array([args[j][0][k] for j in range(nblocks)])
            t = args[0][1]
            dynamic = {}
            for k in args[0][2].keys():
                dynamic[k] = np.stack([args[j][2][k] for j in range(nblocks)]).T
        else:
            dc, t, dynamic = _get_scenario_enhanced(s["conductor"], key=s["key"])

        slv = solver._factory(dc, heat_equation=HeatEquationType(s["heat_equation"]), model=ModelType(s["model"]))

        if block:
            slv.args["transit"] = dynamic["transit"][0, :]
            slv.args["wind_speed"] = dynamic["wind_speed"][0, :]
            if "ambient_temperature" in dynamic:
                slv.args["ambient_temperature"] = dynamic["ambient_temperature"][0, :]
            if "wind_angle" in dynamic:
                slv.args["wind_angle"] = dynamic["wind_angle"][0, :]
        else:
            slv.args["transit"] = dynamic["transit"][0]
            slv.args["wind_speed"] = dynamic["wind_speed"][0]
            if "ambient_temperature" in dynamic:
                slv.args["ambient_temperature"] = dynamic["ambient_temperature"][0]
            if "wind_angle" in dynamic:
                slv.args["wind_angle"] = dynamic["wind_angle"][0]
        slv.update()
        res_steady = slv.steady_temperature(return_power=False)

        s["time"] = t[::100].tolist()

        if s["heat_equation"] == "1t":
            res_transient = slv.transient_temperature(
                t,
                T0=res_steady[VariableType.TEMPERATURE.value],
                dynamic=dynamic,
                return_power=False,
            )
            assert np.allclose(
                res_transient[VariableType.TEMPERATURE.value][::100],
                s[VariableType.TEMPERATURE.value],
                atol=atol,
            )
        elif s["heat_equation"] == "3t":
            res_transient = slv.transient_temperature(
                t,
                surface_temperature_0=res_steady[TemperatureType.SURFACE.value],
                core_temperature_0=res_steady[TemperatureType.CORE.value],
                dynamic=dynamic,
                return_power=False,
            )
            for k in (
                TemperatureType.SURFACE.value,
                TemperatureType.AVERAGE.value,
                TemperatureType.CORE.value,
            ):
                assert np.allclose(res_transient[k][::100], s[k], atol=atol)
        else:
            raise ValueError




