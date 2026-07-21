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
    dct = dp | dict(
        lat=46.0,
        alt=1.0,
        azm=90.0,
        month=6,
        day=21,
        hour=0.0,
        Ta=20.0,
        ws=3.0,
        wa=0,
        I=555.0,
        # D=dp["D"],
        # d=dp["d"],
        # A=dp["A"],
        # a=dp["a"],
        # m=dp["m"],
        # c=dp["c"],
        # l=dp["l"],
        alpha=0.9,
        epsilon=0.8,
        # RDC20=dp["RDC20"],
        # kl=dp["kl"],
        # kq=dp["kq"],
        # km=dp["km"],
        # ki=dp["ki"],
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
            args = [_get_scenario_enhanced(s["conductor"][j], s["key"][j]) for j in range(nblocks)]
            dc = {}
            for k in args[0][0].keys():
                dc[k] = [args[j][0][k] for j in range(nblocks)]
            t = args[0][1]
            dynamic = {}
            for k in args[0][2].keys():
                dynamic[k] = np.stack([args[j][2][k] for j in range(nblocks)]).T
        else:
            dc, t, dynamic = _get_scenario_enhanced(s["conductor"], key=s["key"])

        slv = solver._factory(dc, heateq=s["heateq"], model=s["model"])

        if block:
            slv.args["I"] = dynamic["I"][0, :]
            slv.args["ws"] = dynamic["ws"][0, :]
            if "Ta" in dynamic:
                slv.args["Ta"] = dynamic["Ta"][0, :]
            if "wa" in dynamic:
                slv.args["wa"] = dynamic["wa"][0, :]
        else:
            slv.args["I"] = dynamic["I"][0]
            slv.args["ws"] = dynamic["ws"][0]
            if "Ta" in dynamic:
                slv.args["Ta"] = dynamic["Ta"][0]
            if "wa" in dynamic:
                slv.args["wa"] = dynamic["wa"][0]

        slv.update()
        res_steady = slv.steady_temperature()
        s["time"] = t[::100].tolist()

        if s["heateq"] == "1t":
            res_transient = slv.transient_temperature(
                t,
                T0=res_steady["t"],
                dynamic=dynamic,
                return_power=False,
            )
            s[solver.Solver.Names.temp] = res_transient[solver.Solver.Names.temp][::100].tolist()
            s["temperature"] = s.pop("t")
        elif s["heateq"] == "3t":
            res_transient = slv.transient_temperature(
                t,
                Ts0=res_steady["t_surf"],
                Tc0=res_steady["t_core"],
                dynamic=dynamic,
                return_power=False,
            )
            for k in (solver.Solver.Names.tsurf, solver.Solver.Names.tavg, solver.Solver.Names.tcore):
                s[k] = res_transient[k][::100].tolist()
            s["surface_temperature"] = s.pop("t_surf")
            s["average_temperature"] = s.pop("t_avg")
            s["core_temperature"] = s.pop("t_core")
        else:
            raise ValueError
        s["heat_equation"] = s.pop("heateq")

    yaml.dump(scenario, open(os.path.join("test", "functional_test", "scenario_transient.yaml"), "w"))

def test_scenario_transient():
    """Test for non-reg in transcient computations."""

    # np.random.seed(_nprs)
    # for c in ["ASTER600", "CROCUS400"]
    # for m in ["rte", "cigre", "ieee", "olla"]
    # for k in ["A", "B", "C", "D", "E", "F", "G", "H", "J"]

    atol = 1.0E-06

    scenario = yaml.safe_load(open(os.path.join("test", "functional_test", "scenario_transient.yaml"), "r"))

    for i, s in scenario.items():

        s["heateq"] = s.pop("heat_equation")
        if s["heateq"] == "1t":
            s["t"] = s.pop("temperature")
        elif s["heateq"] == "3t":
            s["t_surf"] = s.pop("surface_temperature")
            s["t_avg"] = s.pop("average_temperature")
            s["t_core"] = s.pop("core_temperature")

        nblocks = len(s["conductor"]) if isinstance(s["conductor"], list) else 1
        block = nblocks > 1

        if block:
            args = [_get_scenario_enhanced(s["conductor"][j], s["key"][j]) for j in range(nblocks)]
            dc = {}
            for k in args[0][0].keys():
                dc[k] = [args[j][0][k] for j in range(nblocks)]
            t = args[0][1]
            dynamic = {}
            for k in args[0][2].keys():
                dynamic[k] = np.stack([args[j][2][k] for j in range(nblocks)]).T
        else:
            dc, t, dynamic = _get_scenario_enhanced(s["conductor"], key=s["key"])

        slv = solver._factory(dc, heateq=s["heateq"], model=s["model"])

        if block:
            slv.args["I"] = dynamic["I"][0, :]
            slv.args["ws"] = dynamic["ws"][0, :]
            if "Ta" in dynamic:
                slv.args["Ta"] = dynamic["Ta"][0, :]
            if "wa" in dynamic:
                slv.args["wa"] = dynamic["wa"][0, :]
        else:
            slv.args["I"] = dynamic["I"][0]
            slv.args["ws"] = dynamic["ws"][0]
            if "Ta" in dynamic:
                slv.args["Ta"] = dynamic["Ta"][0]
            if "wa" in dynamic:
                slv.args["wa"] = dynamic["wa"][0]

        slv.update()
        res_steady = slv.steady_temperature()
        s["time"] = t[::100].tolist()

        if s["heateq"] == "1t":
            res_transient = slv.transient_temperature(
                t,
                T0=res_steady["t"],
                dynamic=dynamic,
                return_power=False,
            )
            assert np.allclose(res_transient[solver.Solver.Names.temp][::100], s[solver.Solver.Names.temp], atol=atol)
        elif s["heateq"] == "3t":
            res_transient = slv.transient_temperature(
                t,
                Ts0=res_steady["t_surf"],
                Tc0=res_steady["t_core"],
                dynamic=dynamic,
                return_power=False,
            )
            for k in (solver.Solver.Names.tsurf, solver.Solver.Names.tavg, solver.Solver.Names.tcore):
                assert np.allclose(res_transient[k][::100], s[k], atol=atol)
        else:
            raise ValueError
