# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Type, Optional, Any

import numpy as np
import pandas as pd
import scipy.linalg
from pyntb.optimize import qnewt2d_v
from scipy.optimize import root

from thermohl.power import PowerTerm
from thermohl.solver.base import Solver as Solver_


def layer_qty(
    dl: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From layer properties dataframe dl, compute some properties of the cable
    per layer.
    """

    # shortcuts
    aw = np.pi * 0.25 * dl["d_brins"].values ** 2
    nw = dl["nb_brins"].values
    lw = dl["l_brins"].values

    # cumulative radius
    rl = np.array(dl["d_brins"].values)
    rl[0] /= 2
    rl = np.cumsum(rl)

    # cumulative section
    al = aw * nw

    # layer mass
    ml = nw * (aw * lw) * (1000.0 * dl["densite"])
    ml = ml.values

    # layer heat capacity
    hl = np.array([_heatcf[m] for m in dl["nuance"]])

    # layer electric resistance
    rw = dl["resistivite_20degres"] * lw / aw
    ol = 1 / (nw / rw)
    ol = ol.values

    return rl, al, ml, hl, ol


def _rgrid(rl: np.ndarray, n: int = 10) -> np.ndarray:
    """From an array of cumulative radii which gives interlayer positions,
    compute a discretization with about n points per mm which guarantees that
    each elementary volume remains in one layer only.
    """
    r = None
    for x in rl:
        if r is None:
            r = np.linspace(0, x, int(np.round(n * x * 1000.0)))
        else:
            r = np.concatenate(
                (r, np.linspace(r[-1], x, int(np.round(n * (x - r[-1]) * 1000.0))))
            )
    r = np.unique(r)
    return r


def _discretization(dl: pd.DataFrame, n: int = 10):
    """Get discretized quantities in a cylinder."""

    # rl is radius of each layer
    # hl is heat capacity of each layer
    # ol is electric resistance of each layer
    rl, _, _, hl, ol = layer_qty(dl)

    # edges
    r = _rgrid(rl, n=n)

    # centers
    s = 0.5 * (r[:-1] + r[1:])

    # layer repartition: ly is an array of same size as s with the layer number
    # in each element
    ly = np.zeros_like(s, dtype=int)
    cc = 0
    for j in range(len(ly)):
        if r[1 + j] > rl[cc]:
            cc += 1
            ly[j] = (rl[cc - 1] - r[j]) / (r[j + 1] - r[j]) * (cc - 1) + (
                r[j + 1] - rl[cc - 1]
            ) / (r[j + 1] - r[j]) * cc
        else:
            ly[j] = cc
    del (cc, j)

    # volumic mass (for each element)
    dy = dl["densite"].values[ly] * 1000.0

    # heat capacity (for each element)
    hy = hl[ly]

    # electric resistance (for each element)
    ry = ol[ly]

    # current distribution (for each element)
    il = 1 / (ol * np.sum(1 / ol))
    iy = il[ly]

    return r, s, rl, ly, dy, hy, ry, iy


def rstat_analytic(slv, Tsg=None, Tcg=None, tol: float = 1.0e-12, maxiter: int = 64):
    """Compute steady state with analytical model."""

    # if no guess provided, use ambient temp
    try:
        shape = (thermohl.utils.dict_max_len(slv.dc),)
    except AttributeError:
        shape = slv.args.shape()
        slv.dc = slv.args
    if Tsg is None:
        Tsg = 3.0 * slv.dc["Ta"]
    if Tcg is None:
        Tcg = 3.0 * slv.dc["Ta"]
    Tsg_ = Tsg * np.ones(shape)
    Tcg_ = Tcg * np.ones(shape)

    # get morgan coefficients
    c = _morgan_coeff(slv.dc["D"], slv.dc["d"], shape)
    D_ = slv.dc["D"] * np.ones(shape)
    d_ = slv.dc["d"] * np.ones(shape)
    i = d_ > 0.0

    # joule effect (depending only on x=tsurf and y=tcore)
    def joule(x, y):
        t = 0.5 * (x + y)
        t[i] = _profile_bim_avg(x[i], y[i], d_[i], D_[i])
        j = _JouleHeating.value(
            t,
            slv.dc["I"],
            slv.dc["D"],
            slv.dc["d"],
            slv.dc["A"],
            slv.dc["a"],
            slv.dc["km"],
            slv.dc["ki"],
            slv.dc["kl"],
            slv.dc["kq"],
            slv.dc["RDC20"],
            shape=shape,
        )
        return j

    # power balance equation
    def f1(x, y):
        return (
            joule(x, y)
            + slv.sh.value(x)
            - slv.cc.value(x)
            - slv.rc.value(x)
            - slv.pc.value(x)
        )

    # surface-core equilibrium
    def f2(x, y):
        return y - x - c * joule(x, y) / (2.0 * np.pi * slv.dc["l"])

    # solve system
    x, y, i, e = qnewt2d_v(
        f1, f2, Tsg_, Tcg_, rtol=tol, maxiter=maxiter, dx=1.0e-03, dy=1.0e-03
    )
    if np.max(e) > tol or i == maxiter:
        print(f"rstat_analytic max err is {np.max(e):.3E} in {i:d} iterations")

    if shape == (1,):
        return x[0], y[0]

    return x, y


def _matrices(r, s, dy, hy, theta, dt, lmbda):
    """Get matrix used in solvers"""
    dsup = np.zeros(len(s) - 1)
    diag = np.zeros(len(s))
    dinf = np.zeros(len(s) - 1)

    dsup[0] = +1.0 * r[1] / (s[1] - s[0])
    diag[0] = -1.0 * r[1] / (s[1] - s[0])

    dsup[1:] = r[2:-1] / (s[2:] - s[1:-1])
    diag[1:-1] = -1.0 * (r[2:-1] / (s[2:] - s[1:-1]) + r[1:-2] / (s[1:-1] - s[:-2]))
    dinf[:-1] = r[1:-2] / (s[1:-1] - s[:-2])

    diag[-1] = -1.0 * r[-2] / (s[-1] - s[-2])
    dinf[-1] = +1.0 * r[-2] / (s[-1] - s[-2])

    C = scipy.sparse.diags((dinf, diag, dsup), [-1, 0, +1])
    D = (dy * hy) * s * np.diff(r)
    Cp = scipy.sparse.diags(D) + (1 - theta) * dt * lmbda * C
    Cm = scipy.sparse.diags(D) - theta * dt * lmbda * C

    Cm2 = np.zeros((3, len(s)))
    Cm2[0, 1:] = Cm.diagonal(+1)
    Cm2[1, :] = Cm.diagonal(0)
    Cm2[2, :-1] = Cm.diagonal(-1)

    # C2 = np.zeros((3, len(s)))
    # C2[0, 1:] = C.diagonal(+1)
    # C2[1, :] = C.diagonal(0)
    # C2[2, :-1] = C.diagonal(-1)
    # return C2, Cp, Cm2, D / (dy * hy)

    return C, Cp, Cm2, D / (dy * hy)


def rstat_num(dl, slv, n=10, homogeneous=True):
    """Compute steady state of discrete system."""

    try:
        if thermohl.utils.dict_max_len(slv.dc) > 1:
            raise ValueError("Dict max len must be 1")
    except AttributeError:
        if len(slv.args.shape()) > 0 and slv.args.shape()[0] > 1:
            raise ValueError("Dict max len must be 1")
        slv.dc = slv.args

    # discretization
    r, s, rl, ly, dy, hy, ry, iy = _discretization(dl, n=n)

    # build matrices
    if homogeneous:
        rho = slv.dc["m"] / slv.dc["A"]
        cp = slv.dc["c"]
        C, _, _, D = _matrices(r, s, rho, cp, 0.0, 0.0, 0.0)
    else:
        C, _, _, D = _matrices(r, s, dy, hy, 0.0, 0.0, 0.0)

    # init guess
    tsg, tcg = rstat_analytic(slv)
    T0 = np.interp(s, r, _profile_mom(tsg, tcg, r, r[-1]))
    del (tsg, tcg)

    # pre-stuff
    b = np.zeros_like(s)
    r2 = r[-1] ** 2

    if homogeneous:

        if slv.dc["d"] > 0.0:
            msk = np.zeros_like(s)
            ri = 0.5 * slv.dc["d"]
            msk[s > ri] = 1.0 / (np.pi * (r2 - ri**2))
        else:
            msk = 1.0 / (np.pi * r2)

        def fun(x):
            b[-1] = (
                (
                    slv.sh.value(x[[-1]])
                    - slv.cc.value(x[[-1]])
                    - slv.rc.value(x[[-1]])
                    - slv.pc.value(x[[-1]])
                )[0]
                * 0.5
                / np.pi
            )
            Ta = np.sum((x[1:] * s[1:] + x[:-1] * s[:-1]) * np.diff(s)) / r2
            qj = (
                _JouleHeating.value(
                    Ta,
                    slv.dc["I"],
                    slv.dc["D"],
                    slv.dc["d"],
                    slv.dc["A"],
                    slv.dc["a"],
                    slv.dc["km"],
                    slv.dc["ki"],
                    slv.dc["kl"],
                    slv.dc["kq"],
                    slv.dc["RDC20"],
                )
                * msk
            )
            return C * x + (D * qj + b) / slv.dc["l"]

    else:

        def fun(x):
            b[-1] = (
                (
                    slv.sh.value(x[[-1]])
                    - slv.cc.value(x[[-1]])
                    - slv.rc.value(x[[-1]])
                    - slv.pc.value(x[[-1]])
                )[0]
                * 0.5
                / np.pi
            )
            Ta = np.sum((x[1:] * s[1:] + x[:-1] * s[:-1]) * np.diff(s)) / r2
            qj = _JouleHeating.value_discr(
                Ta,
                slv.dc["I"],
                slv.dc["D"],
                slv.dc["d"],
                slv.dc["A"],
                slv.dc["a"],
                slv.dc["km"],
                slv.dc["ki"],
                slv.dc["kl"],
                slv.dc["kq"],
                slv.dc["RDC20"],
                rl,
                ly,
                ry,
                iy,
            )
            return C * x + (D * qj + b) / slv.dc["l"]

    # solve
    opr = root(fun, T0)
    if not opr.success:
        print(f"{opr.message}")

    return s, opr.x


def rsolve(t, I, Ta, ws, wa, T0, dl, slv, n=10, theta=0.5, homogeneous=True):
    """Solver for heat equation.

    T0 is initial temperature,
    dl is a pd.DataFrame with layer properties
    t is a time vector
    n is a space-discretization parameter
    """

    try:
        if thermohl.utils.dict_max_len(slv.dc) > 1:
            raise ValueError("Dict max len must be 1")
    except AttributeError:
        if len(slv.args.shape()) > 0 and slv.args.shape()[0] > 1:
            raise ValueError("Dict max len must be 1")
        slv.dc = slv.args

    # discretization
    r, s, rl, ly, dy, hy, ry, iy = _discretization(dl, n=n)

    # init
    dt_ = (t[-1] - t[0]) / (len(t) - 1)
    T = np.zeros((len(s), len(t)))
    try:
        n = len(T0)
        T[:, 0] = np.interp(s, np.linspace(0.0, r[-1], n), T0)
    except TypeError:
        T[:, 0] = T0

    # build matrices
    if homogeneous:
        rho = slv.dc["m"] / slv.dc["A"]
        cp = slv.dc["c"]
        _, Cp, Cm2, D = _matrices(r, s, rho, cp, theta, dt_, slv.dc["l"])

        if slv.dc["d"] > 0.0:
            msk = np.zeros_like(s)
            ri = 0.5 * slv.dc["d"]
            msk[s > ri] = 1.0 / (np.pi * (r[-1] ** 2 - ri**2))
        else:
            msk = 1.0 / (np.pi * r[-1] ** 2)

    else:
        _, Cp, Cm2, D = _matrices(r, s, dy, hy, theta, dt_, slv.dc["l"])

    # boundary condition vector
    b = np.zeros_like(s)

    # loop pre-computing
    r2 = r[-1] ** 2

    # time loop
    Tavg = np.zeros_like(t)
    Pjou = np.zeros_like(t)
    for i in range(len(t)):
        if Ta is not None:
            slv.args["Ta"] = Ta[i]
        if ws is not None:
            slv.args["ws"] = ws[i]
        if wa is not None:
            slv.args["wa"] = wa[i]
        slv.update()
        b[-1] = (
            (
                slv.sh.value(T[[-1], [i]])
                - slv.cc.value(T[[-1], [i]])
                - slv.rc.value(T[[-1], [i]])
                - slv.pc.value(T[[-1], [i]])
            )
            * 0.5
            / np.pi
        )
        Tavg[i] = np.sum((T[1:, i] * s[1:] + T[:-1, i] * s[:-1]) * np.diff(s)) / r2

        if homogeneous:
            qj = _JouleHeating.value(
                Tavg[i],
                I[i],
                slv.dc["D"],
                slv.dc["d"],
                slv.dc["A"],
                slv.dc["a"],
                slv.dc["km"],
                slv.dc["ki"],
                slv.dc["kl"],
                slv.dc["kq"],
                slv.dc["RDC20"],
            )
            Pjou[i] = qj
            qj *= msk
        else:
            qj = _JouleHeating.value_discr(
                Tavg[i],
                I[i],
                slv.dc["D"],
                slv.dc["d"],
                slv.dc["A"],
                slv.dc["a"],
                slv.dc["km"],
                slv.dc["ki"],
                slv.dc["kl"],
                slv.dc["kq"],
                slv.dc["RDC20"],
                rl,
                ly,
                ry,
                iy,
            )
            Pjou[i] = _JouleHeating.integrate_value_disc(qj, s)
        if i == len(t) - 1:
            break
        T[:, i + 1] = scipy.linalg.solve_banded(
            (1, 1), Cm2, Cp * T[:, i] + dt_ * (D * qj + b)
        )

    # postproc
    rlay = np.concatenate(([0.0], rl))

    def _interp(x, xp, fp):
        j = np.searchsorted(xp, x) - 1
        z = j < 0
        m = j >= len(xp) - 1
        k = ~np.logical_or(m, z)
        d = np.zeros_like(x)
        j[z] = 0
        d[z] = 0.0
        d[k] = (x[k] - xp[j[k]]) / (xp[j[k] + 1] - xp[j[k]])
        j[m] = len(s) - 2
        d[m] = 1.0
        y = np.zeros((len(x), fp.shape[1]))
        for l in range(len(d)):
            y[l, :] = (1.0 - d[l]) * fp[j[l], :] + fp[j[l] + 1, :] * d[l]
        return y

    pp = dict(
        t=t,
        layr=ly,
        rlay=rlay,
        Tmin=np.min(T, axis=0),
        Tmax=np.max(T, axis=0),
        Tsurf=T[-1, :],
        Tcore=T[0, :],
        Tavg=Tavg,
        Tlay=_interp(rlay, s, T),
        Pjou=Pjou,
        Psol=slv.sh.value(T[-1, :]),
        Pcnv=slv.cc.value(T[-1, :]),
        Prad=slv.rc.value(T[-1, :]),
        Ppre=slv.pc.value(T[-1, :]),
    )

    return s, T, pp


class Solver1D(Solver_):

    def __init__(
        self,
        dic: Optional[dict[str, Any]] = None,
        joule: Type[PowerTerm] = PowerTerm,
        solar: Type[PowerTerm] = PowerTerm,
        convective: Type[PowerTerm] = PowerTerm,
        radiative: Type[PowerTerm] = PowerTerm,
        precipitation: Type[PowerTerm] = PowerTerm,
    ):
        super().__init__(dic, joule, solar, convective, radiative, precipitation)
        self.update()
