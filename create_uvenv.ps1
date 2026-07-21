# SPDX-FileCopyrightText: 2026 Mews Labs (https://www.mews-labs.com)
# SPDX-FileCopyrightText: 2026 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# -- env name
$DEF_NAME = "mewslabs-thermohl-uv"

# $PYTHON_VERSION = "--python-preference=system"
$PYTHON_VERSION="--python 3.12"

# -----------------------------------------------------------------------------

# -- dir for all envs
$ENV_DIR = "$HOME\ENV"
if (-not (Test-Path $ENV_DIR)) {
    New-Item -ItemType Directory -Path $ENV_DIR | Out-Null
}

# -- check uv install, update if necessary
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; installing ..." -ForegroundColor Cyan
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
} else {
    uv self update
    Write-Host "uv already installed ($((uv --version)))" -ForegroundColor Green
}

# -- setup uv env
$VENV_PATH = Join-Path $ENV_DIR $DEF_NAME
Invoke-Expression "uv venv --clear $VENV_PATH $PYTHON_VERSION"

# -- env activation
. "$VENV_PATH\Scripts\Activate.ps1"

# -- upgrade pip
uv pip install --upgrade pip

# # -- install local package
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
uv sync --active --group dev

# -- end text
Write-Host "---"
Write-Host "to start the environment, type :"
Write-Host ". $VENV_PATH\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "to stop the environment, type :"
Write-Host "deactivate" -ForegroundColor Yellow
