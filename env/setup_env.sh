#!/usr/bin/env bash
set -euo pipefail

# 1) conda/mamba enviroment recommended
# mamba would be the fastest option

ENV_NAME=docking
YML_FILE=environment.yml

if command -v mamba >/dev/null 2>&1; then
  echo "Using mamba to create environment..."
  mamba env create -f "$YML_FILE" -n "$ENV_NAME" || mamba env update -f "$YML_FILE" -n "$ENV_NAME"
else
  echo "Using conda to create environment..."
  conda env create -f "$YML_FILE" -n "$ENV_NAME" || conda env update -f "$YML_FILE" -n "$ENV_NAME"
fi

# 2) installing via pip

if [ -f requirements.txt ]; then
  echo "Installing pip packages from requirements.txt..."
  conda run -n "$ENV_NAME" pip install -r requirements.txt
fi

# 3) environment check

echo "Done. To activate:"
echo "  conda activate $ENV_NAME"
echo "Installed top-level packages:"
conda run -n "$ENV_NAME" python -c "import sys, pkgutil; print('python', sys.version.split()[0]); import numpy as np; print('numpy', np.__version__)"