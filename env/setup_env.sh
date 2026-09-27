#!/usr/bin/env bash
# Build the PeriodicWave environment. Idempotent: safe to re-run.
# No root required -- uv and its Python live under $HOME.
set -eo pipefail
ROOT="$HOME/moire-nnvmc"
cd "$ROOT"

echo "=== 1/6  userspace toolchain (uv) ==="
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== 2/6  venv on Python 3.13 ==="
uv venv --python 3.13 .venv
. .venv/bin/activate
python --version

echo "=== 3/6  pinned package set ==="
if ! uv pip install -r env/requirements-gpu.txt; then
  echo "!!! pinned install failed; relaxing the tfp-nightly pin (nightly builds get yanked from PyPI)"
  sed -i 's/^tfp-nightly==.*/tfp-nightly/' env/requirements-gpu.txt
  uv pip install -r env/requirements-gpu.txt
fi

echo "=== 4/6  PeriodicWave itself (--no-deps: the pins above are authoritative) ==="
uv pip install -e external/PeriodicWave --no-deps

echo "=== 5/6  verify ==="
python - <<'PY'
import sys, jax, jaxlib
print("python      ", sys.version.split()[0])
print("jax         ", jax.__version__)
print("jaxlib      ", jaxlib.__version__)
devs = jax.devices()
print("n_devices   ", len(devs))
for d in devs:
    print("  device    ", d.id, d.platform, d.device_kind)
import jax.numpy as jnp
x = jnp.ones((2048, 2048))
print("matmul_ok   ", float((x @ x)[0, 0]) == 2048.0)
import periodicwave, folx, kfac_jax, optax, chex, distrax
print("periodicwave", periodicwave.__file__)
print("imports_ok   folx kfac_jax optax chex distrax")
PY

echo "=== 6/6  freeze ==="
uv pip freeze > env/freeze-actual.txt
wc -l env/freeze-actual.txt
echo "=== SETUP COMPLETE ==="
