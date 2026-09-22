#!/usr/bin/env bash
#
# Launch Inline Studio (Inline Core + the web UI) on one port. Friendly flags map onto the engine's
# environment knobs, so you do not have to remember the INLINE_* variables. This is the one command
# an end user needs: it installs deps (with --install), makes sure a web UI is present (the prebuilt
# omnichar-frontend package, or a local SPA build), then serves the app. Run `./webui.sh --help`.
#
#   ./webui.sh                              # loopback, port 8848 (UI + API)
#   ./webui.sh --listen --port 9000         # bind all interfaces on 9000
#   ./webui.sh --multi-gpu                  # multi-GPU denoise (auto-detected when 2+ GPUs)
#   ./webui.sh --multi-gpu pipefusion=2     # force a split
#   ./webui.sh --lowvram                    # tight-VRAM profile
#   ./webui.sh --install --extra runtime    # set up the venv with the local model runtime, then exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

HOST="127.0.0.1"
PORT="8848"
EXTRAS="server"
RUN_INSTALL=0
DEV_MODE=0
FORCE_REBUILD=0
SMART_MEMORY=0
USE_ACTIVE_ENV=0
RECREATE=0
PRINT_TORCH_INDEX=0
TORCH_INDEX_CHOICE="${INLINE_TORCH_INDEX:-}"

usage() {
  cat <<'EOF'
Usage: ./webui.sh [options]

Networking
  --listen               bind all interfaces (0.0.0.0), so other machines can reach it
  --host ADDR            bind a specific address (default 127.0.0.1)
  --port N               listen on port N (default 8848)

Multi-GPU (split one image's denoise across GPUs)
  --multi-gpu [SPEC]     enable the split; auto-detected with 2+ GPUs. Optional SPEC overrides the
                         degrees, e.g. pipefusion=2 or pipefusion=2,ulysses=2
  --parallel SPEC        alias for --multi-gpu SPEC

Device / memory
  --lowvram              tight-VRAM profile (slicing + tiling + int8, weights resident)
  --smart-memory         spread a too-big model across VRAM + RAM + CPU: graduated CPU offload
                         (model, or sequential on a tiny GPU) + int8 quant so the offloaded half
                         fits in RAM. Slower per image, but runs where full-resident OOMs.
  --cpu                  force the CPU profile
  --profile NAME         set the profile explicitly (gpu-max | lowvram | cpu)
  --vram-budget GB       treat the GPU as having GB of usable VRAM

Paths
  --models-dir PATH      where weights are scanned from (default ./models)
  --data-dir PATH        where runs and takes are written (default ./.inline)

Setup
  --install              create ./.venv (via uv) and install, then exit. An existing ./.venv is
                         reused, and an unrelated environment activated in your shell is never
                         touched.
  --extra NAME           add an install extra (repeatable): runtime, parallel, server, training
  --torch-index WHICH    with --install, override the PyTorch wheel index picked from your GPU's
                         compute capability. A short name (cu130, cu128, cu126), a full index URL,
                         or "cpu" to force the CPU-only build. Also settable as INLINE_TORCH_INDEX.
                         cu128 is frozen at torch 2.11 and exists only for Blackwell cards whose
                         driver predates R580; --install picks it for you in that case.
  --print-torch-index    print what the GPU probe read and which index would be used, then exit
                         without installing anything
  --recreate             with --install, rebuild ./.venv from scratch (discards anything installed
                         into it by hand, e.g. a ROCm build of PyTorch)
  --use-active-env       install into / run from the environment activated in this shell instead of
                         ./.venv
  -h, --help             show this help

Development
  --dev                  live-reload dev loop: run Core (API) in the background and the Vite HMR
                         server, then edit src/ and see changes instantly. Open http://localhost:5173
                         (NOT the Core port). Needs the repo checkout + Node/npm.
  --rebuild              force a fresh SPA build (npm run build:spa) from source and serve that local
                         build on the one port. Use after code changes when not running --dev.

Web UI
  Served automatically on the same port. On start, if no UI is found we install the prebuilt
  omnichar-frontend package, else build it from source (npm) when this is the repo checkout,
  else run API-only. Point at a local build with INLINE_FRONTEND_ROOT (or main.py --front-end-root).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --listen) HOST="0.0.0.0"; shift ;;
    --host) HOST="${2:?--host needs an address}"; shift 2 ;;
    --port) PORT="${2:?--port needs a number}"; shift 2 ;;
    --multi-gpu|--parallel)
      if [[ $# -ge 2 && "${2:-}" != -* ]]; then
        export INLINE_PARALLEL="$2"; shift 2
      else
        echo "Multi-GPU split is auto-detected with 2+ GPUs; pass e.g. pipefusion=2 to override."
        shift
      fi ;;
    --lowvram) export INLINE_PROFILE="lowvram"; shift ;;
    --smart-memory)
      # Spread a too-big model across VRAM + RAM + CPU. Force the lowvram profile so the offload +
      # int8 machinery engages. (The expandable allocator that cuts CUDA fragmentation OOMs is now
      # set unconditionally below, so every run - not just smart memory - benefits.)
      SMART_MEMORY=1
      export INLINE_SMART_MEMORY="1"
      export INLINE_PROFILE="${INLINE_PROFILE:-lowvram}"
      shift ;;
    --cpu) export INLINE_PROFILE="cpu"; shift ;;
    --profile) export INLINE_PROFILE="${2:?--profile needs a name}"; shift 2 ;;
    --vram-budget) export INLINE_VRAM_BUDGET_GB="${2:?--vram-budget needs a number}"; shift 2 ;;
    --models-dir) export INLINE_MODELS_DIR="${2:?--models-dir needs a path}"; shift 2 ;;
    --data-dir) export INLINE_DATA_DIR="${2:?--data-dir needs a path}"; shift 2 ;;
    --install) RUN_INSTALL=1; shift ;;
    --extra) EXTRAS="$EXTRAS,${2:?--extra needs a name}"; shift 2 ;;
    --torch-index) TORCH_INDEX_CHOICE="${2:?--torch-index needs a name, URL or 'cpu'}"; shift 2 ;;
    --print-torch-index) PRINT_TORCH_INDEX=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    --use-active-env) USE_ACTIVE_ENV=1; shift ;;
    --dev) DEV_MODE=1; shift ;;
    --rebuild) FORCE_REBUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

export INLINE_HOST="$HOST"
export INLINE_PORT="$PORT"

# Always use PyTorch's expandable CUDA segments (unless the user set their own config): it lets the
# allocator grow/shrink a single reservation instead of fragmenting into many, which is the
# difference between a small allocation failing "with VRAM still free" and succeeding - a common
# low-VRAM (e.g. T4) OOM. Harmless on CPU-only runs (torch just ignores it).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ACTIVE_ENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"

# Prints the environment active in this shell when it is NOT ours. Everything below keys off this: an
# env that merely happens to be activated must never absorb an install or a launch-time dependency.
foreign_env() {
  [[ -n "$ACTIVE_ENV" ]] || return 1
  [[ "$(cd "$ACTIVE_ENV" 2>/dev/null && pwd -P)" != "$(cd "$VENV_DIR" 2>/dev/null && pwd -P)" ]] || return 1
  printf '%s\n' "$ACTIVE_ENV"
}

# Validate and dedupe the extras, so a typo fails here with the valid list instead of reaching uv as a
# raw resolver error. The names must match [project.optional-dependencies] in pyproject.toml.
normalize_extras() {
  local name seen="" out=""
  for name in ${EXTRAS//,/ }; do
    case " runtime server parallel training dev all " in
      *" $name "*) ;;
      *) echo "unknown extra: $name (valid: runtime, server, parallel, training, dev, all)" >&2; exit 1 ;;
    esac
    case " $seen " in *" $name "*) continue ;; esac
    seen="$seen $name"
    out="${out:+$out,}$name"
  done
  EXTRAS="$out"
}

# Only Windows needs us to name a CUDA index: PyPI's torch is CPU-only there, while the Linux wheels
# already bundle CUDA. This script also runs under Git Bash / MSYS on Windows.
is_windows() {
  case "$(uname -s 2>/dev/null || true)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT) return 0 ;;
    *) return 1 ;;
  esac
}

# One query for both facts, cached, so the raw line can be echoed verbatim. A silent guess is what
# made two field reports unfalsifiable: whatever we decide, the input is printed alongside it.
GPU_PROBE_RAW=""
GPU_CAP_MAJOR=""
GPU_DRIVER_MAJOR=""

read_gpu_probe() {
  local line cap driver major
  GPU_PROBE_RAW="$(nvidia-smi --query-gpu=compute_cap,driver_version --format=csv,noheader 2>/dev/null || true)"
  while IFS=, read -r cap driver; do
    cap="${cap// /}"; driver="${driver// /}"
    major="${cap%%.*}"
    # Numeric guard, not a coercion: an old driver answers an unknown query with an error string,
    # and a bare word must never become a capability.
    [[ "$major" =~ ^[0-9]+$ ]] || continue
    if [[ -z "$GPU_CAP_MAJOR" || "$major" -gt "$GPU_CAP_MAJOR" ]]; then
      GPU_CAP_MAJOR="$major"
      GPU_DRIVER_MAJOR="${driver%%.*}"
    fi
  done <<<"$GPU_PROBE_RAW"
}

#: Why the index was chosen, for the message and the install record: autodetect | driver-floor-cu128
#: | override | no-gpu.
TORCH_INDEX_REASON="autodetect"

# cu126 is the last index built for Maxwell..Volta; sm_100/sm_120 need cu128 or newer. cu128 serves
# but is frozen at torch 2.11 (checked 2026-02), so it is only for Blackwell below CUDA 13's R580.
# Assigns rather than prints: through $(...) the probe globals would die in the subshell.
pick_torch_index() {
  read_gpu_probe
  if [[ -z "$GPU_CAP_MAJOR" ]]; then TORCH_CHOICE="cu126"; return 0; fi
  if [[ "$GPU_CAP_MAJOR" -ge 10 ]]; then
    if [[ -n "$GPU_DRIVER_MAJOR" && "$GPU_DRIVER_MAJOR" -lt 580 ]]; then
      TORCH_CHOICE="cu128"
      TORCH_INDEX_REASON="driver-floor-cu128"
      return 0
    fi
    TORCH_CHOICE="cu130"
    return 0
  fi
  TORCH_CHOICE="cu126"
}

# cu128 is frozen, so the current torch/torchvision pair does not exist there. Pin the last one it
# has instead of resolving, or uv picks whatever is newest on an index that stopped moving.
torch_pins_for() {
  case "$1" in
    cu128) printf 'torch==2.11.0 torchvision==0.26.0\n' ;;
    *) printf 'torch torchvision\n' ;;
  esac
}

torch_index_url() {
  case "$1" in
    http://*|https://*) printf '%s\n' "$1" ;;
    *) printf 'https://download.pytorch.org/whl/%s\n' "$1" ;;
  esac
}

# Detect-only: report the probe and the decision, change nothing. This is what CI asserts against
# and what a bug report should paste, instead of a whole reinstall log.
if [[ "$PRINT_TORCH_INDEX" -eq 1 ]]; then
  TORCH_CHOICE=""
  if [[ -n "$TORCH_INDEX_CHOICE" ]]; then
    read_gpu_probe
    PRINT_CHOICE="$TORCH_INDEX_CHOICE"
    TORCH_INDEX_REASON="override"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    pick_torch_index
    PRINT_CHOICE="$TORCH_CHOICE"
    # Linux PyPI wheels already bundle CUDA, so the installer names no index there at all.
    if ! is_windows; then PRINT_CHOICE="$PRINT_CHOICE (linux: installer uses the default PyPI wheels)"; fi
  else
    read_gpu_probe
    PRINT_CHOICE="cpu"
    TORCH_INDEX_REASON="no-gpu"
  fi
  echo "probe: ${GPU_PROBE_RAW:-<nvidia-smi unavailable>}"
  echo "capability-major: ${GPU_CAP_MAJOR:-unknown}"
  echo "driver-major: ${GPU_DRIVER_MAJOR:-unknown}"
  echo "torch-index: $PRINT_CHOICE"
  echo "reason: $TORCH_INDEX_REASON"
  exit 0
fi

if [[ "$RUN_INSTALL" -eq 1 ]]; then
  command -v uv >/dev/null 2>&1 || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
  normalize_extras
  # Every uv call below is pinned with --python. uv's pip interface otherwise targets an active
  # VIRTUAL_ENV/CONDA_PREFIX ahead of the local .venv, which is how an unrelated venv activated in the
  # user's shell silently absorbed the install and had its packages replaced.
  TARGET_PY="$VENV_PY"
  if [[ "$USE_ACTIVE_ENV" -eq 1 ]]; then
    [[ -n "$ACTIVE_ENV" ]] || { echo "--use-active-env needs an activated environment." >&2; exit 1; }
    TARGET_PY="$ACTIVE_ENV/bin/python"
    echo "Installing into the active environment: $ACTIVE_ENV"
  else
    if FOREIGN="$(foreign_env)"; then
      echo "NOTE: the environment active in this shell will NOT be modified:"
      echo "        $FOREIGN"
      echo "      Installing into $VENV_DIR instead (--use-active-env overrides this)."
    fi
    if [[ "$RECREATE" -eq 1 ]]; then
      echo "Recreating $VENV_DIR - anything installed into it by hand (e.g. a ROCm build of PyTorch) is lost."
      uv venv --clear "$VENV_DIR"
    elif [[ -x "$VENV_PY" ]]; then
      # uv venv refuses an existing environment and --clear would wipe manual installs, so a repeat
      # --install (say, to add an extra) has to reuse what is already there.
      echo "Reusing the existing environment at $VENV_DIR (--recreate rebuilds it from scratch)."
    else
      uv venv "$VENV_DIR"
    fi
  fi
  # Pick the right torch wheel. PyPI's default torch is CPU-only on Windows (Linux wheels bundle
  # CUDA), so installing blind there yields a working install that generates on the CPU ~100x
  # slower, with no error. Which CUDA index is right depends on the card - see pick_torch_index.
  TORCH_INDEX=()
  TORCH_CHOICE="$TORCH_INDEX_CHOICE"
  TORCH_EXPLICIT=0
  if [[ -n "$TORCH_CHOICE" ]]; then TORCH_EXPLICIT=1; TORCH_INDEX_REASON="override"; fi
  GPU_PRESENT=0
  NO_GPU_WHY=""
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    NO_GPU_WHY="nvidia-smi is not on PATH"
  elif ! nvidia-smi -L >/dev/null 2>&1; then
    NO_GPU_WHY="nvidia-smi ran but listed no GPU"
  else
    GPU_PRESENT=1
  fi
  if [[ -z "$TORCH_CHOICE" ]]; then
    if [[ "$GPU_PRESENT" -eq 1 ]]; then
      if is_windows; then pick_torch_index; fi
    else
      TORCH_CHOICE="cpu"
      TORCH_INDEX_REASON="no-gpu"
    fi
  fi
  # Whatever we decided, show the input it came from. Both field reports were unfalsifiable because
  # the launcher printed a conclusion and never the evidence.
  if [[ "$GPU_PRESENT" -eq 1 && -n "$GPU_PROBE_RAW" ]]; then
    echo "GPU probe (compute_cap, driver_version): $(printf '%s' "$GPU_PROBE_RAW" | tr '\n' ';')"
  fi
  if [[ "$TORCH_INDEX_REASON" == "driver-floor-cu128" ]]; then
    echo "Driver ${GPU_DRIVER_MAJOR}.xx predates R580, which CUDA 13 needs. Installing from cu128,"
    echo "which is frozen at torch 2.11 and will never update. Updating the NVIDIA driver and"
    echo "re-running --install gets you current torch via cu130."
  fi
  if [[ "$TORCH_CHOICE" == "cpu" && "$GPU_PRESENT" -eq 0 ]]; then
    echo "No NVIDIA GPU detected ($NO_GPU_WHY) - installing the default (CPU) build of PyTorch."
    echo "  On an AMD or Intel GPU this is not what you want: pass a full index URL, e.g."
    echo "  ./webui.sh --install --torch-index https://download.pytorch.org/whl/rocm6.2"
    echo "  See the README section \"AMD (ROCm) setup\"."
  elif [[ "$TORCH_CHOICE" == "cpu" ]]; then
    echo "Installing the default (CPU) build of PyTorch (--torch-index cpu)."
  elif [[ -z "$TORCH_CHOICE" ]]; then
    echo "NVIDIA GPU detected - installing the CUDA build of PyTorch."
  else
    # unsafe-best-match: without it uv stops at the older torchao on the CUDA index, and PyPI's
    # plain torch outranks the +cuXXX build on Windows.
    # no-sources: the pyproject pin names one index, and the card decides here. The broad flag, not
    # --no-sources-package, which needs uv 0.10+; torch is the only entry so they are equivalent.
    TORCH_INDEX=(--extra-index-url "$(torch_index_url "$TORCH_CHOICE")" \
      --index-strategy unsafe-best-match --no-sources)
    echo "NVIDIA GPU detected - installing the CUDA build of PyTorch ($TORCH_CHOICE)."
  fi
  # A venv reused from a bad install keeps its torch: uv leaves a satisfying version alone, so
  # neither a corrected detector nor --torch-index would replace it. Ask the installed torch whether
  # it actually covers this card.
  TORCH_FORCE="$TORCH_EXPLICIT"
  if [[ "$TORCH_FORCE" -eq 0 && "$GPU_PRESENT" -eq 1 && "$TORCH_CHOICE" != "cpu" && -n "$TORCH_CHOICE" ]]; then
    PROBE_OUT="$("$TARGET_PY" -c 'from inline_core.device.probe import probe; p = probe(); print(p["status"], int(p["replaceable"]))' 2>/dev/null || true)"
    read -r PROBE_STATUS PROBE_REPLACEABLE <<<"${PROBE_OUT:-unknown 0}"
    if [[ "$PROBE_STATUS" == "uncovered" && "$PROBE_REPLACEABLE" == "1" ]]; then
      TORCH_FORCE=1
      echo "The installed torch has no kernels for this GPU; replacing it from $TORCH_CHOICE."
    elif [[ "$PROBE_STATUS" == "uncovered" ]]; then
      # Not a pytorch.org build: a ROCm wheel, a nightly or something hand-built also fails the arch
      # rule, and reinstalling over a deliberate choice is worse than the wrong wheel.
      echo "WARNING: the installed torch has no kernels for this GPU, but it is not a pytorch.org"
      echo "         build, so it is left alone. To rebuild the environment from scratch:"
      echo "         ./webui.sh --install --extra $EXTRAS --recreate"
    fi
  fi

  # The engine's distribution was renamed and both names own the `inline_core` files, so the old
  # one goes before the new one lands: two distributions owning one import path report versions
  # that disagree with each other.
  uv pip uninstall --python "$TARGET_PY" openchar-core inline-core >/dev/null 2>&1 || true
  echo "+ uv pip install --python $TARGET_PY ${TORCH_INDEX[*]} -e .[$EXTRAS]"
  uv pip install --python "$TARGET_PY" "${TORCH_INDEX[@]}" -e ".[$EXTRAS]"
  # Torch LAST and through --index-url (exclusive): [tool.uv.sources] pins it to cu126 on win32,
  # and --extra-index-url picks the highest version ACROSS indexes, so PyPI's CPU wheel can win.
  if [[ "$TORCH_FORCE" -eq 1 && -n "$TORCH_CHOICE" ]]; then
    read -r -a TORCH_PINS <<<"$(torch_pins_for "$TORCH_CHOICE")"
    TORCH_URL="$(torch_index_url "$TORCH_CHOICE")"
    echo "+ uv pip install --python $TARGET_PY --index-url $TORCH_URL --reinstall ${TORCH_PINS[*]}"
    uv pip install --python "$TARGET_PY" --index-url "$TORCH_URL" --reinstall "${TORCH_PINS[@]}"
  fi
  # Pull the prebuilt web UI so there's no Node build (best-effort - it may not be published yet).
  # --upgrade, because uv leaves an already-satisfied requirement alone: without it a re-run of
  # --install kept whatever UI was first installed while the engine moved on underneath it.
  if uv pip install --python "$TARGET_PY" --upgrade omnichar-frontend >/dev/null 2>&1; then
    FRONTEND_VERSION="$("$TARGET_PY" -c 'from importlib.metadata import version
print(version("omnichar-frontend"))' 2>/dev/null || true)"
    echo "Installed the prebuilt web UI (omnichar-frontend ${FRONTEND_VERSION:-unknown})."
  else
    echo "Note: omnichar-frontend not installed; the UI will build from source or run API-only."
  fi
  # A CPU-only wheel on a GPU box is silent at runtime and ~100x slower, so say it here rather than
  # let the resolve fail quietly - it can still happen if PyPI outranks the CUDA index on version.
  if [[ "$TORCH_CHOICE" != "cpu" ]] && ! "$TARGET_PY" -c 'import importlib.util, sys
if importlib.util.find_spec("torch") is None:
    sys.exit(0)
import torch
sys.exit(0 if torch.version.cuda else 1)' 2>/dev/null; then
    echo "WARNING: the torch that got installed is a CPU-ONLY build. Generation would run on the"
    echo "         CPU, roughly 100x slower. Re-run with an explicit index, e.g."
    # The extras and the index this run actually chose, not a fixed pair: a re-run without them
    # installs less than the person already had.
    echo "         ./webui.sh --install --extra $EXTRAS --torch-index $TORCH_CHOICE"
  fi
  # What the installer intended versus what landed. Nothing reads this yet; it exists so the next
  # bug report carries its own diagnosis instead of a guess. Beside the target interpreter, not a
  # hardcoded .venv, because --use-active-env means there may not be one.
  INLINE_RECORD_DIR="$(dirname "$(dirname "$TARGET_PY")")"
  INLINE_RECORD="$INLINE_RECORD_DIR/.inline-install.json" \
  INLINE_RECORD_INDEX="${TORCH_CHOICE:-default}" \
  INLINE_RECORD_REASON="$TORCH_INDEX_REASON" \
  INLINE_RECORD_CAP="${GPU_CAP_MAJOR:-}" \
  INLINE_RECORD_DRIVER="${GPU_DRIVER_MAJOR:-}" \
  INLINE_RECORD_RAW="${GPU_PROBE_RAW:-}" \
  INLINE_RECORD_EXTRAS="$EXTRAS" \
    "$TARGET_PY" -c 'import json, os
from inline_core.device.probe import probe
rec = {
    "index": os.environ["INLINE_RECORD_INDEX"],
    "reason": os.environ["INLINE_RECORD_REASON"],
    "capabilityMajor": os.environ.get("INLINE_RECORD_CAP") or None,
    "driverMajor": os.environ.get("INLINE_RECORD_DRIVER") or None,
    "probeRaw": os.environ.get("INLINE_RECORD_RAW") or None,
    "extras": os.environ["INLINE_RECORD_EXTRAS"],
    "installed": probe(),
}
with open(os.environ["INLINE_RECORD"], "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=2)
' 2>/dev/null || true
  CORE_VERSION="$("$TARGET_PY" -c 'from importlib.metadata import version
print(version("omnichar-core"))' 2>/dev/null || true)"
  echo "Installed omnichar-core ${CORE_VERSION:-unknown} with extras: $EXTRAS. Start with: ./webui.sh"
  exit 0
fi

# Pick the Python interpreter and the matching pip, in priority order. Our own .venv outranks an env
# that merely happens to be activated, so a foreign venv can never absorb the on-demand installs below.
if [[ "$USE_ACTIVE_ENV" -eq 1 ]]; then
  [[ -n "$ACTIVE_ENV" ]] || { echo "--use-active-env needs an activated environment." >&2; exit 1; }
  PY="$ACTIVE_ENV/bin/python"
elif [[ -x "$VENV_PY" ]]; then
  PY="$VENV_PY"
  if FOREIGN="$(foreign_env)"; then
    echo "NOTE: running from $VENV_DIR, not the environment active in this shell ($FOREIGN)."
  fi
elif [[ -n "$ACTIVE_ENV" && -x "$ACTIVE_ENV/bin/python" ]]; then
  PY="$ACTIVE_ENV/bin/python"      # omnichar-core pip-installed into the user's own environment
elif python3 -c "import inline_core" >/dev/null 2>&1; then
  PY="$(command -v python3)"       # ...or onto the ambient interpreter (pip/pipx)
else
  echo "No .venv found. Run './webui.sh --install' first." >&2
  exit 1
fi
PY_CMD=("$PY")
# uv venvs are not seeded with pip, so 'python -m pip' can never work in one - go through uv instead.
if command -v uv >/dev/null 2>&1; then
  PIP_INSTALL=(uv pip install --python "$PY")
else
  PIP_INSTALL=("$PY" -m pip install)
fi

# True when a web UI is resolvable: an explicit --front-end-root, the installed frontend package, or
# a local dist-web/ build (the repo checkout). Sets INLINE_FRONTEND_ROOT for the local-build case.
frontend_available() {
  if [[ -n "${INLINE_FRONTEND_ROOT:-}" && -f "${INLINE_FRONTEND_ROOT}/index.html" ]]; then return 0; fi
  if "${PY_CMD[@]}" - <<'PY' >/dev/null 2>&1; then return 0; fi
import os, sys
from importlib import import_module

# Both names, new first, so an install predating the rename still counts as having a UI.
for name in ("omnichar_frontend", "openchar_frontend"):
    try:
        f = import_module(name)
    except ModuleNotFoundError:
        continue
    index = os.path.join(os.path.dirname(f.__file__), "static", "index.html")
    if os.path.isfile(index):
        sys.exit(0)
sys.exit(1)
PY
  if [[ -f "../dist-web/index.html" ]]; then
    INLINE_FRONTEND_ROOT="$(cd .. && pwd)/dist-web"; export INLINE_FRONTEND_ROOT; return 0
  fi
  return 1
}

# Force a fresh local SPA build and point Core at it (overriding any installed frontend package), so a
# one-port run reflects the current source. Used by --rebuild.
rebuild_frontend() {
  if [[ ! -f "../package.json" ]] || ! command -v npm >/dev/null 2>&1; then
    echo "--rebuild needs the repo checkout and npm; skipping the rebuild." >&2
    return 0
  fi
  echo "Rebuilding the web UI from source (npm run build:spa)…"
  ( cd .. && { [[ -d node_modules ]] || npm ci; } && npm run build:spa )
  INLINE_FRONTEND_ROOT="$(cd .. && pwd)/dist-web"; export INLINE_FRONTEND_ROOT
}

# Live-reload dev loop: Core (API) in the background, Vite (HMR) in the foreground proxying to it. The
# user opens the Vite port (5173), not Core's - edits to src/ hot-reload without a rebuild. Used by --dev.
run_dev() {
  if [[ ! -f "../package.json" ]] || ! command -v npm >/dev/null 2>&1; then
    echo "--dev needs the repo checkout and Node/npm (https://nodejs.org/)." >&2
    exit 1
  fi
  ( cd .. && { [[ -d node_modules ]] || npm ci; } )
  echo "Starting Inline Core (API) on ${HOST}:${PORT} in the background…"
  "${PY_CMD[@]}" -m inline_core.server &
  local core_pid=$!
  trap 'kill "$core_pid" 2>/dev/null || true' EXIT INT TERM
  echo "Starting the Vite dev server (HMR). Open http://localhost:5173  (NOT :${PORT})"
  ( cd .. && INLINE_CORE_URL="http://127.0.0.1:${PORT}" npm run dev:web )
}

# Smart memory needs torchao (int8 weight-only quant). It ships in the runtime extra, but an older
# venv predates it - install it on demand at launch so --smart-memory just works. Best-effort: if the
# install fails, generation still runs, only without int8 (the loader logs and loads full precision).
ensure_smart_memory_deps() {
  "${PY_CMD[@]}" -c "import torchao" >/dev/null 2>&1 && return 0
  echo "Smart memory: installing torchao (int8 quantization)…"
  "${PIP_INSTALL[@]}" torchao >/dev/null \
    && echo "Installed torchao." \
    || echo "WARNING: could not install torchao; smart memory will run without int8 quant." >&2
}

# Make sure a UI is present before serving: try the pip package, then a local npm build, else warn.
ensure_frontend() {
  frontend_available && return 0
  echo "No web UI found - installing the prebuilt package (omnichar-frontend)…"
  "${PIP_INSTALL[@]}" omnichar-frontend >/dev/null && frontend_available && return 0
  if [[ -f "../package.json" ]] && command -v npm >/dev/null 2>&1; then
    echo "Building the web UI from source (npm)…"
    ( cd .. && npm ci && npm run build:spa ) && frontend_available && return 0
  fi
  echo "WARNING: no web UI available - serving API only. Install Node to build it, or run" >&2
  echo "         '${PIP_INSTALL[*]} omnichar-frontend' once it's published." >&2
  return 0
}

if [[ "$DEV_MODE" -eq 1 ]]; then
  run_dev
  exit 0
fi

if [[ "$FORCE_REBUILD" -eq 1 ]]; then
  rebuild_frontend
fi

if [[ "$SMART_MEMORY" -eq 1 ]]; then
  ensure_smart_memory_deps
fi

ensure_frontend
exec "${PY_CMD[@]}" -m inline_core.server
