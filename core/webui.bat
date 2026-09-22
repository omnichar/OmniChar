@echo off
rem ===========================================================================================
rem Launch Inline Studio (Inline Core + the web UI) on one port, on Windows.
rem
rem This is the Windows twin of webui.sh - keep the two in sync. Friendly flags map onto the
rem engine's INLINE_* environment knobs, so you do not have to remember them. Double-click this
rem file, or run it from PowerShell/cmd:
rem
rem   .\webui.bat                             loopback, port 8848 (UI + API)
rem   .\webui.bat --listen --port 9000        bind all interfaces on 9000
rem   .\webui.bat --install --extra runtime   set up .venv with the model runtime, then exit
rem
rem Run  .\webui.bat --help  for every flag.
rem ===========================================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Absolute, so uv is pinned to this venv no matter what is activated in the shell.
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

set "HOST=127.0.0.1"
set "PORT=8848"
set "EXTRAS=server"
set "RUN_INSTALL=0"
set "DEV_MODE=0"
set "FORCE_REBUILD=0"
set "SMART_MEMORY=0"
set "USE_ACTIVE_ENV=0"
set "RECREATE=0"
set "PRINT_TORCH_INDEX=0"
set "TORCH_EXPLICIT=0"
set "TORCH_INDEX_REASON=autodetect"
rem Empty means "decide from the GPU's compute capability at install time" (see :install_torch_args).
set "TORCH_CHOICE=%INLINE_TORCH_INDEX%"

:parse
if "%~1"=="" goto after_parse
if /i "%~1"=="--listen"       ( set "HOST=0.0.0.0" & shift & goto parse )
if /i "%~1"=="--host"         ( set "HOST=%~2" & shift & shift & goto parse )
if /i "%~1"=="--port"         ( set "PORT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--multi-gpu"    goto multigpu
if /i "%~1"=="--parallel"     goto multigpu
if /i "%~1"=="--lowvram"      ( set "INLINE_PROFILE=lowvram" & shift & goto parse )
if /i "%~1"=="--smart-memory" goto smartmem
if /i "%~1"=="--cpu"          ( set "INLINE_PROFILE=cpu" & shift & goto parse )
if /i "%~1"=="--profile"      ( set "INLINE_PROFILE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--vram-budget"  ( set "INLINE_VRAM_BUDGET_GB=%~2" & shift & shift & goto parse )
if /i "%~1"=="--models-dir"   ( set "INLINE_MODELS_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--data-dir"     ( set "INLINE_DATA_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--install"      ( set "RUN_INSTALL=1" & shift & goto parse )
if /i "%~1"=="--extra"        ( set "EXTRAS=!EXTRAS!,%~2" & shift & shift & goto parse )
if /i "%~1"=="--torch-index"  ( set "TORCH_CHOICE=%~2" & set "TORCH_EXPLICIT=1" & shift & shift & goto parse )
if /i "%~1"=="--print-torch-index" ( set "PRINT_TORCH_INDEX=1" & shift & goto parse )
if /i "%~1"=="--recreate"     ( set "RECREATE=1" & shift & goto parse )
if /i "%~1"=="--use-active-env" ( set "USE_ACTIVE_ENV=1" & shift & goto parse )
if /i "%~1"=="--dev"          ( set "DEV_MODE=1" & shift & goto parse )
if /i "%~1"=="--rebuild"      ( set "FORCE_REBUILD=1" & shift & goto parse )
if /i "%~1"=="-h"             ( call :usage & exit /b 0 )
if /i "%~1"=="--help"         ( call :usage & exit /b 0 )
echo unknown option: %~1
call :usage
goto fail

:multigpu
rem Multi-GPU split is auto-detected with 2+ GPUs; an optional SPEC (e.g. pipefusion=2) overrides it.
set "SPEC=%~2"
if "!SPEC!"=="" (
  echo Multi-GPU split is auto-detected with 2+ GPUs; pass e.g. pipefusion=2 to override.
  shift & goto parse
)
if "!SPEC:~0,1!"=="-" (
  echo Multi-GPU split is auto-detected with 2+ GPUs; pass e.g. pipefusion=2 to override.
  shift & goto parse
)
set "INLINE_PARALLEL=!SPEC!"
shift & shift & goto parse

:smartmem
rem Spread a too-big model across VRAM + RAM + CPU; force lowvram so the offload + int8 path engages.
set "SMART_MEMORY=1"
set "INLINE_SMART_MEMORY=1"
if not defined INLINE_PROFILE set "INLINE_PROFILE=lowvram"
shift & goto parse

:after_parse

set "INLINE_HOST=%HOST%"
set "INLINE_PORT=%PORT%"

rem Expandable CUDA segments cut low-VRAM fragmentation OOMs (harmless on CPU). Respect a user value.
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

set "ACTIVE_ENV="
if defined CONDA_PREFIX set "ACTIVE_ENV=%CONDA_PREFIX%"
if defined VIRTUAL_ENV set "ACTIVE_ENV=%VIRTUAL_ENV%"
call :foreign_env

if "%PRINT_TORCH_INDEX%"=="1" goto print_torch_index
if "%RUN_INSTALL%"=="1" goto do_install
goto pick_python

rem --- Install: create .venv (via uv) and install, then exit -------------------------------------
rem Flat label-based control, not nested (...) blocks: multi-line parens inside parens break .bat.
:do_install
where uv >nul 2>nul || ( echo uv not found: https://docs.astral.sh/uv/ & goto fail )
call :normalize_extras || goto fail
rem Every uv call below is pinned with --python. uv's pip interface otherwise targets an active
rem VIRTUAL_ENV/CONDA_PREFIX ahead of the local .venv, which is how an unrelated venv activated in the
rem user's shell silently absorbed the install and had its packages replaced.
set "TARGET_PY=%VENV_PY%"
if "%USE_ACTIVE_ENV%"=="1" goto install_active_env
if defined FOREIGN_ENV echo NOTE: the environment active in this shell will NOT be modified:
if defined FOREIGN_ENV echo         %FOREIGN_ENV%
if defined FOREIGN_ENV echo       Installing into %VENV_DIR% instead ^(--use-active-env overrides this^).
if "%RECREATE%"=="1" goto install_recreate
rem uv venv refuses an existing environment and --clear would wipe manual installs, so a repeat
rem --install (say, to add an extra) has to reuse what is already there.
if exist "%VENV_PY%" ( echo Reusing the existing environment at %VENV_DIR% ^(--recreate rebuilds it from scratch^). & goto install_torch_args )
uv venv "%VENV_DIR%" || goto fail
goto install_torch_args

:install_recreate
echo Recreating %VENV_DIR% - anything installed into it by hand is lost.
uv venv --clear "%VENV_DIR%" || goto fail
goto install_torch_args

:install_active_env
if not defined ACTIVE_ENV ( echo --use-active-env needs an activated environment. & goto fail )
set "TARGET_PY=%ACTIVE_ENV%\Scripts\python.exe"
echo Installing into the active environment: %ACTIVE_ENV%

:install_torch_args
rem PyPI's default torch is CPU-only on Windows, so installing blind generates on the CPU ~100x
rem slower with no error. Which CUDA index is right depends on the card: Blackwell (sm_120) has no
rem wheels before cu128, while cu126 is the last index still built for Maxwell..Volta (sm_50..sm_70).
set "TORCH_ARGS="
if defined TORCH_CHOICE ( set "TORCH_INDEX_REASON=override" & goto install_torch_index )
set "NO_GPU_WHY=nvidia-smi is not on PATH"
where nvidia-smi >nul 2>nul || goto install_cpu
set "NO_GPU_WHY=nvidia-smi ran but listed no GPU"
rem `call`, because nvidia-smi on PATH is not always an .exe. A .bat or .cmd shim would otherwise
rem take over this script and never return, leaving no output and no error.
call nvidia-smi -L >nul 2>nul || goto install_cpu
call :read_gpu_probe
echo GPU probe (compute_cap, driver_version): !GPU_PROBE_RAW!
set "TORCH_CHOICE=cu126"
if !CAP_MAJOR! GEQ 10 set "TORCH_CHOICE=cu130"
rem Verified 2026-02 and load-bearing: cu128 still SERVES but is frozen at torch 2.11.0, and it was
rem the first index with sm_120. So a Blackwell card on a pre-R580 driver (CUDA 13's floor) has
rem exactly one workable choice; hand it that rather than erroring. Re-check the ceiling in a year.
if !CAP_MAJOR! GEQ 10 if !DRIVER_MAJOR! GTR 0 if !DRIVER_MAJOR! LSS 580 (
  set "TORCH_CHOICE=cu128"
  set "TORCH_INDEX_REASON=driver-floor-cu128"
  echo Driver !DRIVER_MAJOR!.xx predates R580, which CUDA 13 needs. Installing from cu128,
  echo which is frozen at torch 2.11 and will never update. Updating the NVIDIA driver and
  echo re-running --install gets you current torch via cu130.
)

:install_torch_index
if /i "!TORCH_CHOICE!"=="cpu" goto install_cpu_forced
set "TORCH_URL=https://download.pytorch.org/whl/!TORCH_CHOICE!"
if /i "!TORCH_CHOICE:~0,4!"=="http" set "TORCH_URL=!TORCH_CHOICE!"
rem unsafe-best-match: without it uv stops at the older torchao on the CUDA index, and PyPI's plain
rem torch outranks the +cuXXX build on Windows.
rem no-sources: the pyproject pin names one index, and the card decides here. The broad flag, not
rem --no-sources-package, which needs uv 0.10+; torch is the only entry so they are equivalent.
set "TORCH_ARGS=--extra-index-url !TORCH_URL! --index-strategy unsafe-best-match --no-sources"
echo NVIDIA GPU detected - installing the CUDA build of PyTorch (!TORCH_CHOICE!).
goto install_pkgs

:install_cpu_forced
echo Installing the default (CPU) build of PyTorch (--torch-index cpu).
goto install_pkgs

:install_cpu
set "TORCH_CHOICE=cpu"
set "TORCH_INDEX_REASON=no-gpu"
echo No NVIDIA GPU detected (!NO_GPU_WHY!) - installing the default (CPU) build of PyTorch.
echo   On an AMD or Intel GPU this is not what you want: pass a full index URL, e.g.
echo   .\webui.bat --install --torch-index https://download.pytorch.org/whl/rocm6.2
echo   See the README section "AMD (ROCm) setup".

:install_pkgs
rem A venv reused from a bad install keeps its torch: uv leaves a satisfying version alone, so
rem neither a corrected detector nor --torch-index would replace it. Ask the installed torch whether
rem it actually has kernels for this card.
set "TORCH_FORCE=!TORCH_EXPLICIT!"
if "!TORCH_FORCE!"=="1" goto install_run
if /i "!TORCH_CHOICE!"=="cpu" goto install_run
if not defined TORCH_CHOICE goto install_run
set "PROBE_STATUS=unknown"
set "PROBE_REPLACEABLE=0"
for /f "usebackq tokens=1,2" %%a in (`"!TARGET_PY!" -c "from inline_core.device.probe import probe; p = probe(); print(p['status'], int(p['replaceable']))" 2^>nul`) do (
  set "PROBE_STATUS=%%a"
  set "PROBE_REPLACEABLE=%%b"
)
if /i "!PROBE_STATUS!"=="uncovered" (
  if "!PROBE_REPLACEABLE!"=="1" (
    set "TORCH_FORCE=1"
    echo The installed torch has no kernels for this GPU; replacing it from !TORCH_CHOICE!.
  ) else (
    rem Not a pytorch.org build: a ROCm wheel, a nightly or something hand-built also fails the arch
    rem rule, and reinstalling over a deliberate choice is worse than the wrong wheel.
    echo WARNING: the installed torch has no kernels for this GPU, but it is not a pytorch.org
    echo          build, so it is left alone. To rebuild the environment from scratch:
    echo          .\webui.bat --install --extra !EXTRAS! --recreate
  )
)

:install_run
echo + uv pip install --python "!TARGET_PY!" !TORCH_ARGS! -e ".[!EXTRAS!]"
uv pip install --python "!TARGET_PY!" !TORCH_ARGS! -e ".[!EXTRAS!]" || goto fail
rem Torch LAST and through --index-url (exclusive): [tool.uv.sources] pins it to cu126 on win32,
rem and --extra-index-url picks the highest version ACROSS indexes, so PyPI's CPU wheel can win.
if "!TORCH_FORCE!"=="1" if defined TORCH_CHOICE (
  set "TORCH_PINS=torch torchvision"
  rem cu128 is frozen, so the current pair does not exist there. Pin the last one it has.
  if /i "!TORCH_CHOICE!"=="cu128" set "TORCH_PINS=torch==2.11.0 torchvision==0.26.0"
  set "TORCH_URL=https://download.pytorch.org/whl/!TORCH_CHOICE!"
  if /i "!TORCH_CHOICE:~0,4!"=="http" set "TORCH_URL=!TORCH_CHOICE!"
  echo + uv pip install --python "!TARGET_PY!" --index-url !TORCH_URL! --reinstall !TORCH_PINS!
  uv pip install --python "!TARGET_PY!" --index-url !TORCH_URL! --reinstall !TORCH_PINS! || goto fail
)
rem --upgrade, because uv leaves an already-satisfied requirement alone: without it a re-run of
rem --install kept whatever UI was first installed while the engine moved on underneath it.
set "FRONTEND_VERSION="
uv pip install --python "!TARGET_PY!" --upgrade omnichar-frontend >nul 2>nul && (
  for /f "usebackq delims=" %%v in (`"!TARGET_PY!" -c "from importlib.metadata import version; print(version('omnichar-frontend'))" 2^>nul`) do set "FRONTEND_VERSION=%%v"
  if not defined FRONTEND_VERSION set "FRONTEND_VERSION=unknown"
  echo Installed the prebuilt web UI ^(omnichar-frontend !FRONTEND_VERSION!^).
) || echo Note: omnichar-frontend not installed; the UI will build from source or run API-only.
call :write_install_record
rem A CPU-only wheel on a GPU box is silent at runtime and ~100x slower, so say it here rather than
rem let it through: it can still happen if PyPI ever outranks the CUDA index on version.
if /i "!TORCH_CHOICE!"=="cpu" goto install_done
rem TARGET_PY, not PY which is unset until :pick_python, and a torch probe not a web UI one: the question here is whether the wheel that landed carries CUDA.
"!TARGET_PY!" -c "import importlib.util as u,sys;sys.exit(0 if u.find_spec('torch') is None else (0 if __import__('torch').version.cuda else 1))" >nul 2>nul && goto install_done
echo WARNING: the torch that got installed is a CPU-ONLY build. Generation would run on the
echo          CPU, roughly 100x slower. Re-run with an explicit index, e.g.
echo          .\webui.bat --install --torch-index !TORCH_CHOICE! --recreate

:install_done
set "CORE_VERSION=unknown"
for /f "usebackq delims=" %%v in (`"!TARGET_PY!" -c "from importlib.metadata import version; print(version('omnichar-core'))" 2^>nul`) do set "CORE_VERSION=%%v"
echo Installed omnichar-core !CORE_VERSION! with extras: !EXTRAS!. Start with: .\webui.bat
exit /b 0

rem One query for both, keeping the highest capability. set /a not a findstr guard: `if LSS` would
rem string-compare an error string like "Unknown" above 10; set /a yields 0 for it, as intended.
:read_gpu_probe
set "CAP_MAJOR=0"
set "DRIVER_MAJOR=0"
set "GPU_PROBE_RAW="
rem Comma ONLY: adding "." would cut "12.0, 610.88" into 12/0/610/88, making token 2 the minor
rem rather than the driver, so the R580 floor could never fire.
for /f "usebackq tokens=1,2 delims=," %%c in (`nvidia-smi --query-gpu^=compute_cap^,driver_version --format^=csv^,noheader 2^>nul`) do (
  rem Reset every iteration: set /a errors on garbage and would otherwise leave the previous line's
  rem value in place, double-counting a good line followed by a bad one.
  set "CAP_TRY=0"
  set "DRV_TRY=0"
  for /f "tokens=1 delims=. " %%m in ("%%c") do set /a "CAP_TRY=%%m" 2>nul
  for /f "tokens=1 delims=. " %%n in ("%%d") do set /a "DRV_TRY=%%n" 2>nul
  if not defined GPU_PROBE_RAW ( set "GPU_PROBE_RAW=%%c,%%d" ) else ( set "GPU_PROBE_RAW=!GPU_PROBE_RAW!; %%c,%%d" )
  if !CAP_TRY! GTR !CAP_MAJOR! (
    set "CAP_MAJOR=!CAP_TRY!"
    set "DRIVER_MAJOR=!DRV_TRY!"
  )
)
if not defined GPU_PROBE_RAW set "GPU_PROBE_RAW=<query returned nothing>"
exit /b 0

rem --- Pick the Python interpreter (and matching pip), in priority order -------------------------
rem Our own .venv outranks an env that merely happens to be activated, so a foreign venv can never
rem absorb the on-demand installs in :ensure_frontend / :ensure_smart_memory_deps.
:pick_python
if "%USE_ACTIVE_ENV%"=="1" goto pick_active_env
if exist "%VENV_PY%" goto pick_venv
if defined ACTIVE_ENV if exist "%ACTIVE_ENV%\Scripts\python.exe" ( set "PY=%ACTIVE_ENV%\Scripts\python.exe" & goto have_pip )
python -c "import inline_core" >nul 2>nul && ( set "PY=python" & goto have_pip )
echo No .venv found. Run  .\webui.bat --install  first.
goto fail

:pick_venv
set "PY=%VENV_PY%"
if defined FOREIGN_ENV echo NOTE: running from %VENV_DIR%, not the environment active in this shell ^(%FOREIGN_ENV%^).
goto have_pip

:pick_active_env
if not defined ACTIVE_ENV ( echo --use-active-env needs an activated environment. & goto fail )
set "PY=%ACTIVE_ENV%\Scripts\python.exe"

:have_pip
rem uv venvs are not seeded with pip, so 'python -m pip' can never work in one - go through uv instead.
set PIP="%PY%" -m pip install
where uv >nul 2>nul && set PIP=uv pip install --python "%PY%"
goto have_python

:have_python
if "%DEV_MODE%"=="1" ( call :run_dev & exit /b !ERRORLEVEL! )
if "%FORCE_REBUILD%"=="1" call :rebuild_frontend
if "%SMART_MEMORY%"=="1" call :ensure_smart_memory_deps

call :ensure_frontend

"%PY%" -m inline_core.server
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" ( echo. & echo Inline Studio exited with code %RC%. & pause )
exit /b %RC%

rem ================================ subroutines =================================================

:foreign_env
rem Sets FOREIGN_ENV when the environment active in this shell is NOT ours. Everything keys off this:
rem an env that merely happens to be activated must never absorb an install or a launch-time dep.
set "FOREIGN_ENV="
if not defined ACTIVE_ENV exit /b 0
for %%I in ("%ACTIVE_ENV%") do set "ACTIVE_FULL=%%~fI"
if /i "%ACTIVE_FULL%"=="%VENV_DIR%" exit /b 0
set "FOREIGN_ENV=%ACTIVE_FULL%"
exit /b 0

:normalize_extras
rem Validate and dedupe the extras, so a typo fails here with the valid list instead of reaching uv as
rem a raw resolver error. The names must match [project.optional-dependencies] in pyproject.toml.
set "EXTRAS_OUT="
for %%E in ("%EXTRAS:,=" "%") do call :add_extra %%E || exit /b 1
set "EXTRAS=!EXTRAS_OUT!"
exit /b 0

:add_extra
echo runtime server parallel training dev all | findstr /r /c:"\<%~1\>" >nul || goto bad_extra
if "!EXTRAS_OUT!"=="" ( set "EXTRAS_OUT=%~1" & exit /b 0 )
echo !EXTRAS_OUT! | findstr /r /c:"\<%~1\>" >nul && exit /b 0
set "EXTRAS_OUT=!EXTRAS_OUT!,%~1"
exit /b 0

:bad_extra
echo unknown extra: %~1 ^(valid: runtime, server, parallel, training, dev, all^)
exit /b 1

:frontend_available
rem Succeeds (errorlevel 0) when a web UI is resolvable; sets INLINE_FRONTEND_ROOT for a local build.
if defined INLINE_FRONTEND_ROOT if exist "%INLINE_FRONTEND_ROOT%\index.html" exit /b 0
rem Both names, new first, so an install predating the rename still counts as having a UI.
rem util imported by name, never reached as an attribute: a uv venv has no .pth file to pull it in, so the attribute form raised AttributeError and the discarded stderr hid it.
"%PY%" -c "import os,sys;from importlib import util,import_module;found=[n for n in ('omnichar_frontend','openchar_frontend') if util.find_spec(n)];sys.exit(0 if any(os.path.isfile(os.path.join(os.path.dirname(import_module(n).__file__),'static','index.html')) for n in found) else 1)" >nul 2>nul && exit /b 0
if exist "..\dist-web\index.html" (
  for %%I in ("..\dist-web") do set "INLINE_FRONTEND_ROOT=%%~fI"
  exit /b 0
)
exit /b 1

:ensure_frontend
call :frontend_available && exit /b 0
echo No web UI found - installing the prebuilt package (omnichar-frontend)...
%PIP% omnichar-frontend >nul
call :frontend_available && exit /b 0
if exist "..\package.json" (
  where npm >nul 2>nul && (
    rem Parens escaped: cmd ends the enclosing block at a bare one and then chokes on the ellipsis.
    echo Building the web UI from source ^(npm^)...
    pushd ..
    call npm ci && call npm run build:spa
    popd
    call :frontend_available && exit /b 0
  )
)
echo WARNING: no web UI available - serving API only. Install Node to build it, or run
echo          %PIP% omnichar-frontend   once it's published.
exit /b 0

:rebuild_frontend
if not exist "..\package.json" ( echo --rebuild needs the repo checkout and npm; skipping the rebuild. & exit /b 0 )
where npm >nul 2>nul || ( echo --rebuild needs npm; skipping the rebuild. & exit /b 0 )
echo Rebuilding the web UI from source (npm run build:spa)...
pushd ..
if not exist node_modules call npm ci
call npm run build:spa
popd
for %%I in ("..\dist-web") do set "INLINE_FRONTEND_ROOT=%%~fI"
exit /b 0

:ensure_smart_memory_deps
"%PY%" -c "import torchao" >nul 2>nul && exit /b 0
echo Smart memory: installing torchao (int8 quantization)...
%PIP% torchao >nul ^
  && echo Installed torchao. ^
  || echo WARNING: could not install torchao; smart memory will run without int8 quant.
exit /b 0

:run_dev
if not exist "..\package.json" ( echo --dev needs the repo checkout and Node/npm. & goto fail )
where npm >nul 2>nul || ( echo --dev needs Node/npm ^(https://nodejs.org/^). & goto fail )
pushd ..
if not exist node_modules call npm ci
popd
echo Starting Inline Core (API) on %HOST%:%PORT% in a new window...
start "Inline Core (API)" "%PY%" -m inline_core.server
echo Starting the Vite dev server (HMR). Open http://localhost:5173  (NOT :%PORT%)
pushd ..
set "INLINE_CORE_URL=http://127.0.0.1:%PORT%"
call npm run dev:web
popd
exit /b 0

rem Detect-only: report the probe and the decision, change nothing. This is what CI asserts against
rem and what a bug report should paste, instead of a whole reinstall log.
:print_torch_index
if defined TORCH_CHOICE (
  set "TORCH_INDEX_REASON=override"
  set "GPU_PROBE_RAW=<not queried: index given>"
  set "CAP_MAJOR=0"
  set "DRIVER_MAJOR=0"
  where nvidia-smi >nul 2>nul && call :read_gpu_probe
) else (
  set "GPU_PROBE_RAW=<nvidia-smi unavailable>"
  set "CAP_MAJOR=0"
  set "DRIVER_MAJOR=0"
  set "TORCH_CHOICE=cpu"
  set "TORCH_INDEX_REASON=no-gpu"
  where nvidia-smi >nul 2>nul && call nvidia-smi -L >nul 2>nul && call :decide_print_index
)
echo probe: !GPU_PROBE_RAW!
if "!CAP_MAJOR!"=="0" ( echo capability-major: unknown ) else ( echo capability-major: !CAP_MAJOR! )
if "!DRIVER_MAJOR!"=="0" ( echo driver-major: unknown ) else ( echo driver-major: !DRIVER_MAJOR! )
echo torch-index: !TORCH_CHOICE!
echo reason: !TORCH_INDEX_REASON!
exit /b 0

:decide_print_index
call :read_gpu_probe
set "TORCH_CHOICE=cu126"
set "TORCH_INDEX_REASON=autodetect"
if !CAP_MAJOR! GEQ 10 set "TORCH_CHOICE=cu130"
if !CAP_MAJOR! GEQ 10 if !DRIVER_MAJOR! GTR 0 if !DRIVER_MAJOR! LSS 580 (
  set "TORCH_CHOICE=cu128"
  set "TORCH_INDEX_REASON=driver-floor-cu128"
)
exit /b 0

rem What the installer intended versus what landed. Nothing reads this yet; it exists so the next
rem bug report carries its own diagnosis. Beside the target interpreter, not a hardcoded .venv,
rem because --use-active-env means there may not be one.
:write_install_record
for %%I in ("!TARGET_PY!") do set "RECORD_DIR=%%~dpI.."
set "INLINE_RECORD=!RECORD_DIR!\.inline-install.json"
set "INLINE_RECORD_INDEX=!TORCH_CHOICE!"
if not defined INLINE_RECORD_INDEX set "INLINE_RECORD_INDEX=default"
set "INLINE_RECORD_REASON=!TORCH_INDEX_REASON!"
set "INLINE_RECORD_CAP=!CAP_MAJOR!"
set "INLINE_RECORD_DRIVER=!DRIVER_MAJOR!"
set "INLINE_RECORD_RAW=!GPU_PROBE_RAW!"
set "INLINE_RECORD_EXTRAS=!EXTRAS!"
"!TARGET_PY!" -c "import json, os; from inline_core.device.probe import probe; rec = {'index': os.environ['INLINE_RECORD_INDEX'], 'reason': os.environ['INLINE_RECORD_REASON'], 'capabilityMajor': os.environ.get('INLINE_RECORD_CAP') or None, 'driverMajor': os.environ.get('INLINE_RECORD_DRIVER') or None, 'probeRaw': os.environ.get('INLINE_RECORD_RAW') or None, 'extras': os.environ['INLINE_RECORD_EXTRAS'], 'installed': probe()}; open(os.environ['INLINE_RECORD'], 'w', encoding='utf-8').write(json.dumps(rec, indent=2))" 2>nul
exit /b 0

:usage
echo Usage: .\webui.bat [options]
echo.
echo Networking
echo   --listen               bind all interfaces (0.0.0.0), so other machines can reach it
echo   --host ADDR            bind a specific address (default 127.0.0.1)
echo   --port N               listen on port N (default 8848)
echo.
echo Multi-GPU (split one image's denoise across GPUs)
echo   --multi-gpu [SPEC]     enable the split; auto-detected with 2+ GPUs. Optional SPEC, e.g.
echo                          pipefusion=2 or pipefusion=2,ulysses=2
echo   --parallel SPEC        alias for --multi-gpu SPEC
echo.
echo Device / memory
echo   --lowvram              tight-VRAM profile (slicing + tiling + int8, weights resident)
echo   --smart-memory         spread a too-big model across VRAM + RAM + CPU (offload + int8)
echo   --cpu                  force the CPU profile
echo   --profile NAME         set the profile explicitly (gpu-max ^| lowvram ^| cpu)
echo   --vram-budget GB       treat the GPU as having GB of usable VRAM
echo.
echo Paths
echo   --models-dir PATH      where weights are scanned from (default .\models)
echo   --data-dir PATH        where runs and takes are written (default .\.inline)
echo.
echo Setup
echo   --install              create .venv (via uv) and install, then exit. An existing .venv is
echo                          reused, and an unrelated environment activated in your shell is never
echo                          touched.
echo   --extra NAME           add an install extra (repeatable): runtime, parallel, server, training
echo   --torch-index WHICH    with --install, override the PyTorch wheel index picked from your GPU's
echo                          compute capability. A short name (cu130, cu128, cu126), a full index
echo                          URL, or "cpu" to force the CPU-only build. Also settable as
echo                          INLINE_TORCH_INDEX. cu128 is frozen at torch 2.11 and exists only for
echo                          Blackwell cards whose driver predates R580; --install picks it for
echo                          you in that case.
echo   --print-torch-index    print what the GPU probe read and which index would be used, then
echo                          exit without installing anything
echo   --recreate             with --install, rebuild .venv from scratch (discards anything installed
echo                          into it by hand)
echo   --use-active-env       install into / run from the environment activated in this shell instead
echo                          of .venv
echo   -h, --help             show this help
echo.
echo Development
echo   --dev                  live-reload dev loop (Core in a new window + Vite HMR on :5173)
echo   --rebuild              force a fresh SPA build and serve it on the one port
exit /b 0

:fail
echo.
pause
exit /b 1
