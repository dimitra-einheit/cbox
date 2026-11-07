#!/usr/bin/env bash
# dronemap setup for macOS (Apple Silicon). Safe to run again at any time.
#
#   setup/setup_mac.sh              Python environment + Docker Desktop + OpenDroneMap
#   setup/setup_mac.sh --opensplat  ...and build OpenSplat for Gaussian splats (needs Xcode)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$HERE/tools"
WANT_SPLAT=0
TORCH_VER="2.3.1"                                          # libtorch version OpenSplat's macOS CI builds with
OPENSPLAT_REF="e33f9dde8ef9c9d59852ffe4d12e0adddcf1fc1b"   # OpenSplat commit this toolkit was written against

for arg in "$@"; do
  case "$arg" in
    --opensplat) WANT_SPLAT=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

say()  { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[33mWARNING: %s\033[0m\n" "$*"; }
die()  { printf "\033[31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "This script is for macOS. On Windows follow setup/SETUP_WINDOWS.md"
[ "$(uname -m)" = "arm64" ] || warn "This Mac is not Apple Silicon: Gaussian splats would run on the CPU (very slow)."

# ---------------------------------------------------------------- 1. Homebrew
if ! command -v brew >/dev/null 2>&1; then
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  else
    say "Installing Homebrew (it will ask for your Mac password)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
    grep -q 'brew shellenv' "$HOME/.zprofile" 2>/dev/null || \
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
  fi
fi
export HOMEBREW_NO_AUTO_UPDATE=1

# ---------------------------------------------------------------- 2. Python environment
say "Python environment"
PYBIN=""
for v in 3.12 3.13 3.11; do
  if command -v "python$v" >/dev/null 2>&1; then PYBIN="$(command -v "python$v")"; break; fi
done
if [ -z "$PYBIN" ]; then
  brew install python@3.12
  PYBIN="$(brew --prefix python@3.12)/bin/python3.12"
fi
echo "Using $PYBIN"
[ -x "$HERE/.venv/bin/python" ] || "$PYBIN" -m venv "$HERE/.venv"
"$HERE/.venv/bin/python" -m pip install --quiet --upgrade pip
"$HERE/.venv/bin/python" -m pip install --quiet -r "$HERE/requirements.txt"
"$HERE/.venv/bin/python" -c "import numpy, scipy, rasterio, matplotlib, PIL; print('Python packages OK')"

# ---------------------------------------------------------------- 3. Docker Desktop
say "Docker Desktop"
export PATH="$PATH:/usr/local/bin:$HOME/.docker/bin:/Applications/Docker.app/Contents/Resources/bin"
if [ ! -d /Applications/Docker.app ] && ! command -v docker >/dev/null 2>&1; then
  brew install --cask docker-desktop
fi
if ! docker info >/dev/null 2>&1; then
  open -a Docker || true
  echo "Waiting for Docker Desktop to start. If a window asks you to accept terms or sign in,"
  echo "accept the terms (signing in is optional)."
  for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 3; done
fi
docker info >/dev/null 2>&1 || die "Docker is not running yet. Open Docker Desktop, wait until it says 'Engine running', then run this script again."
MEM_GB=$(docker info --format '{{.MemTotal}}' | awk '{printf "%d", $1/1024/1024/1024}')
CPUS=$(docker info --format '{{.NCPU}}')
echo "Docker has ${CPUS} CPUs and ${MEM_GB} GB memory."
if [ "$MEM_GB" -lt 32 ]; then
  warn "Give Docker more memory for big jobs: Docker Desktop > Settings > Resources >"
  warn "Memory 48 GB, CPUs 10, Swap 4 GB > Apply & restart. (Your Mac has 64 GB.)"
fi

# ---------------------------------------------------------------- 4. OpenDroneMap
say "Downloading OpenDroneMap (about 2 GB, first time only)"
docker pull opendronemap/odm:latest

# ---------------------------------------------------------------- 5. 'dronemap' command on PATH
chmod +x "$HERE/dronemap"
BIN="$(brew --prefix)/bin"
ln -sf "$HERE/dronemap" "$BIN/dronemap"
echo "Installed the 'dronemap' command ($BIN/dronemap -> $HERE/dronemap)"

# ---------------------------------------------------------------- 6. OpenSplat (optional)
build_opensplat() {
  say "OpenSplat (Gaussian splats on the Apple GPU)"
  if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    if [ -d /Applications/Xcode.app ]; then
      echo "Pointing the developer tools at Xcode (asks for your password)..."
      sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
      sudo xcodebuild -license accept || true
      xcrun -sdk macosx metal --version >/dev/null 2>&1 || xcodebuild -downloadComponent MetalToolchain || true
    fi
  fi
  if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    warn "The Apple Metal compiler is missing, so OpenSplat would run on the CPU (~100x slower)."
    warn "Install Xcode from the App Store (free), open it once, then run:  setup/setup_mac.sh --opensplat"
    return 1
  fi
  brew install cmake ninja opencv libomp
  mkdir -p "$TOOLS"
  if [ ! -f "$TOOLS/libtorch/share/cmake/Torch/TorchConfig.cmake" ]; then
    echo "Downloading libtorch $TORCH_VER (about 200 MB)..."
    curl -fL -o "$TOOLS/libtorch.zip" "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${TORCH_VER}.zip"
    rm -rf "$TOOLS/libtorch"
    unzip -q "$TOOLS/libtorch.zip" -d "$TOOLS"
    rm -f "$TOOLS/libtorch.zip"
  fi
  xattr -dr com.apple.quarantine "$TOOLS/libtorch" 2>/dev/null || true
  if [ ! -d "$TOOLS/OpenSplat/.git" ]; then
    git clone https://github.com/pierotofy/OpenSplat "$TOOLS/OpenSplat"
  fi
  ( cd "$TOOLS/OpenSplat" && git fetch -q origin && git checkout -q "$OPENSPLAT_REF" )
  rm -rf "$TOOLS/OpenSplat/build"
  mkdir -p "$TOOLS/OpenSplat/build"
  ( cd "$TOOLS/OpenSplat/build" && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TOOLS/libtorch" -DGPU_RUNTIME=MPS && \
    ninja )
  if [ ! -f "$TOOLS/OpenSplat/build/default.metallib" ]; then
    warn "OpenSplat was built WITHOUT Apple GPU support (no default.metallib)."
  fi
  if ! "$TOOLS/OpenSplat/build/opensplat" --version >/dev/null 2>&1; then
    echo "Fixing the OpenMP library link..."
    brew link libomp --force || true
  fi
  "$TOOLS/OpenSplat/build/opensplat" --version || die "OpenSplat built but does not start. See the README troubleshooting section."
  echo "OpenSplat ready: $TOOLS/OpenSplat/build/opensplat"
}

if [ "$WANT_SPLAT" = 1 ]; then
  build_opensplat || true
else
  echo
  echo "Skipped OpenSplat (Gaussian splats). Add it later with:  setup/setup_mac.sh --opensplat"
fi

say "Checking everything"
"$HERE/dronemap" doctor || true
echo
echo "Done. Open a new Terminal window and type:  dronemap -h"
