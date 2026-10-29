#!/usr/bin/env bash
# Build HOS-Ocean (GPLv3, Ecole Centrale de Nantes / LHEEA) at a known commit with our bathymetry-file fix.
# Needs: git, cmake, a Fortran compiler, FFTW3 and LAPACK (Debian/Ubuntu: gfortran cmake libfftw3-dev liblapack-dev).
# The executable ends up in third_party/hos-ocean/src/build/sources/HOS-Ocean, where nls.hos_ocean looks for it.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
commit=a76d642
if [ ! -d "$here/src/.git" ]; then
  git clone https://gitlab.com/lheea/HOS-ocean.git "$here/src"
fi
cd "$here/src"
git fetch -q origin master || true
git checkout -q "$commit"
git checkout -q -- sources/bathymetry/bottom_def.f90
git apply "$here/file-bathymetry-units.patch"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc 2>/dev/null || echo 2)"
echo "Built: $here/src/build/sources/HOS-Ocean"
