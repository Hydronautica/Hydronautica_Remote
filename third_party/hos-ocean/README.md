# HOS-Ocean

[HOS-Ocean](https://gitlab.com/lheea/HOS-Ocean) is the high-order spectral wave solver from École Centrale de Nantes (LHEEA), licensed under GPLv3. We do not vendor it. `nls/hos_ocean.py` writes its input files and reads its output, and `build.sh` fetches and builds it locally.

## The patch

`file-bathymetry-units.patch` fixes reading a sea bed from a file (`bathymetry.type: file`) together with an analytical spectrum or stream-function waves. In those cases HOS-Ocean keeps the physical domain size `xlen`/`ylen` in peak wavelengths but makes the relaxation zones dimensional. The file-bathymetry block mixes the two, so only a strip `xlen` metres wide was interpolated from the file and the rest of the domain was left at the minimum or total depth.

The patch converts `xlen`/`ylen` to metres inside that block and restores them afterwards. The WAVEWATCH III case, whose inputs are already dimensional, is left unchanged.

Before the patch, a Gaussian shoal 15 m deep at (1976, 1025) m in the physical zone came out as a 26 m patch in a corner. With the patch it is 15.3 m deep at (2843, 1171) m, which is the expected point once the 878 m and 146 m relaxation zones are added.

## Building

```
third_party/hos-ocean/build.sh
```

`nls.hos_ocean.run()` finds the executable there, on `PATH`, or via `$HOS_OCEAN`.
