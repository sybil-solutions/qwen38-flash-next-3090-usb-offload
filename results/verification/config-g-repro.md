# Config G reproduction check, 2026-10-18 (card-free; the live server, the card and every `~/shim` `~/strata*` `~/exl3` tree on the mini were only read)

Evidence in this directory: `config-g-sha256.txt` (hash manifest, checked by `scripts/verify-card-free.sh patches`), `config-g-verify-patches-mini.txt` (that gate on the mini), `config-g-run-check-mini.txt` (`./run.sh --check` on the mini, scratch root `/tmp/recipe`), `config-g-engine-compare-mini.txt` (`scripts/compare-engine.py` output).

## What was run

1. Driver Mac (MacBook Pro M1 Max, macOS 26.5, Docker Desktop 29.1.3 linux/arm64): `make build` (`docker buildx build --target bundle`) on the repo with patches `0001`-`0024` + the vision patch: Strata v0.1.41 clone, 24 patches, CMake configure, 63 nvcc rules, the four EXL3 fatbins and the shim blobs. `./run.sh --build-metal` for the Metal binaries.
2. Mini (Apple clang 21.0.0, MacOSX26.5.sdk, uv CPython 3.12.15; read-only against the live trees): the repo and `./out` copied to `/tmp/rcpG-repo`, `~/cuda-12.8` and `~/simde` copied to `/tmp/recipe`, then `RUN_ROOT=/tmp/recipe JOBS=2 nice -n 19 ./run.sh --check` (`/tmp/recipe` has the length of `/Users/sero`, so the rebuilt binaries are compared with the deployed ones after rewriting the root). Result: 63 ok, 8 warnings (no Metal compiler on the mini: the Metal binaries come from `./out/metal`; models/packs absent under the scratch root), 0 failures. Engine, USB dock, the card and the server were not touched (live config md5 `ff78909a`, engine pid unchanged afterwards); all scratch directories were removed.
3. Mini: `STRATA_DIR=~/strata WORK=/tmp/rcpG-work scripts/verify-card-free.sh patches`: 24 + 6 + 1 patches apply in order, **25 manifest files MATCH** (`config-g-sha256.txt`), shim chain MATCH, `d8w4/profile.bin` regenerated (md5 `2ac3b7e1` MATCH). `scripts/test-gates.sh` (`make test`) on the driver Mac: ALL GATES OK (7 checks).

## Result table (live = the deployed file on the mini, 2026-10-18)

| artifact | rebuilt | result |
|---|---|---|
| Strata source: 14 patched files + 3 server files | `git apply` of `0001`-`0024` to a pristine `v0.1.41` export | sha256 equal to the manifest; `src/core/verify.cpp` = `15a8d268...` = `~/strata-opt23/src/core/verify.cpp` (the only file patch `0024` touches) |
| `exl3/exl3_o23.cu`, `exl3/strata_m4/exl3_o23_host.cu`, `exl3/metal/exl3_metal.{metal,mm}` | repo files | sha256 in the manifest; md5 equal to the mini's `~/strata-opt23/lib` copies (`2026578a`, `77ac59fb`) and the driver's `runs/opt23/metal` copies |
| `exl3_o23.fatbin` (nvcc 13.4.92, sm_86) | Docker (build-device.sh) | **byte-identical**: sha256 `a6fad4f2...`, md5 `c291e020`; the container build fails on a mismatch. Finding: it comes out identical only with the plain flags of the deployed build (`-std=c++17 -O3 --fatbin -arch=sm_86 -no-compress`); with the flags of the other three modules (`-Xptxas -v -I torchstub --expt-relaxed-constexpr` + the cuBLAS `-D`s) the same source gives `937aef03...`. `exl3/build.sh` and `docker/build-device.sh` now use the plain flags for `exl3_o23`. Not path dependent (`/Users/sero/exl3` and `/Users/sero/strata-opt23/lib` give the same bytes) |
| `libexl3metal.dylib` (fused-decode prefill GEMM) | driver Mac, `exl3/metal/build.sh` and `./run.sh --build-metal` | **byte-identical** to the deployed `libexl3metal_g.dylib`: sha256 `53d4deab43590de4...`, md5 `c49abac8` |
| `libstrata_o23.dylib` | on the mini at `/tmp/recipe` (`run.sh --check`, `build_o23.sh`) | identical after rewriting the root except **47 bytes** (the 16-byte `LC_UUID` and 31 bytes of the ad-hoc code signature; same size 50,768 B). Rebuilt md5 `e5ba9cf8`, live `6a4e09a7` |
| `libstrata_exl3.dylib` | same | same pattern as config C: 48 bytes (UUID + signature), same size 4,450,936 B |
| `libcudart.dylib`, `libtgusb.dylib` | same | unchanged from config F: signature removed, md5 `9fdda6f2` and `35be5355` = the recorded config-F values (the live `~/strata-prefill22/shim` files are md5 `00771147`, `7e3e04fd`) |
| `libexl3cpu.dylib` | same | byte-identical, sha256 `9fe9cab5491e32e3...` |
| Strata engine `strata_o23b` | on the mini at `/tmp/recipe`, host half (135 rules, 2 jobs, `nice -n 19`) | **not byte-identical, identical except for four nvcc fatbins** (below). Live md5 `05ec22e5`, rebuilt md5 `5ccea896`, sizes 113,438,976 vs 113,438,968 B |

### The engine, in detail (`config-g-engine-compare-mini.txt`)

- 9,396 symbols on both sides, none only on one side, every symbol the same size except `_ggml_table_f32_ue4m3`, which sits after the fatbin section;
- sections `__stubs`, `__init_offsets`, `__const`, `__gcc_except_tab`, `__cstring`, `__unwind_info`, `__eh_frame`, `__got`, `__const`, `__data`, `__thread_*`, `__bss`, `__common`: byte-identical;
- `__text`: same size (4,469,140 B); 70 of 1,117,285 instruction words differ, all `adrp` (14) or `add` immediate (56): a global's address moved with the larger `__nv_fatbin`;
- `__nv_fatbin`: 56 embedded fatbins, 52 byte-identical; the four that differ (`kernels`, `sampler`, `fused_gr`, `moe_mmq`) are the nvcc-nondeterministic ones already found for config C (`config-c-repro.md`): nvcc puts a per-compilation id into the names of anonymous-namespace kernels. Config G adds no CUDA code to the engine itself (its device code is the separate `exl3_o23` fatbin, which is identical).

## What could not be reproduced

- the exact bytes of the four nvcc fatbins and therefore of the engine file;
- the `LC_UUID` and ad-hoc signature of the two EXL3 host libraries;
- anything that needs the card: no decode or prefill number was re-measured here (the numbers are the Opt23 campaign's, `results/raw/opt23/`), the engine was not started, no model file was fetched;
- the `fatbins`, `cpu`, `metal` and `cudart` steps of `scripts/verify-card-free.sh` were not re-run as such (the Docker bundle and `run.sh --check` above cover the same builds; the `metal` step's parity test needs the 45.8 GB pack and was run by the Opt23 campaign against the live library: 39/39 `cmplibs` cases and 7/7 zipf cases bit-identical).
