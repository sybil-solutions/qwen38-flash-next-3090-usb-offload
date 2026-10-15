# Config C reproduction check, 2026-10-15 (card-free; the live server, the card and ~/shim ~/strata ~/strata-* trees were not written)

Evidence files in this directory: `config-c-sha256.txt` (hash manifest, checked by `scripts/verify-card-free.sh patches`), `config-c-run-check-mini.txt` (`./run.sh --check` on the mini, scratch root `/tmp/recipe`),
`config-c-run-check-driver.txt` (the same on the driver Mac with Homebrew LLVM 22, scratch root `/tmp/recipe`, before the `STRATA_ISA_FLOOR` fix below), `config-c-engine-compare-mini.txt` (`scripts/compare-engine.py` output), `config-c-fatbin-sass.txt` (cuobjdump SASS of the four fatbins that differ).

## What was run

1. Driver Mac (MacBook Pro M1 Max, macOS 26.5, Docker Desktop 29.1.3 linux/arm64): `docker compose build build && docker compose run --rm build` on the repo with patches `0001`-`0019` (Strata v0.1.41 clone, 19 patches + the vision patch, CMake configure, 63 nvcc rules, EXL3 fatbins), then `./run.sh --build-metal`.
2. Mini (read-only against the live trees; Apple clang 21.0.0, MacOSX26.5.sdk, uv CPython 3.12.15): the repo and `./out` copied to `/tmp/rcpC-repo`, then `RUN_ROOT=/tmp/recipe JOBS=2 nice -n 19 ./run.sh --check` (`/tmp/recipe` has the same length as `/Users/sero`: the rebuilt binaries are compared with the deployed ones after
   rewriting the root, byte for byte). CUDA 12.8 headers and SIMDe copied from the mini's `~/cuda-12.8`, `~/simde`; tinygrad cloned into the scratch root. `libcudart.dylib` also built standalone in `~/rcpC-scratch` with `shim/build-libcudart.sh`. Afterwards every scratch directory was removed; the live engine (md5 `fc6e4ddd`) and `libcudart.dylib` (md5 `ec0517ec`) were checked unchanged.
3. `scripts/verify-card-free.sh patches` and `scripts/test-gates.sh` on the driver Mac (see the end of this file).

## Result table (live = the deployed file on the mini, 2026-10-15)

| artifact | rebuilt | result |
|---|---|---|
| Strata source: 14 patched files + 3 server files + 5 shim/data files (`config-c-sha256.txt`) | `git apply` of `0001`-`0019` to a pristine `v0.1.41` export | sha256 equal to the manifest (20 files); the same files were also `cmp`-ed against the mini's `~/strata-decode15/src`, `~/strata-xport8`, `~/strata-lockstep`, `~/strata/serve` |
| `shim/tgcu.py`, `tgcu_api.py`, `tgcublas.cpp`, `run-d7.sh`, `exl3/placement/d8w4/profile.bin` | repo files | md5 equal to the mini's `~/shim/...` (`d2692131`, `9697e39c`, `c8d597b4`, `11d7d6f3`, `2ac3b7e1`); `patches/shim/base` + `0001`-`0005` reproduces `tgcu.py`, `tgcu_api.py`, `tgcudart.cpp` |
| `exl3/placement/d8w4/profile.bin` | `exl3/make_profile_wholelayers.py ... 0,1,2,6` from `f512c` | md5 `2ac3b7e1` equal |
| `libcudart.dylib` (from `shim/tgcudart.cpp` + `tgcublas.cpp`) | on the mini, `shim/build-libcudart.sh`, and inside `run.sh --check` | **byte-identical**: sha256 `20e8167d271ba797...`, md5 `ec0517ec`. On the driver Mac (Apple clang 17, SDK 26.2, CPython 3.12.13): same 136 exported symbols, different bytes |
| `libexl3cpu.dylib` | on the mini, `run.sh --check` | **byte-identical**, sha256 `9fe9cab5491e32e3...` (driver Mac: `d874f168...`, other toolchain) |
| `libexl3metal.dylib` | driver Mac, `./run.sh --build-metal` | **byte-identical**, sha256 `e38a011c79933a8e...` |
| `libstrata_exl3.dylib` | on the mini at `/tmp/recipe` | identical after rewriting the root except 48 bytes: the 16-byte `LC_UUID` and 32 bytes of the ad-hoc code signature (same size 4,450,936) |
| Strata engine `strata` | on the mini at `/tmp/recipe`, `run.sh --check` (135 host rules + link, 2 jobs, ~45 s) | **not byte-identical, but identical except for four nvcc fatbins**: see below |
| device code: EXL3 fatbins `exl3_strata/_recon/_pf`, `tgcu_builtin.fatbin`, `tgcu_libc.cubin` | Docker | byte-identical to the deployed files (the Docker build fails on a mismatch) |
| device code: the 63 engine fatbins | Docker (`out/engine/strata-mac-fatbins.sha256`) | 59 byte-identical to the mini's `~/strata-mac/**.fatbin`; 4 differ: `kernels.cu` (changed by patch `0008`, live copy = `~/strata-xport8/build/obj/kernels.cu.o.fatbin`), `fused_gr`, `sampler`, `moe_mmq`. nvcc puts a per-compilation id into the names of anonymous-namespace kernels, so those four differ from build to build (also between two Docker builds of this session: the first had `fused_gr` identical to the deployed one and `kernels` at 1,062,968 B, the second `fused_gr` different and `kernels` at 1,063,048 B). SASS, `config-c-fatbin-sass.txt`: `kernels` (incl. the new xport8 quant/dequant kernels), `fused_gr` and `moe_mmq` have **identical instruction streams** (names differ), `sampler` has the same 16,416 instructions of which 3,970 differ in register allocation/scheduling (as in the 2026-10-12 check) |

### The engine, in detail (`scripts/compare-engine.py live rebuilt /Users/sero /tmp/recipe`, `config-c-engine-compare-mini.txt`)

- file size 113,402,080 B (live) vs 113,402,072 B; 9,372 symbols on both sides, none only on one side, every symbol the same size except `_ggml_table_f32_ue4m3`, which sits after the fatbin section (it moved by the fatbin size difference);
- sections `__stubs`, `__init_offsets`, `__const`, `__gcc_except_tab`, `__cstring`, `__unwind_info`, `__eh_frame`, `__got`, `__const`, `__data`, `__thread_*`, `__bss`, `__common`: **byte-identical**;
- `__text`: same size (4,447,156 B); **70 of 1,111,789 instruction words differ, all `adrp` (14) or `add` immediate (56)**: the address of a global moved with the larger `__nv_fatbin`; no other instruction differs;
- `__nv_fatbin`: 56 embedded fatbins, 52 byte-identical; the other four are the nvcc-nondeterministic ones above (sizes: `kernels` 1,062,968 vs 1,063,048, `sampler` 818,352 vs 818,584, `fused_gr` 7,863,808 vs 7,863,968, `moe_mmq` 165,696 both). Together +472 B, which is why the file size and the code-signature blob differ.
- Conclusion: the host half of the deployed engine is reproduced instruction for instruction from the repo's patches with the mini's toolchain; byte identity of the whole file is out of reach only because nvcc does not produce the same four fatbins twice.
- For comparison, the driver Mac build (Homebrew LLVM 22, `config-c-run-check-driver.txt`, built before the fix below) has the same 2,210 exported symbols and the same set of string literals (6,839 vs 6,834 lines, the 5 missing lines are empty strings) but different code sizes (+0.7 % in `__text`: another compiler).

### Finding and fix: `STRATA_ISA_FLOOR`

The first scratch build on the mini differed from the deployed engine in one function, `strata::kernels::cpu::cpu_avxvnni_ok()` (56 vs 132 B, `__text` +256 B): the deployed `~/strata-linux/CMakeCache.txt` has `STRATA_ISA_FLOOR=none` (and `CMAKE_EXPORT_COMPILE_COMMANDS=ON`), which adds `-DSTRATA_ISA_FLOOR_NONE=1` to four host rules; `docker/build-engine.sh`
had not passed it (the repo had lagged). Fixed in `docker/build-engine.sh`; after the fix `host.ninja` of the bundle equals the deployed `~/strata-mac/host.ninja` rule for rule (diffed with the root rewritten) and the engine differs only as above.

### `./run.sh --check` on the mini (`config-c-run-check-mini.txt`)

59 ok, 8 warnings, 0 failures. The warnings: no Metal compiler on the mini (the Metal binaries came from `./out/metal`, built on the driver Mac), the models/packs are not under the scratch root (6 config paths), no `libamd_comgr.dylib` under the scratch root. The new host-prerequisite section on the mini itself:
`wired limits: iogpu.wired_limit_mb 14336, vm.global_user_wire_limit 15032385536, vm.user_wire_limit 15032385536` ok, `caffeinate -dimsu is running` ok, `free disk 31 GiB` ok, `STRATA_EXL3_METAL_SLOTS 6400: 11953 MB`. On the driver Mac the same section warns that the wired limit is the macOS default (`iogpu.wired_limit_mb 0`) on a machine whose `vm.global_user_wire_limit` is 27 GB (fixed afterwards:
0 counts as default when `vm.global_user_wire_limit` is large enough) and that `caffeinate` is not running.

### Card-free behaviour checks on the patched sources (driver Mac)

`src/prefill/path_select_test.cpp` (compiled with `clang++ -std=c++20 -Iinclude`): `all ok`; `src/prefill/xport8_test.cpp` (the int8 block-32 format, CPU path): `ALL OK`, round-trip rel RMS 0.44-0.83 % on the synthetic distributions it prints (one distribution is below fp16's range by construction and flushes to 0); `serve/test_prefix_replay.py`: 17 passed, 6 skipped (need the engine).
`scripts/verify-card-free.sh patches`: all 19 + 5 patches apply in order, 20 manifest files MATCH, shim chain MATCH, profile regenerated. `scripts/test-gates.sh`: ALL GATES OK.

### What could not be reproduced

- the exact bytes of four nvcc fatbins (`kernels`, `fused_gr`, `sampler`, `moe_mmq`) and therefore of the engine file (above);
- anything that needs the card: no decode or prefill number was re-measured, the engine was not started, and the repo's `libstrata_exl3.dylib` UUID/signature differs by construction;
- the weights: no pack or GGUF was fetched for this check (`pack_td.py` one-layer fetch was not re-run; results.md section 8 has the 2026-10-12 run);
- the Docker build ran on the driver Mac (M1 Max), not in the mini's colima; the engine's host half was compiled on the mini (scratch) and, separately, on the driver Mac (cross-toolchain, informational).
