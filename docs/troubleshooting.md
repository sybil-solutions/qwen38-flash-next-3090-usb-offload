# Troubleshooting

Every entry below was hit during the campaign; the source is the plan entry (`results/campaign/EXL3_PLAN-log.md`, date in UTC) or a file in this repo.

## Link and device

**Engine fails at the first GPU call, or shows stale/shifted USB reads.** A process killed mid-transfer leaves state in the ASM2464 (bulk IN overflows).
The shim runs `pcie_cycle.py` + `usb_drain.py` before every start (`shim/tgcu_api.py: init`); `TGCU_NO_USB_RESET=1` turns that off. If the dock reports
LTSSM other than `0x78` (for example after swapping the card): `PYTHONPATH=~/tg-strata ~/tinygrad/.venv/bin/python ~/shim/pcie_cycle.py` (GOAL.md, Link).

**Two engines running.** They fight over the USB device. `srv.sh` / `ls_srv.sh` send `SIGTERM` to the server, wait until no `strata` process is left, then start. Do it
by hand the same way (`kill -TERM $(cat ~/shim/server.pid)`; wait for `pgrep -x strata` to be empty).

**GPU boot reads garbage after a killed session** (`PROM` window read while the previous GSP is still winding down). Fixed by the tinygrad patch (64 KB pieces, accepted
once two consecutive reads agree), `patches/tinygrad/0001-tg-strata-nv-usb-fixes.patch`.

**`async copy timeout`, `GPU timeout waiting for semaphore`, 10 s hangs under G=4 or at ~100K tokens.** Two separate root causes (plan 2026-10-12T05:25Z):
(1) `tgcu.Context._transient` waited for a semaphore value that already counted the push buffer being submitted when the ring wrapped: a self-deadlock, fixed
in `shim/tgcu.py` (`shim/tests/test_wrap.py`: the old code fails, the new one runs 20,000 flushes); (2) with `STRATA_PF_GROUP=4`, at the periodic checkpoint
(deterministic at 100K tokens) a fresh 256 MiB tgcu chunk memset hangs when VRAM headroom is too small: use `--vram-reserve-mib 400` (7,400 slots).
`TGCU_HANG_DUMP=1 TGCU_WAIT_S=30` dumps every thread and a native `sample` 8 s into a GPU wait.

## Memory

**`Insufficient Memory` command-buffer failure at layer 4 of the first prompt.** Metal's `recommendedMaxWorkingSetSize` on this mini is 11.84 GiB
(`results/raw/lru/ws.m` prints it). 5,600 LRU slots (10.46 GB) plus separate per-prompt prefill sets (3 x 400 slots = 2.24 GB) exceeded it and the engine had to be restarted. The prefill
sets are now borrowed from the coldest LRU slots (`STRATA_PF_BORROW_LRU`, default on). If you raise `STRATA_EXL3_METAL_SLOTS` past ~5,600 expect this error; more needs `iogpu.wired_limit_mb`
(a system setting, not changed here).

**`no VRAM is left for the expert cache` at start.** The engine sizes the expert cache after the session; verify/draft/prompt buffers need the ~600 MiB that must stay free.
Lower `--max-context`, use `--kv q4_0`, or drop the vision key (GOAL.md, Vision + 200K plan).

**A 2.5K-token prefill kills the engine (`no progress for 60 s (reading the prompt (batched): layer 0 ...)`, then `GPU timeout waiting for semaphore`)** on configs that let the prompt path borrow VRAM expert
slots. Use `--no-prefill-borrow` (every config here does; plan 2026-10-11T18:35Z).

**Decode slower after the first image.** The Metal encoder holds ~1.1 GB wired RAM while loaded and the file cache / LRU need it: `idle_unload_s: 120` in the `vision` key
(server patch 0002) ends it after 120 idle seconds. Resident encoder: decode 8.2/12.0/9.0 vs 9.0/12.3/9.4 on the config of that day.

**Disk full.** The mini's 228 GiB volume was at 96 %. The experts (45.8 GB), the two GGUF shards (83.6 GB), the dense pack (1.5 GB), the MTP files (1.7 GB), the mmproj (0.9 GB),
and `td/experts_index.json` (27 MB) are what you need (plus the engine log, which reached 33 MB); `pack_td.py sync` into a temporary pack needs the same space again.

## Build

**`ld: tapi error: malformed file ... MacOSX27.0.sdk/usr/lib/libSystem.B.tbd`.** The default SDK is 27.0; pass `-isysroot .../MacOSX26.5.sdk` (all build scripts do).

**The mini cannot build `libexl3metal` or `strata-vision-metal`** (no `metal` compiler, cannot link ObjC): build on a Mac with Xcode's Metal toolchain and copy the dylib / binary.

**`libexl3cpu.dylib` differs from the deployed one.** clang takes the install name from the `-o` argument: build with a relative output name (`exl3/cpu/build.sh` does).

**`docker run ... nvcc` finds no files.** The VM mounts `$HOME` only; run builds from a directory under `$HOME`, not `/tmp`. `exl3/build.sh` removes the old fatbin first, so a failed nvcc run (for example colima stopped) cannot leave a stale one that looks fresh.

**Strata CUDA kernels fail with `cublas_v2.h` errors.** `~/cuda-13.4/include/cublas_v2.h` must be the shim's minimal header (`shim/include/cublas_v2.h`).

**Docker build stops at apt: `Version '...' for '...' was not found`.** The Dockerfile pins the apt tools by version and Ubuntu removes superseded versions from its archive. Read the current ones
(`docker run --rm ubuntu:24.04@sha256:534baea6... sh -c 'apt-get update -qq; apt-cache policy g++ make cmake ninja-build python3 xz-utils file git ca-certificates curl'`), update the pins in the Dockerfile (`toolchain` and
`fetcher` stages), rebuild, and compare the fatbin hashes the build prints against the deployed ones (it fails on a mismatch).

**`docker compose run --rm build` is very slow or `exec format error` on a PC.** The image is `linux/arm64` (NVIDIA's `linux-sbsa` redist tarballs): native on Apple silicon, emulated elsewhere (QEMU/binfmt needed, several times slower).

**`./run.sh`: `host.ninja failed` with `no matching function for call to '__ldcs'`.** The engine's CUDA host compile was done with Apple clang 21 (CLT 26.6, the mini's); older clangs (17 here) lack the overloads. Point `CLANG_BIN` at an LLVM >= 21
(`CLANG_BIN=/opt/homebrew/opt/llvm@22/bin ./run.sh`); only the engine and `libstrata_exl3` use it. With clang <= 17, `run.sh` also writes an empty `curand_mtgp32_kernel.h` into `~/cuda-12.8/include` (the old CUDA wrapper force-includes it).

**`./run.sh`: `no Metal compiler`.** The Command Line Tools have no `metal`: run `./run.sh --build-metal` on a Mac that does (Xcode, or `xcodebuild -downloadComponent MetalToolchain`) and copy `./out/metal` to the mini's `./out/metal`.

**`./run.sh`: `... differs from the bundle's copy: kept`.** `run.sh` never overwrites an installed file that differs from what it would install (the mini's live `~/shim` is such a case); `--force` replaces, otherwise the difference is only reported.

## Config C (2026-10-15)

Entries from the Decode3-Decode16 logs ([results.md](results.md) section 0; plan entries in `results/campaign/EXL3_PLAN-log.md`).

**Decode 13 tok/s instead of 26-31, `mlock failed errno 35` (or the engine's anonymous LRU slots in swap).** The 6,400-slot mini-side LRU locks 12.0 GB; the default wired limit of the 16 GiB mini (12,451 MiB) is below that. `sudo sysctl iogpu.wired_limit_mb=14336 vm.global_user_wire_limit=15032385536 vm.user_wire_limit=15032385536`
(once per boot; it resets on reboot; `run.sh` checks it and prints the command). Without it set `STRATA_EXL3_METAL_SLOTS=5600`. Decode6: at 6,600 slots with the default limit the engine swapped 9.9 GB out and decoded 13/13/12.

**Image request fails with `Mac cold-expert lane failed at layer 10 (-3)` (Metal `Insufficient Memory`).** The LRU plus the 1.2 GB vision encoder exceeded the M4 GPU's working set. Config C gives 1,400 MiB of the LRU back while the encoder runs (`vision.lru_release_mib`, patch `0012`); do not remove that key at 6,400 slots.

**Seconds-to-minutes freezes, bimodal decode (7-8 vs 15-21 tok/s).** The mini idle-slept about a minute after the last terminal session (the engine holds no power assertion; Decode4 found a 411 s "freeze" that was a sleep): run `caffeinate -dimsu` (or `pmset sleep 0`); `run.sh` starts it.

**`190 MiB of VRAM free with everything loaded - LOW: requests may stall` at start.** Expected: `--vram-reserve-mib 310` is the floor of config C (290 failed to start with a draft-head OOM; the Decode15 ABBA, a 190K needle and 5 x 33K needles ran without a GPU timeout). If a GPU timeout does appear after a long prompt,
raise the reserve (the engine suggests `--vram-reserve-mib 632`; each +100 MiB costs ~57 expert slots).

**`MemoryError: Failed to allocate memory (OOM) Request size=0x7400000` tracebacks in `serve-engine.log`.** Handled: tgcu's `_new_chunk` asked for a fresh 116 MiB chunk and the free list served the request (no request failed; Decode7/8 logs).

**First short request after a 100K/190K prompt takes 6.5 / 13.7 s.** The elastic K/V trims back and the 538 / 1,241 lent expert slots are refilled from the profile over USB at ~290 MB/s; the next request is normal.

**GPU timeout when the prefill borrow is on (before `0009`).** The lockstep group constants lived in the borrowed slots and the refill overwrote them: fixed in `0009`; `TGCU_LAUNCH_RING=64` prints the last launches with every parameter resolved to chunk + offset to stderr on a timeout.

**Decode after the mini was rebooted is slow for ~10 minutes.** Spotlight re-indexing and the cold file cache; compare arms only interleaved (the campaign's ABBA rule).

**`--batch` is slower than one request at a time.** Measured (results.md 0.4): batch windows compute their misses on the CPU tier, not through the Metal lane. Config C runs one request at a time and queues the rest.

**`docker compose run --rm build` writes an old bundle after a patch change.** `run` reuses the image; run `docker compose build build` (or `make`) first.

## Server and benchmarks

**`bench/final_vision.py`: `No module named 'PIL'`.** Run it with `~/strata/.venv/bin/python` (pillow is in Strata's requirements). The `final_lru` log contains exactly this error.

**Needle request returns HTTP 400/503.** `lr_g4_100k.log` (HTTP 400) and `pf500_final.log` (HTTP 503 on the 32K needle) are failed attempts kept as they ran: the 400 run was on a config (`g4b`, `--vram-reserve-mib 100`) that was then replaced; the 100K number comes from `lr_g4c_100k.log`. The causes of the 400 and the 503 were
not investigated.

**Numbers jump between restarts.** Decode right after a restart depends on the OS file cache and LRU contents: run each probe twice (the scripts do), compare p2 first (it is the least
SSD-bound), and keep the encoder unloaded. The first prompt after a restart reads its misses from NVMe (p1: 7,598 and 8,023 MiB of page-ins in the two `final_lru` runs).

**`STRATA_MAC_WARM=1` crashes the engine** (`bulk IN short read: 139784/147968 bytes` in `hs_wait`). Root-caused and fixed (the look-ahead thread's first touch of lazily downloaded router
pages ran a USB device-to-host copy concurrently with the armed pull) but it is a measured no-gain option and stays off.
