# How it works

Qwen3.8-Flash-Next (`qwen4exp`): 48 layers, 36 gated-delta-net (GDN) layers + 12 full-attention (QSA) layers, hidden 2560, 512 routed experts per
layer with 10 used per token, expert FFN 640, one shared expert, hyper-connections, an MTP draft layer. The routed experts are 45.77 GB at 3.05 bpw
(24,576 experts x 1,862,400 B). The rest (attention, GDN, shared experts, norms, hyper-connections: the "dense part") comes from the IQ3_S GSQ-RCO GGUF
(83.6 GB on disk, its own experts unused): the engine reads the dense tensors from a 1.5 GB pack made by Strata's `iq_pack.py` and the per-layer-embedding
table from the second shard (`--ple-gguf`). A 24 GB card cannot hold the experts and the 16 GB mini cannot either, so they are split three ways.

```text
  RTX 3090 (24 GB), behind the USB dock                    Mac mini M4 (16 GiB)
  +-------------------------------------------+            +-------------------------------------------------+
  | dense part, attention + GDN, MTP head      |            | Metal lane (libexl3metal): LRU of 6,400 mlocked, |
  | elastic 200K k8v4 KV (0.23 GiB at 16K cells,|  x, ids, w | registered expert slots (11.95 GB) + 0.2 GiB    |
  |  2.30 GiB at 196K)                         | ---------> | static tier; computes every expert that is not   |
  | 9,958 expert slots (17.27 GiB), hottest    |   USB 3.2  | resident in VRAM                                 |
  | experts by profile rank, 4 whole layers;   | <--------- | NEON pool (libexl3cpu) = fallback + exact path   |
  | exl3_moe_coop kernels compute them         |     y     +----------------------^--------------------------+
  +-------------------------------------------+                                  | F_NOCACHE preads, 4 KiB-aligned, ~2 GB/s
                                                          +----------------------+--------------------------+
                                                          | NVMe: experts.bin, all 24,576 experts, one 1.86 MB |
                                                          | blob each (sha256 per layer in the manifest)       |
                                                          +-----------------------------------------------------+
```

(Config C, the recommended recipe; the diagram of the host-tail step had 7,400 VRAM slots, a 5,600-slot LRU and a fixed 2.6 GB K/V.) Of the 24,576 experts, 9,958 (41 %) sit in VRAM, 6,400 (26 %) in the Mac's LRU, the other 8,218 only on NVMe. Nothing is pruned or masked: a router
pick that is in neither VRAM nor the LRU is read from NVMe and computed on the Mac. The static ranking that fills VRAM and primes the LRU is
`exl3/placement/d8w4/profile.bin` = every expert of layers 0, 1, 2, 6 first (those four layers need no doorbell round trip at decode), then the routing-frequency order of `exl3/placement/f512c/profile.bin`
(from the earlier P256 calibration, mapped onto the 512-expert ids; `exl3/make_profile512_freq.py`, `exl3/make_profile_wholelayers.py`).

## The link

The dock is an ASM2464 bridge: USB 3.2 10 Gb/s outside, PCIe Gen3 x2 to the card (`LNKSTA 0x1023`). Measured on this rig: ~190 MB/s with the stock copy
path; 353 MB/s host-to-device and 328 MB/s device-to-host with the pipelined transfers in `shim/tgcu.py` (one doorbell per message, two SRAM
regions); the link is half duplex (a download and an upload cannot overlap: 104.0 MB/s combined vs 104.2 sequential, `results/raw/duplex/`).
A single USB transaction round costs ~0.4 ms, which is why the design uses as few rings as possible per layer.

There is no CUDA driver for macOS. `shim/` is one: tinygrad's NV backend boots the GPU over the USB bridge (GSP firmware, VA space, channels);
`tgcu.py` loads nvcc-built cubins, builds QMDs and chains launches into batches on tinygrad's compute ring; `tgcu_api.py` is the Python half of
`libcudart.dylib` (`tgcudart.cpp`, a drop-in for the CUDA runtime/driver API that clang `--cuda-host-only` objects link against) and a cuBLAS subset
(`tgcublas.cpp`, WMMA GEMM kernels in `tgcu_builtin.cu`). Strata's CUDA sources compile on macOS in two halves: `nvcc` (13.4, in a Linux container) makes
sm_86 fatbins, `clang --cuda-host-only` makes the host objects that embed them (`shim/macbuild.py`). CUDA graphs are captured launch lists.

## Decode: one ring per layer

Decode runs in verify windows (MTP drafts up to 4 tokens, the target verifies them in one pass: 1.8-3.3 accepted tokens per window on the three probe
prompts). Per window and per layer, the GPU computes attention/GDN and the router, then runs the resident experts; the experts that are not in VRAM
are the "cold" ones:

1. **Pre-armed pull** (`STRATA_HS_PRE`, `STRATA_HS_DEFER`): a copy-engine job armed before the ring and gated on the ring word fetches the doorbell payload
   (x rows, expert ids, weights) to the host the moment the GPU writes it. One pull and one gated push per ring; the next ring's pull is queued behind the push.
2. **Early start** (`STRATA_EXL3_METAL_EARLY`, `STRATA_MAC_PULL_EARLY`): once ids are on the host, the lane (a) submits the cold experts that are already in
   place (static tier + LRU hits) as one async Metal job and (b) reads the misses from NVMe straight into free LRU slots with `F_NOCACHE` preads
   (`STRATA_MAC_PULL_NOCACHE`), overlapping both with the queueing of the push.
3. **Metal job for the misses** (`STRATA_EXL3_METAL_SPLIT`): a second job; `y = A + B` in fp32.
4. **Push and flag**: the fp32 rows and the "landed" flag go back in one gated copy; the GPU waits on the flag and adds the rows to the residual.

The Metal library computes an expert exactly like the CUDA kernels do (trellis decode, Hadamard, GEMV, SwiGLU, Hadamard, GEMV with the per-tensor
scales; fp32 accumulation) and is gated against the NEON code and a float64 reference (`exl3/metal/test_exl3_metal.py`).

**The LRU.** Metal reads a blob in place only if it is registered (`newBufferWithBytesNoCopy`), and registering file-backed pages wires them, so the
cold set lives in `STRATA_EXL3_METAL_SLOTS` mlocked, registered slots: an O(1) doubly-linked recency list over `(layer, expert)` -> slot, preloaded at startup in
profile order (4.8 s for 5,600 slots). On the probe prompts 89.2 % of cold lookups are in place (67.5 % with the previous 1,000-slot LRU + 5.1 GiB static tier), 1.23
misses per call. The static tier shrank to 0.2 GiB because at equal RAM a pure LRU beat tier + LRU in a simulation on the traced calls (84 % vs 79 %). The
cap is Metal's `recommendedMaxWorkingSetSize` on this mini (11.84 GiB), not RAM.

**Slim doorbell** (`STRATA_EXL3_SLIM`, default on for EXL3 configs): the host pulls only the rows of x that have a miss. **Pool spin off**
(`STRATA_POOL_SPIN_US=0`, default in `shim/run-strata.sh`): spinning NEON workers starved the USB host thread (13-15 ms per layer of stall).

## Prefill: the Mac as the cold-expert lane

Streaming every cold expert over the 190-350 MB/s link took 138 s per 2,048-token chunk (12 tok/s). With `STRATA_PF_MAC=1` the cold experts never
cross the link: per layer, the chunk's fp16 MoE input comes down (lazy device-to-host), the Mac runs the cold (token, expert) pairs with the
grouped Metal prefill entry (`exl3m_moe_prefill*`) while the 3090 runs the resident experts, and the partial sums go back as fp16
(`STRATA_PF_MAC_Y16`) to be combined on the GPU. The blobs come from the LRU in place (hits) or are read with `F_NOCACHE` into slot sets borrowed from the
LRU's coldest slots for the duration of the prompt (`STRATA_PF_SETS=3`; no extra wired memory).

**Lockstep groups** (`STRATA_PF_GROUP=4`): G = 4 chunks of 2,048 tokens run layer-major, so each layer's cold blobs are read from NVMe once per group
instead of once per chunk. One thread per chunk, a fixed turn order forming a software pipeline: front half (attention/GDN, routing, resident
experts, x16 download, async Metal submit), then the back half of the chunk `lag` turns earlier (Metal wait, y16 push, combine). Each extra chunk
keeps its own residual stream and QSA step table on the device (an 84 MB stream per chunk, which is where the VRAM headroom for G=4 goes:
`--vram-reserve-mib 400`). At equal slot counts the lockstep output is bitwise equal to the plain lane.

## KV cache and context

`--kv k8v4 --max-context 204800`: int8 keys with exact scores plus rotated 4-bit values, 816 B per cell per KV layer (12 QSA layers + the MTP draft
layer), 2.61 GB at 200K, 1,308 of the 9,108 expert slots that would otherwise fit. fp16 would cost 2,969 slots, int8 1,584, q4_0 914; `--kv-resident`
(host-resident KV) costs 2.75 GB of mini RAM and was measured negative. The 36 GDN layers hold constant-size state.

## Vision

`serve/server.py` runs `strata-vision` (llama.cpp mtmd + the BF16 mmproj) as a child process: it encodes the image on the **Mac's GPU** (Metal), writes an
embeddings file, the server expands `<|image_pad|>` and the engine's `GENI` path prefills the picture rows with M-RoPE positions through the batched
prefill path. The encoder holds ~1.1 GB wired RAM while loaded, so `idle_unload_s: 120` (patch 0002) ends it after 120 idle seconds.

## Settings in the previous final config (`final-g4-lru`)

`configs/strata-exl3-full512-final-g4-lru.json.tmpl` (the host-tail step; config C adds to it, see the next section):

| setting | value | what it does |
|---|---|---|
| `--pack` | `~/exl3/pack_full512` | dense pack (symlinks to the IQ3_S pack) + `experts.bin` + `exl3_experts.txt` marker |
| `--expert-profile` | `exl3/placement/f512c/profile.bin` | hottest-first ranking of all 24,576 (layer, expert) pairs |
| `--expert-cache auto` + `--vram-reserve-mib 400` | | 7,400 slots; 400 MiB headroom (needed for G=4: the 256 MiB tgcu chunk at the periodic checkpoint) |
| `--resident-budget-gib 0.2` | | static locked tier 0.2 GiB (115 blobs) |
| `--kv k8v4 --max-context 204800` | | 200K context |
| `--spec 4 --spec-min-p 0.5 --mtp ~/models/qwen38fn/mtp/rt` | | MTP drafting |
| `--prefill 2048` `--no-prefill-borrow` | | chunk size; the prompt path allocates its own buffers (borrowing VRAM slots for prompts crashed 2.5K-token prefills on this transport) |
| `--mmap-experts --pcie-frac 0 --adapt-every 0` | | `experts.bin` is mmapped; the other two switch off Strata's PCIe-streaming fraction and adaptive re-placement [INFERENCE from the flag names] |
| `STRATA_EXL3_METAL=1`, `_SLOTS=5600` | | Metal lane, LRU size |
| `STRATA_EXL3_METAL_SPLIT=1`, `_EARLY=1`, `STRATA_MAC_PULL_EARLY=1`, `STRATA_MAC_PULL_NOCACHE=1` | | decode lane as above |
| `STRATA_PF_MAC=1`, `_Y16=1`, `_NOCACHE=1`, `STRATA_PF_GROUP=4`, `STRATA_EXL3_METAL_JOBS=4`, `STRATA_PF_SETS=3` | | prefill lane as above |
| `STRATA_RESIDENT_HEADROOM_GIB=1.5` | | carried over from the first full-512 configs; with `--resident-budget-gib 0.2` it no longer sizes anything [INFERENCE] |
| defaults from `shim/run-strata.sh` | `STRATA_POOL_SPIN_US=0`, `STRATA_EXL3_SLIM=1`, `STRATA_HS_PRE=1`, `STRATA_HS_DEFER=1`, `STRATA_MAC_PULL=1` | |
| `vision` | Metal `strata-vision-metal`, mmproj BF16, `max_tokens 1024`, `idle_unload_s 120` | |

Older configs kept for rollback: `final-g4` (G=4, 1,000-slot LRU + 5.1 GiB tier) and `final` (G=3, no `--vram-reserve-mib 400`). Going back to `final-g4`
needs the engine binary from before the LRU change (`strata.bak-1012-prelru` on the mini); the new binary on the old config was not run.

## Config C: what changed since the host-tail step

`configs/strata-exl3-full512-c.json.tmpl`. Everything below is in this repo (patch numbers: [patches/README.md](../patches/README.md)); the numbers are in [results.md](results.md) section 0.

**VRAM is the currency.** Decode time per window is `25.8 ms + rings + in-place Metal work + 0.64 ms per LRU miss` (results.md 0.5), and every extra expert slot in VRAM removes ring work and misses (~7 ms per +1,000 slots), so most of the work was taking VRAM back:

- **Elastic K/V** (`--kv-grow` machinery of Strata, `STRATA_KV_GROW=1`, `STRATA_KV_GROW_RESIDENT=1`, patch `0018` + the cuMem* emulation of `shim/0004`): the K/V pools and the expert cache are virtual ranges; the pools map physical memory only for the cells the context has reached (16,384 at start, step 8,192), the expert cache keeps the rest, and when a context grows
  the cache gives up slots below the prompt path's loan (hotter experts move into colder slots first; the given-up ones become ordinary mini-lane misses) and gets them back, refilled from the profile, when the next short request trims the K/V. 8,393 -> 9,467 slots at the same 200K maximum; 100K needs 538 slots lent, 190K 1,241 (first short request after a 190K one reads 13.7 s instead of 1.7 s while the slots refill).
- **Decode15 census and knobs** (`0019`, `shim/0005`): the VRAM census of an idle engine at 16K cells (physical pool 24,256 MiB; expert cache + elastic K/V 17,028; dense.bin 1,467; native projections 2,019; LM head 497; MTP 675 + 111 + 213; session 474; ...; `results/raw/decode15/census.out`) priced every permanent item. The indexer's pooled keys became elastic
  (session 474 -> ~198 MiB), the tgcu arenas shrank (`TGCU_CODE_MIB=32`, `TGCU_MAPPED_MIB=16`), the RAM-tier VRAM mirror (204 MiB that `--pcie-frac 0` never reads) is not allocated (`TGCU_NO_MIRROR_HOSTALLOC_MIB=150:260`: locked pageable memory), the lazy-D2H shadow is capped (`TGCU_LAZY_D2H_MB=128`), shader local memory is
  480 B/thread (`TGCU_LMEM_STACK`), and cudaMemGetInfo subtracts 742 MiB instead of 1,024 (`TGCU_MEMINFO_FIXED_MIB`, with `TGCU_TOP_RESERVE_MIB=170` keeping chunks out of the pool's dead top). With `--vram-reserve-mib 310`: 9,958 slots, **190 MiB of VRAM free after load** (the engine warns LOW; a reserve of 290 failed to start with a draft-head OOM).
- **Prefill borrow back on** (no `--no-prefill-borrow`, `0009`): the prompt path lends 884 expert slots (1.53 GiB: the group buffers of G=6, pair jobs and lane buffers) and the refill after the prompt is guided by the prompt's own routing histogram (`STRATA_D7_MODE=7`, `0011`) and staged (one multi-blob H2D per 8-slot piece, 346 MB/s = the link rate); the startup fill uses the same path (`STRATA_D8_FILL=1`, 63.7 s for 9,958 slots).

**Decode ring.** Four whole-resident layers (profile `d8w4`: 2,048 slots) need no doorbell, the other 44 ring as before (pre-armed pull, gated push). `TGCU_ARM_EARLY=1` issues the OUT arm of the push in `h2d_pre` (off the critical path), `TGCU_PULL_PACK=1` pulls only the ring's three segments instead of the arena span between them
(`shim/0003`). The mini-side LRU is 6,400 slots (11.95 GB mlocked + registered; needs `iogpu.wired_limit_mb=14336` and the two `vm.*wire_limit` values, while the vision encoder runs it takes 1,400 MiB of the LRU back (`vision.lru_release_mib`, `0012`; `LRUMIB` command, `0011`): at 6,000 slots with the encoder resident an image request failed with Metal `Insufficient Memory` (Decode6)).
LRU-slot reads are 4 KiB aligned (`STRATA_D11_ALIGN=1`, `0016`): a blob (1,862,400 B at a 128 B-aligned offset) is read from the enclosing aligned range into the start of its 1,867,776 B slot and addressed `d11_skew` (<= 3,968 B) bytes in; unaligned `F_NOCACHE` reads were measurably slower on this SSD (1 blob 0.9-1.4 vs 0.68 ms).

**Prefill lane.** `STRATA_PF_XPORT8=1`: the lane's x rows (card -> Mac) and y rows (Mac -> card) cross the link as int8 + one fp16 scale per 32 values (53 % of the fp16 bytes; quantized on the card by a CUDA kernel, on the Mac with NEON; format in `include/strata/prefill/xport8.hpp`); `STRATA_PF_PAIR=1` runs two chunks per Metal job; `STRATA_PF_GROUP=6`
(six 2,048-token chunks in lockstep, `STRATA_EXL3_METAL_JOBS=6`, `STRATA_PF_MIN_CHUNKS=6`) gives the pair job the depth it needs; the lane takes every batched chunk (`0006`); prompt-set reads are aligned too (`STRATA_D12_PFALIGN=1`, `0017`). In the G=6 stats a Metal pair job takes ~115 ms = ~80 ms per chunk-layer, against the 87-91 ms chunk-layer period of the host-tail steady state; prefill speed rises with prompt length (176 -> 584 tok/s from 2.5K to 33K), short prompts remain SSD-bound (233 ms per layer pass at 2.5K).

**Server.** `prefix_replay` (`0007`): the model's own reply tokens are put back into the next turn's prompt (and a new chat sharing >= `prefix_pin_min` = 2,048 tokens with a recent one gets a checkpoint pin), so agent turns read only what is new (tool-call reuse: read 40 of 3,379).

**Settings of config C that differ from the table above**:

| setting | value |
|---|---|
| `--expert-profile` | `exl3/placement/d8w4/profile.bin` |
| `--vram-reserve-mib` | 310 (was 400); `--no-prefill-borrow` removed |
| `exe` / `D7_BIN` | `~/shim/run-d7.sh` (= `run-strata.sh` with the engine path from `D7_BIN`) / `~/strata-mac/strata` |
| prefill lane | `STRATA_PF_GROUP=6`, `STRATA_EXL3_METAL_JOBS=6`, `STRATA_PF_MIN_CHUNKS=6`, `STRATA_PF_XPORT8=1`, `STRATA_PF_PAIR=1` (carried over from the host-tail config: `STRATA_PF_MAC*`, `STRATA_PF_SETS=3`, `STRATA_PF_SPLITF=1`, `STRATA_PF_SPLITF_MIN=4`) |
| mini lane | `STRATA_EXL3_METAL_SLOTS=6400`, `STRATA_D7_MODE=7`, `STRATA_D8_FILL=1`, `STRATA_D11_ALIGN=1`, `STRATA_D12_PFALIGN=1` |
| K/V | `STRATA_KV_GROW=1`, `STRATA_KV_GROW_RESIDENT=1` |
| tgcu | `TGCU_TOP_RESERVE_MIB=170`, `TGCU_BESTFIT=1`, `TGCU_HANG_DUMP=2`, `TGCU_ARM_EARLY=1`, `TGCU_PULL_PACK=1`, `TGCU_CODE_MIB=32`, `TGCU_MAPPED_MIB=16`, `TGCU_NO_MIRROR_HOSTALLOC_MIB=150:260`, `TGCU_LAZY_D2H_MB=128`, `TGCU_LMEM_STACK=480`, `TGCU_MEMINFO_FIXED_MIB=742` (carried over: `TGCU_POKE1=15`, `TGCU_FREE_DEFER=1`, `TGCU_H2D_RELAX=1`, `TGCU_TS_G=150`, `EXL3M_WAIT_SPIN=1`) |
| `vision` | `lru_release_mib: 1400` added |
| server | `prefix_replay: true`, `prefix_pin_min: 2048` (the API key and bind address are not in the template: `run.sh` passes `STRATA_API_KEY` and `--host`) |

Fallbacks: the templates of the earlier steps stay in `configs/`; they need the engine and shim of their own step (patches up to `0003` / `0004` / `0005`, see [patches/README.md](../patches/README.md)).

## What is not here

A KL/top-1 comparison against stock exllamav3; a decode path that hides the 0.85 ms two-transaction floor (GPU-written landed flags are not
possible because layer l's router input only exists after the GPU finishes attention l); GPU-side expert prefetch from a layer-ahead router (measured
negative: 42-49 % coverage); an LRU beyond 6,400 slots (config C already needs the raised `iogpu.wired_limit_mb` / `vm.*wire_limit`; the Decode13 sweep measured 6,700-7,000 slots at +2.4..+4.4 % with growing swap-ins on this 16 GB mini, 6,600 also passed its regression); a KL against stock exllamav3.
