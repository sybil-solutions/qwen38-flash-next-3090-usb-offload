PREREQUISITES: 

- Tinygrad's Chestnut eGPU: https://comma.ai/shop/chestnut - 250$ + tax
- M4 Mac Mini 16GB RAM / 256GB SSD:  900$ + tax
- RTX 3090 24 GB VRAM: 1100$ + tax
- Model: Qwen 3.8 Flash in exl3-3bpw by turboderp: https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5

Total Cost: 2250$ + Tax and Shipping
Peak power consumption: 450w/h

Model scores 40 on AA, on Max reasoning it's tying GPT-5.6-Sol low & Opus-4.8 low 
This is still rather unoptimised, I for a 3090 I would recommend running Qwen3.8-27B you'll get 5-6x better performance. 

<img width="958" height="590" alt="image" src="https://github.com/user-attachments/assets/14569d15-e801-4a1f-a3b5-1fa3696b8caf" />


# qwen38-flash-next-3090-usb-offload

![Qwen3.8-Flash-Next EXL3 3bpw on one RTX 3090 over USB plus a Mac mini M4](docs/exl3-card.png)

Run **Qwen3.8-Flash-Next** (EXL3 3.05 bpw, 48 MoE layers x 512 experts, 45.8 GB of experts, all of them computed, none pruned) with a **200K context**,
**image input** and **MTP speculative decoding** on one **RTX 3090 24 GB attached to a Mac mini M4 (16 GiB) through a USB dock**. The GPU holds the
hottest experts and the attention; the mini computes the other experts on its own GPU (Metal) from an LRU of registered RAM slots and an NVMe drive.
The engine is [Strata](https://github.com/Niko1221/Strata) v0.1.41 plus the patches here, running on macOS through a CUDA-runtime shim ("tgcu") on
tinygrad's NV backend.

## One-click

```sh
docker compose run --rm build      # any machine with Docker (linux/arm64): the pinned CUDA 13.4 build -> ./out (fatbins, patched Strata, configure tree); 3 min cold on an M-series Mac, seconds when cached
docker compose --profile fetch run --rm fetch all    # optional, ~130 GB: experts (sha256-checked against results/pack/exl3_pack.json) + dense GGUF and mmproj (sha256-checked)
./run.sh                           # on the Mac mini: installs the bundle, builds the macOS libraries and the engine, renders config C, checks the host prerequisites, starts the server
./run.sh --check                   # the same, without starting the engine or touching USB
./run.sh --config final-g4-lru-ht  # an older config (fallbacks, see below)
```

(`docker compose build build` first when the repo changed: `run` reuses the image.) `./run.sh` defaults to **config C** (`configs/strata-exl3-full512-c.json.tmpl`, below), the recommended recipe. Host prerequisites it checks and prints
(nothing is stored, no password is asked for):

```sh
sudo sysctl iogpu.wired_limit_mb=14336 vm.global_user_wire_limit=15032385536 vm.user_wire_limit=15032385536   # once per boot (it resets on reboot): the 6,400-slot mini-side expert LRU locks 12.0 GB
caffeinate -dimsu &                # no idle sleep (run.sh starts it when none runs; `./run.sh stop` ends it): an idle-sleeping mini froze the engine for minutes during the campaign
# STRATA_EXL3_METAL_SLOTS 6400 is in the config; free disk: a few GiB for swap and logs, ~190 GB on a fresh install
```

The serving runtime is native macOS (the engine drives the 3090 over USB through libusb and computes the cold experts with Metal), and Docker on a Mac runs a Linux VM with no Metal and no
USB passthrough of this dock, so Docker makes the build and the model download one command but cannot run the server; `./run.sh` is the one native step. Two Metal binaries need Xcode's Metal toolchain,
which the mini's Command Line Tools lack: `./run.sh --build-metal` builds them on any Mac that has it (into `./out/metal`; copy that directory to the mini). What was run and what was not: [docs/reproduce.md](docs/reproduce.md#10-one-click-what-was-run) and [section 11](docs/reproduce.md#11-config-c-what-was-reproduced).

## Results: config C

Measured 2026-10-14/15 on the mini (Decode16), one request at a time, greedy, MTP spec 4, `--max-context 204800`, vision encoder configured, all 512 experts computed. Config `configs/strata-exl3-full512-c.json.tmpl`
(live config md5 `bc8450a1`, engine `strata_d15a` md5 `fc6e4ddd`; every setting in [docs/how-it-works.md](docs/how-it-works.md#config-c-what-changed-since-the-host-tail-step)).

| prefill length | prefill speed | decode speed | kv cache | concurrency |
|---|---|---|---|---|
| 29-35 tokens (3 probe prompts, 12 requests) | TTFT 0.9-1.7 s | **26.8 tok/s** mean (27.0 / 29.6 / 23.3 and 27.0 / 31.1 / 23.6 in the final regression) | elastic k8v4, 204,800 cells max; 16,384 (0.23 GiB of VRAM) in use at start | 1 |
| 3.7-19.8K tokens (29 document prompts, 300 tokens out) | TTFT mean 23.75 s | **31.0 tok/s** (ABBA, +2.6 % over the Decode14 config; 30.8 in the final regression) | elastic, grows with the context | 1 |
| 2,565 | 176.1 and 176.7 tok/s (SSD-bound) | not measured at this length | elastic k8v4 | 1 |
| 8,018 | 383.0 and 417.8 tok/s | not measured at this length | elastic k8v4 | 1 |
| 18,330 | 445.9 and 470.1 tok/s | not measured at this length | elastic k8v4 | 1 |
| 33,842 | 555.4 and 584.2 tok/s | not measured at this length (5 successive 33K needles found) | elastic k8v4 | 1 |
| 97,811 (100K needle) | 560.9 tok/s (TTFT 175 s) | 27.0 tok/s, needle found | 98,304 cells = 1.16 GiB | 1 |
| 189,481 (190K needle) | 548.1 tok/s (TTFT 346 s) | 24.2 tok/s, needle found | 196,608 cells = 2.30 GiB | 1 |
| 6-9K prompts, `--max-context 40960` with the batch engine | 100-120 tok/s (prompt read in parts) | **1.4-1.8 tok/s per request at 2 in flight, 0.6-1.7 at 4** (c=1 control 31.9) | 40,960 cells (elastic K/V is off with `--batch`) | 2 and 4: measured, slower than one at a time |

Where two prefill numbers are given they are two runs of the same prompt shape with different random first lines (no prefix cache). Decode "26.8" and "31.0" are means over the 12 probe requests and the 29 realistic requests of the two config-C
arms of the ABBA run (probes 26.58 / 27.09, realistic 31.28 / 30.79, control 26.02 / 30.25); the 12 probe requests are the three short prompts of the earlier tables (a code prompt, an explanation, a story), the 29 realistic prompts are
document questions of 3.7-19.8K tokens. Raw logs: [`results/raw/decode16/logs/`](results/raw/decode16/logs) (`ab15.out`, `newA.out`, `newB.out`, `ctlA.out`, `ctlB.out`; `d15_final.log` = the final regression with the prefill rows, tool reuse, image, the 190K needle and
5 x 32K needles; `long_C.log` = 100K / 190K; `d15_q.log` = the quality gate; `abconc_*` = the batch measurement), tools as run in `results/raw/decode16/tools/`, the VRAM census behind the slot count in `results/raw/decode15/`.
Every table with dates and source files: [docs/results.md](docs/results.md). The previous recipe, `final-g4-lru-ht` (2026-10-12, decode 18-21 tok/s, prefill <= 481), stays in `configs/` as a fallback and its rows in results.md section 0a.

**Quality gate.** Teacher-forced on 8 prompts (705 positions, coverage 100 %) against the saved greedy replies of an earlier stack of this campaign ("p-off": the stack with the int8 prefill transport switched off): top-1 agreement **98.01 %**, mean KL **0.0041**, **p99 KL 0.058** (gate < 0.1), mean |dlogp| 0.0695: PASS
(gates: top-1 >= 97 %, mean KL < 0.01, p99 KL < 0.1, coverage >= 99 %; KL is a top-64-support estimate; [`d15_q.log`](results/raw/decode16/logs/d15_q.log)). Two caveats: the reference is this engine's own earlier output, **not stock exllamav3** (no KL against exllamav3 was measured), and comparing config C with its immediate predecessors
the p99 is 0.104 (Decode13 stack) and 0.111 (Decode12 stack), above 0.1: those comparisons are between stacks that move experts between the in-place and miss jobs (fp32 association), not the gate, but they show that the output of every stack since Decode7 differs from the others at the rounding level (free-running replies identical on 5 of 8 prompts).

**Concurrency.** One request at a time. The server queues the rest (FIFO): with the normal config 1, 2 and 4 clients give 9.71 / 10.16 / 10.21 tok/s aggregate and 21.5 / 46.2 / 86.7 s mean TTFT. The batch engine (`--batch`, measured at `--max-context 40960` only) is **slower**: a batch window of 2-3.5 rows takes 540-1,650 ms because the misses are computed on the generic CPU tier,
not through the Metal lane that gives ~100 ms windows at c=1; aggregate 1.7-2.7 tok/s at c=2/4 against 10.2 for the control, c=1 on a batch config -11 % (28.3 vs 31.9 per-request), and elastic K/V is off with `--batch`, so each slot session carves the whole 40,960 cells (expert slots 9,983 -> 8,387-9,192). No fault, no GPU timeout.
Engine work to route the batch misses through the Metal lane and to make the K/V elastic per slot is in progress and is **not part of this repo**.

**What is exact.** The weights: every routed expert of every layer is the turboderp checkpoint's trellis and scale tensors, byte for byte (each layer was range-fetched from the pinned
revision, field-checked, hashed into a sha256 manifest, and 6 experts per layer were re-fetched and byte-compared; nothing pruned, masked or re-quantized), and the 3090 / NEON kernels are
bit-exact (decode) or within 1.1e-3 rel RMS (GEMM; GEMV 5.6e-4) of the exllamav3 reference. **What is not bit-exact in config C:** the Mac prefill lane ships the MoE input and the partial sums over the USB link as int8 + one fp16 scale per 32 values (`STRATA_PF_XPORT8=1`, patch `0008`:
rel RMS 0.55-0.65 % round trip on real MoE inputs, 0.80-0.83 % on the cold partial sum), the fp16 partial sums of the lane (`STRATA_PF_MAC_Y16=1`), and the fp32 split between in-place and missed experts (`STRATA_EXL3_METAL_SPLIT=1`), whose partition depends on the LRU contents and the slot count, so greedy text can
differ from a single-pass run after a few hundred characters (coherent, same acceptance). The quality gate above is what bounds this. The lossy routing-weight miss tolerance (`STRATA_MISS_TOL`, patch `0010`) is in the source and **off**.

**Gaps against the targets** (33 tok/s decode, 600 tok/s prefill; set after the earlier 18 / 500 were met): **neither is reached.**

- decode: **31.0** on the realistic set (-6 %), **26.8** on the short probes (-19 %); 27.0 at 100K and 24.2 at 190K context. The campaign's own ceiling estimate for the exact stack is ~31 realistic / ~27 probes: 33 would need about +1,000 more expert slots in VRAM (+6.5 %) or a near-zero-miss mini-side LRU
  (impossible on 16 GB), neither available without touching the experts, the 200K context, vision or MTP; the 3 probes sit at 23-31 depending on the prompt;
- prefill: best **584** (33.8K, second run; 555 first run), 470 at 18K, 418 at 8K, **176 at 2.5K** (SSD-bound: 477 MB of cold blobs per layer); 600 was never reached in any run of config C (the earlier x8 + pair step touched 586-595 at 33K on a different config);
- not run: GPQA Diamond, tool-calling benchmarks, concurrency above 1 with the shipped engine, a KL against stock exllamav3.

## Run it

The [One-click](#one-click) flow does all of this; the manual steps below are what `run.sh` and the Docker build automate. You need the hardware below, about 190 GB of free disk on the mini, and a Mac with Xcode's Metal toolchain to build two Metal binaries (the mini's Command Line Tools have none).

| | |
|---|---|
| host | Mac mini M4, 16 GiB, macOS 26.6.2, Command Line Tools 26.6 (SDK 26.5 for builds) |
| GPU | RTX 3090 24 GB in an ASM2464 USB dock (USB 3.2 10 Gb/s, PCIe Gen3 x2), external PSU |
| weights | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) branch `3.05bpw_h5_ng5`, revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`; dense part + mmproj from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (IQ3_S); MTP from Qwen/Qwen3.8-Flash-Next `de4b8e4d` |

```sh
git clone https://github.com/Niko1221/Strata ~/strata && git -C ~/strata checkout v0.1.41       # fb58e0dbc839
for p in patches/strata/0*.patch; do git -C ~/strata apply $PWD/$p; done                        # 19 patches, in order (0001-0005 host-tail step, 0006-0019 config C)
zsh shim/install.sh                                                                             # tgcu shim + libcudart.dylib into ~/shim
cp -R exl3/. ~/exl3/ && cd ~/exl3 && zsh cpu/build.sh                                           # NEON tier
python pack_td.py index && python pack_td.py sync && python pack_td.py verify                   # 45.8 GB of experts, sha256-verified against the pinned revision
# libexl3metal (exl3/metal/build.sh) and strata-vision-metal (vision/build_vision_mac.sh) on a Mac with the Metal toolchain; the engine build: docs/reproduce.md section 5
sudo sysctl iogpu.wired_limit_mb=14336 vm.global_user_wire_limit=15032385536 vm.user_wire_limit=15032385536   # config C's 6,400-slot LRU; resets on reboot
caffeinate -dimsu &
zsh configs/render.sh configs/strata-exl3-full512-c.json.tmpl > ~/shim/strata-exl3-full512-c.json    # exe = ~/shim/run-d7.sh, D7_BIN = ~/strata-mac/strata
zsh ~/shim/ls_srv.sh ~/shim/strata-exl3-full512-c.json                                          # ~125-146 s to ready, http://127.0.0.1:8080/v1
python3 bench/split_probe.py try 300                                                            # decode probe
```

The exact steps, every version actually read from the machine, what was re-run and what was not: [docs/reproduce.md](docs/reproduce.md).
`scripts/verify-card-free.sh` re-runs everything that does not need the card (patch apply and the hash manifest of config C, library builds and tests, one layer of the weights fetched and
compared with the manifest). The older configs are fallbacks: `./run.sh --config final-g4-lru-ht` (or `final-g4-lru`, `final-g4`, `final`) renders those templates; they were measured with the engine and shim of their own step (patches up to `0003` / `0004` / `0005` for `final` / `final-g4-lru` / `final-g4-lru-ht`, see patches/README.md); the engine built from all 19 patches was not run with them. Things that could not be pinned are listed in reproduce.md section 8.

## How it works

```text
  RTX 3090 (24 GB), behind the USB dock                    Mac mini M4 (16 GiB)
  +-------------------------------------------+            +-------------------------------------------------+
  | dense part, attention + GDN, MTP head      |            | Metal lane: LRU of 6,400 mlocked, registered     |
  | elastic 200K k8v4 KV (0.23 GiB at 16K cells)|  x, ids, w | expert slots (11.95 GB) + 0.2 GiB static tier;   |
  | 9,958 expert slots (17.27 GiB), hottest    | ---------> | computes every expert that is not in VRAM        |
  | experts by profile rank, 4 whole layers;   |   USB 3.2  | NEON pool = fallback and exact path              |
  | MoE kernels compute them                   | <--------- |                                                 |
  +-------------------------------------------+     y     +----------------------^--------------------------+
                                                                                 | F_NOCACHE preads, 4 KiB-aligned, ~2 GB/s
                                                          +----------------------+--------------------------+
                                                          | NVMe: all 24,576 experts, one 1.86 MB blob each    |
                                                          +-----------------------------------------------------+
```

- **Experts (config C).** 9,958 of 24,576 in VRAM (41 %), 6,400 in the Mac's LRU (26 %), the rest on NVMe. A pick that is in neither is read from NVMe and computed on the Mac; nothing is masked. The 200K K/V is elastic: the cache keeps every VRAM byte the context has not reached and lends it back as the context grows (9,467 -> 8,717 slots at 190K).
- **Decode.** Per layer and verify window, one pre-armed copy-engine pull brings the router output (x rows, expert ids, weights) to the mini, the Metal lane computes the cold experts
  (the ones already in place as one job, the misses read straight into free LRU slots as a second), and one gated push returns the rows plus a flag the GPU waits on. Two USB transactions
  per ring is the floor (0.85 ms); the engine spends about 1.9 ms per ring (0.34 wait for the ring, 1.29 pool, 0.25 push; Decode9 trace) on 44 of the 48 layers (4 are whole-resident and need no ring); those ring numbers are from the Decode9 trace of the Decode8 stack, not re-traced for config C.
- **Prefill.** Cold experts never cross the link: the chunk's fp16 MoE input goes down, the Mac computes the cold (token, expert) pairs with a grouped Metal kernel while the 3090 runs the resident experts, fp16 partial sums come back.
  Six 2,048-token chunks run layer-major in lockstep so each layer's cold blobs are read from NVMe once per group, and the rows cross the link as int8 + a scale per 32 values (config C; the previous configs: four chunks, fp16).
- **The USB link** is half duplex and moves 328-353 MB/s one way with the pipelined transfers in `shim/tgcu.py`; there is no CUDA driver for macOS, `shim/` is one (tinygrad boots the GPU, `tgcu` loads cubins and builds launches, `libcudart.dylib` is a drop-in runtime API).
- **Vision.** The mmproj encoder runs on the mini's GPU (llama.cpp mtmd, Metal) and costs no 3090 memory; it is unloaded after 120 idle seconds.

Details and the full settings table: [docs/how-it-works.md](docs/how-it-works.md).

## Where the time goes

Decode, ms per verify window, config `final-g4-lru` (two steps before config C; the table at the top is config C, the fitted ceiling model of config C's stack is in docs/results.md section 0.5), `STRATA_DECODE_TIMING=1` (`results/raw/lru/dd_e7.log`; the same prompts on the config before the LRU change are in docs/results.md section 2):

| prompt | tokens/window | ms/window | GPU wait for the ring | cold-expert reads | Metal wait | USB transport (derived) | draft | tok/s |
|---|---|---|---|---|---|---|---|---|
| 1 (code) | 3.33 | 198.9 | 27.6 | 78.0 | 17.9 | 59.4 | 10.1 | 16.7 |
| 2 (explanation) | 2.08 | 109.3 | 17.4 | 16.2 | 6.9 | 54.7 | 8.8 | 19.0 |
| 3 (story) | 1.82 | 114.8 | 17.3 | 17.2 | 12.0 | 54.8 | 8.2 | 15.9 |

Prompt 1 is SSD-bound (78 ms of reads per window here, 88 ms before `MAC_PULL_NOCACHE`, at ~2.4 GB/s; about 8,000 LRU slots would be needed to reach 18 there, and the Metal working-set cap of this mini, 11.84 GiB, stops at ~5,600). Prompts 2 and 3
are bound by the 48 x ~2.8 ms of USB rings plus the GPU's own attention/GDN time, which are floors of this design. Prefill: one chunk-layer costs 96 ms at G=3 (link 58, front 29, host ~8) against the 85 ms that 500 tok/s
needs; prompts of 4K tokens and shorter are SSD-bound (477 MB of cold blobs per layer).

## More

- [docs/results.md](docs/results.md): every table (decode and prefill progressions, memory, receipts, reproducibility checks, exactness) with dates and source files
- [docs/how-it-works.md](docs/how-it-works.md): tiers, rings, LRU, prefill lane, the settings of the final config
- [docs/reproduce.md](docs/reproduce.md): versions, build and run steps, what is and is not pinned
- [docs/troubleshooting.md](docs/troubleshooting.md): link resets, Metal working-set errors, G=4 hangs, build errors
- `patches/` (Strata, tinygrad, vision; history of the per-step diffs), `shim/` (tgcu), `exl3/` (kernels, NEON and Metal libraries, packers, profile), `vision/`, `configs/`, `bench/`, `results/` (raw logs, receipts, campaign ledger), `moetier/` (placement simulator records)

Known issues: config C's greedy output depends on LRU contents and the slot count (rounding-level, see above); a restart takes ~2.5 min (146 s to READY) and the first prompts after it read from NVMe; it needs the raised wired limit (above, resets on reboot) and runs with **190 MiB of VRAM free** after load (the engine warns `LOW`; no OOM in a ~110-request regression); the first short request after a 100K-190K prompt re-reads the lent expert slots (6.5-13.7 s);
a bigger LRU than 6,400 slots swaps on this 16 GB mini; the USB dock needs a PCIe power cycle after crashes (the shim does it at every start).

Credits: Qwen3.8-Flash-Next by the Qwen team; the [EXL3 checkpoint](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) and [exllamav3](https://github.com/turboderp-org/exllamav3) kernels (MIT, vendored unmodified
in `exl3/vendor/`) by turboderp; the IQ3_S GSQ-RCO GGUF and mmproj by ISTA-DASLab; [Strata](https://github.com/Niko1221/Strata) (MIT) by Niko1221, patched here; [tinygrad](https://github.com/tinygrad/tinygrad)
(MIT) and the NV-over-USB work of PR #17369; [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, vision encoder); SIMDe (MIT); prior art:
[qwen38-flash-next-b70-offload](https://github.com/sybil-solutions/qwen38-flash-next-b70-offload) and [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite); card art by @samdape.
MIT license for the code in this repository (patches carry the license of the code they patch); the weights carry their own licenses.
