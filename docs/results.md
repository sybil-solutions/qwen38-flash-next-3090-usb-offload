# Results

Every number here comes from a log or receipt in this repo; the path is in the "source" column. Times are UTC as written in the logs.
Nothing is rounded up. "Not run" means not run.

Rig: Mac mini M4 (Mac16,10, 10 cores, 16 GiB, internal NVMe) running macOS 26.6.2, driving one RTX 3090 24 GB over a "chestnut"
USB dock (ASM2464 bridge, GPU link Gen3 x2, 2026-10-10 `LNKSTA 0x1023`) through the tgcu CUDA shim on tinygrad's NV backend.
Engine: Strata v0.1.41 + the patches in [`../patches/`](../patches/README.md). Weights: turboderp EXL3 3.05 bpw, all 512 experts of all 48 MoE
layers (45.77 GB of experts), dense part IQ3_S GSQ-RCO GGUF, MTP draft layer from the BF16 checkpoint (see [reproduce.md](reproduce.md)).

Method (all rows unless a row says otherwise): one request at a time (concurrency 1), greedy (`temperature 0`), reasoning off, MTP
speculative decoding (`--spec 4 --spec-min-p 0.5`), `--kv k8v4 --max-context 204800`, vision encoder configured, all 512 experts computed.
Decode: 3 fixed prompts, up to 300 tokens, run twice right after a server restart (`bench/split_probe.py`); the tok/s is
`predicted_n / predicted_ms` from the server's `timings` (it counts every token the verify windows accepted, MTP drafts included).
Prefill: a document prompt with a random first line so no prefix cache applies, `max_tokens` 2, tok/s = `prompt_n / prompt_ms`
(`bench/ls_pf.py`, `bench/final_long.py pf`). Needle: `bench/final_long.py needle TOKENS DEPTH` (Strata's `tools/needle_bench.py` haystack).

## 0. Latest configuration: `configs/strata-exl3-full512-c.json.tmpl` (config C, Decode16, 2026-10-15)

= the host-tail configuration (section 0a) plus everything the campaign shipped after it, in this order (every item has its patch or file in this repo, see [patches/README.md](../patches/README.md)): the Mac lane takes every chunk (`0006`) and PLE rows are warmed ahead;
prefix replay in the server (`0007`); int8 block-32 transport of the lane's rows and the pair job, G=6 (`0008`, `STRATA_PF_XPORT8=1`, `STRATA_PF_PAIR=1`, `STRATA_PF_GROUP=6`, `STRATA_PF_MIN_CHUNKS=6`); the borrow fix (`0009`) that turns the prefill borrow on (884 slots lent per prompt); tgcu allocator
top reserve + best fit (`shim/0001`); `TGCU_ARM_EARLY` / `TGCU_PULL_PACK` (`shim/0003`); guided/staged refill of the lent slots and the `LRUMIB` vision hook (`0011`, `0012`, `STRATA_D7_MODE=7`); the placement profile with 4 whole-resident layers + staged startup fill (`exl3/placement/d8w4/profile.bin`, `0013`);
4 KiB-aligned SSD reads (`0016`, `0017`, `STRATA_D11_ALIGN=1`, `STRATA_D12_PFALIGN=1`); the LRU at 6,400 slots (`STRATA_EXL3_METAL_SLOTS`, needs the raised wired limit); elastic K/V (`0018`, `shim/0004`, `STRATA_KV_GROW=1`, `STRATA_KV_GROW_RESIDENT=1`: the 200K K/V occupies only the cells in use); elastic indexer keys and
the VRAM knobs (`0019`, `shim/0005`, `TGCU_CODE_MIB=32` ...). Host: `sudo sysctl iogpu.wired_limit_mb=14336 vm.global_user_wire_limit=15032385536 vm.user_wire_limit=15032385536` (set by the user on the mini, resets on reboot), `caffeinate -dimsu`.
Same rig, method and prompt sets as the sections below unless stated: one request at a time, greedy, MTP spec 4, `--kv k8v4 --max-context 204800`, vision encoder configured, all 512 experts computed. Live config md5 `bc8450a1e583f2ba5b3f756832cc1f62`,
engine `strata_d15a` md5 `fc6e4ddd2c9549d6c52a922ffb8ad718`, shim `tgcu.py d2692131`, `tgcu_api.py 9697e39c`, `libcudart.dylib ec0517ec`, Metal lib sha256 `e38a011c...`, server `serve/server.py 780aa4b1`. Raw logs and the tools that produced them:
[`results/raw/decode16/`](../results/raw/decode16/) (`logs/` = the runs below, `tools/` = the scripts as run, `cfg/` = the configs of the arms, key redacted), [`results/raw/decode15/`](../results/raw/decode15/) (VRAM census, simulator).

Start-up (from `logs/d15_final.log`, 2026-10-14 20:14 PDT, READY after 146 s): `expert cache auto: 17.79 GiB free, 310 MiB reserved (+218 MiB for the draft head) -> 9958 slots`, 17.27 GiB of VRAM; staged profile fill 63.7 s (291 MB/s); `exl3-metal: lane ready (6400 slots of 1867776 B registered)`, LRU preloaded in 5.3 s (11.95 GB, F_NOCACHE);
the prompt path borrows 884 cache slots (1.53 GiB); elastic K/V: 16,384 of 204,800 cells in VRAM (0.23 of 2.43 GiB); **190 MiB of VRAM free with everything loaded** (the engine warns `LOW`; no OOM in the ~110-request regression); kernel wired memory 12.4-12.9 GiB, swap used 118 MB at READY, 683 MB at the end of the regression.

### 0.1 Decode, c1, MTP (tok/s)

ABBA run (`ab15.sh`; fresh server restart per arm, 29 realistic prompts of 3.7-19.8K tokens + 12 probe requests per arm; control = the Decode14 config `live_d14.json` on the Decode15 shim with the new knobs unset; 2026-10-14 18:27-19:52 PDT (the mini's local clock, UTC-7; the prefill rows' own timestamps are UTC)), source `logs/ab15.out`, `ctlA/newA/newB/ctlB.out|jsonl|probe.log`:

| arm | realistic 29 prompts (mean tok/s) | 12 probe requests (mean tok/s) | image request |
|---|---|---|---|
| ctlA (Decode14 config) | 30.24 | 26.03 | - |
| newA (config C) | 31.28 | 26.58 | PASS |
| newB (config C) | 30.79 | 27.09 | PASS |
| ctlB (Decode14 config) | 30.27 | 26.00 | - |
| **control -> config C** | 30.25 -> **31.03** (+2.59 %, 24 of 29 prompts faster, mean ratio 1.026, mean TTFT 24.27 -> 23.75 s) | 26.02 -> **26.84** (+3.1 %) | |

Final regression of the config (`logs/d15_final.log`, 2026-10-14 20:14-20:55 PDT, one fresh start, ~110 requests): realistic 29 prompts mean **30.82** tok/s (TTFT mean 23.75 s; per prompt 28.2-34.0; the 126-token reply of p29 20.0); decode probes
(prompt 1 code / 2 explanation / 3 story; `decode probes x2` = the sequence ran twice): first pass 29.9 / 25.7 / 23.7, then 27.0 / 29.6 / 23.3 and 27.0 / 31.1 / 23.6; second pass 26.3 / 26.9 / 23.5, then 26.9 / 29.8 / 23.4 and 26.9 / 28.6 / 24.7; after the whole regression 29.3 / 29.4 / 23.6 and 26.8 / 29.6 / 24.8.
Time to first token of the 29-35 token prompts: 0.85-1.7 s.
The earlier tables of this file use the same three probe prompts (sections 0a, 1, 3).

### 0.2 Prefill and context, c1 (tok/s)

| prompt tokens | prefill speed (run 1 / run 2) | decode over the answer | note | source |
|---|---|---|---|---|
| 2,565 | 176.1 / 176.7 | not measured | SSD-bound (477 MB of cold blobs per layer) | `d15_final.log` |
| 8,018 / 8,017 | 383.0 / 417.8 | not measured | | same |
| 18,330 / 18,329 | 445.9 / 470.1 | not measured | | same |
| 33,842 / 33,841 | 555.4 / 584.2 | not measured | 5 successive 33K needles (depth 10 / 50 / 90 / 30 / 70 %): all found, 70 / 71 / 46 / 55 / 54 s | same |
| 97,811 | 560.9 (TTFT 174.6 s) | 27.0 (300 tokens, MTP 171/266) | needle `meadow-copper-504` found; K/V grew to 98,304 cells (1.16 GiB), the cache lent 538 slots, trimmed back afterwards | `long_C.log` |
| 189,481 | 548.1 (TTFT 345.9 s) | 24.2 (300 tokens, MTP 168/263) | needle found; K/V grew to 196,608 cells (2.30 GiB), 1,241 slots lent; probes afterwards 27.4 / 29.6 / 25.4 and 26.4 / 29.0 / 24.2 | same |
| 189,450 | 360 s total | not measured | the 190K needle of the final regression: found | `d15_final.log` |

Agent-turn pattern after a 3,073-token base prompt (`d15_final.log`): 30 new tokens 2.56 s, 100 new 6.14 s, 250 new 7.44 s, 1000 new 11.8 s wall. Tool-call reuse: turn 2 read 40 tokens and reused 3,379 of the 3,340-token turn 1 (2.8 s): PASS. Image: the NYT front page photo,
`This is The New York Times, and its headline reads: "MEN WALK ON MOON ..."` (16.4 s, 329 prompt tokens): PASS. GPU timeouts in the whole regression: 0.

### 0.3 Quality gate (teacher-forced)

`logs/d15_q.log` (`q15.sh`): 8 prompts (code C++/Python/CUDA, agentic, 9K long context, chat, math, docs; 1.4-9.5K tokens), the engine logs the top-64 ids and log-probabilities of every position of a second turn read through the decode windows (`STRATA_LOGPOS`),
scored against the saved replies of the baseline stack "p-off" (the config with the int8 prefill transport off, Decode5-era): **705 positions, coverage 100 %, top-1 agreement 98.01 %, mean KL 0.004110 nats, p99 KL 0.058355 nats, mean |dlogp(target)| 0.0695: GATE PASS**
(gate: top-1 >= 97 %, mean KL < 0.01, p99 KL < 0.1, coverage >= 99 %; proposed by the builder of the transport, not a user number; KL is a top-64-support estimate because the server has no logprobs API). Per prompt top-1: 93.5-99.0 %. Comparisons with the previous stacks, which are not the gate:

| compared pair | top-1 | mean KL | p99 KL |
|---|---|---|---|
| p-off -> config C | 98.01 % | 0.00411 | 0.0584 (PASS) |
| Decode13 stack -> config C | 97.59 % | 0.00735 | 0.1041 |
| Decode12 stack -> config C | 97.73 % | 0.00515 | 0.1110 |

Free-running greedy replies of config C and the Decode13 stack are identical on 5 of 8 prompts. The mean |dlogp| gate originally proposed (< 0.03) is not met by any configuration of this campaign (an equivalent config with a different slot count or chunking moves it by ~0.07, the engine itself is bit-deterministic run to run); the gate that was applied is the one above.
The reference is the campaign's own earlier greedy output, **not stock exllamav3**.

### 0.4 Concurrency and the batch engine (not part of config C)

One request at a time is how config C runs: the server queues the other requests (FIFO), `--batch` is off. Measured at `--max-context 40960` (the batch engine has no elastic K/V; each slot session carves all 40,960 cells, so expert slots shrink 9,983 -> 9,113 / 9,192 / 8,387 / 8,543 for batch2m / batch2 / batch4m / batch4;
`m` = with the MTP draft), config C as the base, `logs/abconc_*`, tool `abconc15b.sh`/`abconc15c.sh`. Aggregate = generated tokens / wall span (TTFT included); per-request decode = tokens / (wall - TTFT) at the client; TTFT = mean.

| 6-9K prompts, 300 tokens (aggregate tok/s / per-request decode tok/s / TTFT s) | c=1 | c=2 | c=4 |
|---|---|---|---|
| control (`--max-context 40960`, FIFO queue, 8 prompts) | 9.71 / 31.9 / 21.5 | 10.16 / 32.2 / 46.2 | 10.21 / 32.7 / 86.7 |
| batch4m (8 prompts) | 5.42 / 28.3 / 44.7 | 2.17 / 1.4 / 63.0 | 2.01 / 0.6 / 104.8 |
| batch2m | 5.11 / 29.2 / 48.4 | 2.42 / 1.8 / 70.8 | 2.36 / 1.7 / 211.1 |
| batch4 (no draft) | 4.98 / 28.9 / 49.9 | 2.23 / 1.5 / 65.1 | 2.74 / 0.9 / 93.7 |
| batch2 (no draft) | 5.22 / 30.7 / 47.7 | 2.40 / 1.7 / 65.4 | 2.38 / 1.7 / 209.0 |

Short prompts (30 tokens, 8 x 300 tokens, isolates the batch decode rate): control 25.13 / 27.7 / 1.0 (c=1), 25.29 / 27.4 / 10.0 (c=2), 26.21 / 28.4 / 24.1 (c=4); batch4m 22.60 / 25.6 / 1.4, 2.04 / 1.1 / 2.2, 1.73 / 0.5 / 4.4; batch2m 23.39 / 26.8 / 1.5, 2.14 / 1.1 / 2.1, 2.36 / 1.9 / 195.0;
batch4 23.25 / 25.7 / 1.2, 2.11 / 1.1 / 1.7, 1.67 / 0.45 / 2.9; batch2 24.37 / 26.7 / 1.0, 2.45 / 1.3 / 1.5, 2.48 / 1.3 / 167.4. **Finding:** a batch window averages 2-3.5 rows and takes 540-1,650 ms (GPU-reach wait 18-26 ms + CPU experts 513-1,628 ms, 3.8-12.6 CPU-computed experts per layer-window, `--pcie-frac 0`): the misses are computed on the generic CPU tier, not through the c=1 Metal lane / ring that gives ~100 ms windows;
a 6-9K prompt is read in 3-4 parts with the other slots decoding for 11-19 s between them (prefill 100-120 tok/s against ~400). Aggregate is -80 % against the c=1 control, c=1 on a batch config -11 %; no fault, no GPU timeout, the engine stayed alive on every arm. The +20 % condition was not met, no per-slot elastic K/V was attempted in this step.
Follow-up work (not in this repo, not promoted, config C unchanged): the campaign's later ledger rows record batch misses routed through the Metal lane and per-slot elastic K/V built and measured at 200K: short-prompt c=4 aggregate 24.9-26.5 tok/s against 25.3 for the solo control (+5 % best), real-prompt arms limited by the batch engine's chunk-by-chunk prompt read; engine work is still in progress
(`results/campaign/ledger.jsonl`, 2026-10-15T10:50Z and T19:10Z rows).

### 0.5 Where the decode window goes now (model, not a measurement of config C)

The campaign fitted `W[ms/window] = 25.8 + 1.06 x (48 - k) + 0.0424 x N_in + 0.64 x M` (k = whole-resident layers, N_in = in-place cold experts per window, M = LRU misses per window) to the Decode7/8 stack (final fit: 25.8 fixed + 46.6 rings + 12.4 in-place Metal + 19.1 misses = 104.0 ms modelled, 100.7 measured). The elastic K/V and the VRAM knobs
moved M and N_in (more VRAM slots: +1,074 slots in Decode14, +491 in Decode15; the simulator priced +1,000 slots at ~7 ms/window). Not re-fitted for config C. At the calibrated terms the exact stack tops out near 31 tok/s realistic / 27 probes; 33 would need about +1,000 more VRAM slots (+6.5 %) or a miss-free LRU
(impossible on a 16 GB mini), neither available without touching the experts, the 200K context, vision or MTP (the Decode16 ledger row). Simulator and census: `results/raw/decode15/` (`sim15.py`, `table15_items.out`, `census.out`).

### 0.6 Progression since the host-tail step

Realistic = the 29-prompt set (or the 10-prompt subset where only that was run, marked); probes = the three probe prompts. Prefill = run 1 / run 2 at 2.5K / 8K / 18K / 33K tokens where the step measured them. Sources: the campaign log entries of that date in [`results/campaign/EXL3_PLAN-log.md`](../results/campaign/EXL3_PLAN-log.md) (each step's own arms were interleaved ABBA with a control;
numbers of different steps were taken in different sessions and are not strictly comparable: the plan notes bimodal decode after the 2026-10-12 14:46 reboot until `caffeinate` was set at 19:36).

| step (UTC) | what | decode | prefill 2.5K / 8K / 18K / 33K |
|---|---|---|---|
| 2026-10-12 07:35 | host-tail (section 0a) | probes 18.2 / 21.4 / 18.1 | - / 428 cold, 481 warm @7.9K / 407 @17.6K / 396 @33K |
| 2026-10-12 13:15 | Prefill600: lane min 1 chunk, PLE warm-ahead (`0006`) | probes 18.6 / 21.8 / 19.1 and 18.3 / 22.6 / 17.1 | 225 / 227, 416 / 456, 433 / 435, 435 / 440 |
| 2026-10-12 18:40 | x8 + pair job, G=6, reserve 700 (`0008`) | - | 216 / 218, 441 / 491, 470 / 506, 563 / 586 |
| 2026-10-13 06:30 | Decode5: borrow fix (`0009`), reserve 600, 8,393 slots | probes 19.2 / 21.9 / 18.9 and 19.2 / 22.6 / 19.5 | 123 / 124, 301 / 323, 389 / 412, 506 / 524 |
| 2026-10-13 10:10 | Decode6: ARM_EARLY + PULL_PACK (`shim/0003`) | probes 20.1 / 22.9 / 20.1 and 20.2 / 23.8 / 20.8 | 120 / 125, 301 / 326, 385 / 408, 508 / 525 |
| 2026-10-13 14:10 | Decode7: guided/staged refill, LRU 6,000 (`0011`) | probes mean 21.66 (12 requests) | 156 / 156, 360 / 396, 433 / 453, 540 / 561 |
| 2026-10-13 18:50 | Decode8: 4 whole layers + LRU 6,200 + staged fill (`0013`) | realistic 27.41, probes 23.77 | 156 / 157, 343 / 373, 413 / 428, 509 / 523 |
| 2026-10-14 06:10 | Decode11: aligned SSD reads (`0016`) | +2.2 % realistic (10-prompt set), +3.5 % probes | 155 / 156, 344 / 376, 413 / 429, 501 / 523 |
| 2026-10-14 10:05 | Decode12: aligned prefill reads (`0017`) | realistic 28.49 (10 prompts), probes 24.43 | 163 / 161, 348 / 384, 423 / 442, 524 / 541 |
| 2026-10-14 16:40 | Decode13: LRU 6,400 slots (config only) | realistic 28.63, probes 24.8 | 161 / 162, 347 / 382, 423 / 442, 522 / 538 |
| 2026-10-14 19:55 | Decode14: elastic K/V (`0018`, `shim/0004`) | realistic 30.30 (29 prompts), probes 26.06; 100K decode 24.8, 190K 23.5 | 174 / 176, 367 / 409, 446 / 468, 556 / 575 |
| 2026-10-15 09:00 | **Decode16: config C** (`0019`, `shim/0005`) | **realistic 31.03 (ABBA; 30.82 final regression), probes 26.84; 100K 27.0, 190K 24.2** | **176 / 177, 383 / 418, 446 / 470, 555 / 584** |

## 0a. Previous configuration: `configs/strata-exl3-full512-final-g4-lru-ht.json.tmpl` (host-tail step)

= the configuration of section 1 plus `TGCU_POKE1=15` (GPPut / doorbell / hold-word as one control transfer), `TGCU_FREE_DEFER=1` (cudaFree hands the block back when its submission completed instead of a
full synchronize per free), `TGCU_H2D_RELAX=1` (a bulk push waits only for the launches whose parameters fall inside its destination range), `TGCU_TS_G=150` (compute channel group time slice
1024 -> 150 us: the copy engine is no longer slowed by a busy GPU during the decode window), `STRATA_PF_SPLITF=1` with `STRATA_PF_SPLITF_MIN=4` (patch `0005`: the split order only for groups of 4+ chunks) and
`EXL3M_WAIT_SPIN=1` (decode Metal jobs poll the command buffer). The sources of these changes are in the repo (`shim/tgcu*.py` and `exl3/metal` carry them, `patches/strata/0005`, the per-step diffs in `patches/history/hosttail/`);
`libexl3metal.dylib` rebuilt from the repo has the same sha256 as the deployed one (`e38a011c79933a8e...`), the patched `shim/*.py` files and `prefill.cpp` are byte-identical to the deployed ones (checked 2026-10-12).
Run 2026-10-12 07:26:46Z to 07:34:39Z on a freshly restarted server, same weights/placement/memory layout as section 1 (7,400 VRAM slots, 716 MiB free, LRU 5,600 slots).
Source: [`results/raw/hosttail/ht_final.log`](../results/raw/hosttail/ht_final.log) (`split_HF_{a,b,c}_{0,1,2}.txt` and `split_HT9_st_*.txt` in the same directory are the generated texts of the three decode probes of the tuning/instrumented runs, not timings).

### 0a.1 Decode, c1, MTP (tok/s)

| prompt | run 1 | run 2 | after the prefill rows + 32K needle | MTP drafts accepted/proposed (run 1) |
|---|---|---|---|---|
| 1: "write is_prime and explain it" (29 prompt tokens, 300 out) | **18.2** | 17.9 | 17.5 | 213/258 |
| 2: "explain how an MoE picks experts" (35 prompt tokens, 253 out) | **21.4** | **21.3** | 21.5 | 134/225 |
| 3: "lighthouse keeper story" (29 prompt tokens, 292 out) | **18.1** | 17.7 | 17.5 | 138/260 |

Time to first token for these 29-35 token prompts is 2.1-2.5 s. (Section 1 was 16.8 / 19.0 / 16.3 and 16.6 / 19.0 / 16.0.) The stats-instrumented rerun of the same config
([`ht9_st.log`](../results/raw/hosttail/ht9_st.log), `STRATA_PF_MAC_STATS` + `STRATA_PF_MAC_RDSTATS` on): 18.6 / 21.4 / 17.9.

### 0a.2 Prefill and context, c1 (tok/s)

| prompt tokens | prefill speed | decode over the answer | note |
|---|---|---|---|
| 3,783 | 277.3, 333.6 | not measured | two runs, different random first line (276.2, 325.8 in `ht9_st.log`) |
| 7,905 | 427.8 (cold), 481.3 (warm) | not measured | 431.3, 482.1 in `ht9_st.log`; the cold-vs-warm gap (~2.1 s) is layers 0-1 waiting for SSD reads of experts that are not in the LRU |
| 17,610 | 406.8 | not measured | 9 chunks -> groups of 3, `STRATA_PF_SPLITF_MIN=4` leaves the split order off there |
| 33,253 (32K needle) | 396.0 (TTFT 84.0 s) | 17.3 tok/s (124 tokens, MTP 67/112) | needle `meadow-copper-504`, depth 50 %, found |
| 97,461 / 188,835 | 374.9 / 382.4 (earlier config) | 16.5 / 13.9 (earlier config) | **not re-run** on this config, see section 1.3 |

Text parity: 24 greedy tokens at 3.8K and 7.9K are byte-identical to the reference run with the host-tail env off ([`ht_ex2_ts150.log`](../results/raw/hosttail/ht_ex2_ts150.log)). Vision: the `final_vision.py` step of
`ht_final.log` aborted (`No module named 'PIL'`); the plan notes report the image answers (NYT photo, bar chart) as OK on this config, there is no raw log of that in this repo. Prefill >= 500 tok/s was **not reached** (best 481 warm).

## 1. Previous configuration: `configs/strata-exl3-full512-final-g4-lru.json.tmpl`

Run `final_lru` on 2026-10-12, 06:12:48Z to 06:21:32Z, started on a freshly restarted server. 7,400 expert slots in VRAM (12.84 GiB),
716 MiB of VRAM free after load, Metal LRU of 5,600 registered slots (10.46 GB), 0.2 GiB static RAM tier, prefill group G=4.
Source for 1.1-1.3: [`results/raw/lru/final_lru.log`](../results/raw/lru/final_lru.log).

### 1.1 Decode, c1, MTP (tok/s)

| prompt | run 1 | run 2 | after the prefill rows + 32K needle | MTP drafts accepted/proposed (run 1) |
|---|---|---|---|---|
| 1: "write is_prime and explain it" (29 prompt tokens, 300 out) | 16.8 | 16.6 | 16.3 | 213/258 |
| 2: "explain how an MoE picks experts" (35 prompt tokens, 253 out) | **19.0** | **19.0** | 18.8 | 134/225 |
| 3: "lighthouse keeper story" (29 prompt tokens, 292 out) | 16.3 | 16.0 | 16.3 | 138/260 |

Time to first token for these 29-35 token prompts is 2.2-2.6 s.

### 1.2 Prefill, c1 (tok/s)

| prompt tokens | prefill speed (two runs of the same shape, different random first line) | note |
|---|---|---|
| 3,783 | 276.9, 329.8 | first run after the decode probes, second run |
| 7,905 | 382.6, 421.7 | |
| 8,218 | 346.1 | one run |
| 17,610 | 392.1 | one run |
| 33,253 (32K needle) | 378.1 | TTFT 87.9 s, 32K needle found (`meadow-copper-504`, depth 50 %), decode 15.9 tok/s over 130 tokens |

### 1.3 Context

| prompt tokens | prefill speed | decode over the answer | needle | config | source |
|---|---|---|---|---|---|
| 33,253 | 378.1 tok/s (TTFT 87.9 s) | 15.9 tok/s (130 tokens, MTP 71/102) | found | final-g4-lru | `results/raw/lru/final_lru.log` |
| 97,461 | 374.9 tok/s (TTFT 260.0 s) | 16.5 tok/s (131 tokens, MTP 74/97) | found (depth 25 %) | `g4c` = final-g4 + `TGCU_HANG_DUMP=1 TGCU_WAIT_S=30` (diagnostic env only), before the LRU change | `results/raw/g4/lr_g4c_100k.log` |
| 188,835 | 382.4 tok/s (TTFT 493.8 s) | 13.9 tok/s (134 tokens, MTP 66/121) | found (depth 50 %) | final-g4 (before the LRU change) | `results/raw/g4/final_g4.log` |

The 200K context itself is allocated in every row above (`--max-context 204800`, k8v4 KV, 13 KV layers: 12 QSA + the MTP draft; the 36 GDN
layers carry constant state). The 97K and 189K rows were measured on the preceding config (LRU 1,000 slots, 5.1 GiB static tier): the 190K
needle was **not** re-run on the final LRU config.

### 1.4 Vision

An image request through `/v1/chat/completions` is answered correctly: the New York Times front page photo (headline "MEN WALK ON MOON",
640x488 and 1280x960) and a generated bar chart (Q4 = 45 highest, Q1 = 12 lowest). Encoder = `strata-vision-metal` on the mini (llama.cpp mtmd + the BF16
mmproj), 0 bytes of 3090 VRAM; 1.2 s to encode a 566-token chart, 2.5 s for 993 tokens, 4.1 s cold start + encode with `idle_unload_s: 120`.
Source: `results/campaign/GOAL.md` status table ("vision" row, 2026-10-11) and `results/campaign/EXL3_PLAN-log.md` 2026-10-12T00:05Z and 06:25Z.
**No raw vision log is in this repo**: `bench/final_vision.py` failed in the `final_lru` run (`No module named 'PIL'`, see the log); the answers on
the final-g4-lru config are reported in the 06:25Z plan entry, the encode timings are from the earlier final config.

## 2. Where the decode window goes (final vs the config it replaced)

`STRATA_DECODE_TIMING=1`, per prompt, ms per verify window; "transport" is derived: per-layer host - plan - jobs - CPU. 48 ring layers per window.
Sources: [`results/raw/lru/dd_e0.log`](../results/raw/lru/dd_e0.log) (live G=4 config before the change: LRU 1,000 slots + 5.1 GiB tier) and
[`dd_e7.log`](../results/raw/lru/dd_e7.log) (final: LRU 5,600 + `STRATA_MAC_PULL_NOCACHE=1`).

| prompt | config | tokens/window | ms/window | GPU-reach wait | cold-expert reads ("jobs") | Metal wait ("CPU") | transport | draft | tok/s (derived) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | before | 3.23 | 291.55 | 31.20 | 135.39 | 35.12 | 73.87 | 10.17 | 11.1 |
| 1 | final | 3.33 | 198.88 | 27.59 | 78.04 | 17.91 | 59.39 | 10.10 | 16.7 |
| 2 | before | 2.03 | 119.40 | 20.57 | 17.48 | 10.73 | 56.36 | 8.88 | 17.0 |
| 2 | final | 2.08 | 109.29 | 17.39 | 16.20 | 6.88 | 54.73 | 8.75 | 19.0 |
| 3 | before | 1.98 | 144.19 | 23.22 | 27.44 | 17.65 | 61.77 | 8.73 | 13.7 |
| 3 | final | 1.82 | 114.81 | 17.33 | 17.18 | 12.01 | 54.79 | 8.22 | 15.9 |

(The remaining ~6 ms of each window is stage + commit/emit.) Per ring layer, mean over 750 windows with `STRATA_HS_TRACE=1`
(plan entry 2026-10-12T06:25Z; the trace log itself is not in the repo): wait for the ring 0.431 ms, x pull residue 0.057, pool call 1.85
(SSD misses + Metal wait), y + flag push 0.419, flag 0.023, host after the ring 2.36, so about 2.8 ms x 48 layers = 134 ms of the window.
One pull and one push per ring is already the floor (two USB transactions = 0.85 ms).

The cold-expert placement per lookup (36.5K-call trace, 11.4 cold experts per call), before: static tier 22 %, LRU hit 45 %, miss 33 % (3.58 misses
per call, 67.5 % in place); after: 89.2 % in place, 1.23 misses per call (plan entry 06:25Z; trace `results/raw/decode/` has the sim scripts, not the 2 MB trace).

## 3. Decode progression, same three prompts (tok/s p1 / p2 / p3, c1, MTP)

Full-512 exact-experts configs only. Source for a row = the ledger row (`results/campaign/ledger.jsonl`) and the log named there, kept under `results/raw/`.

| date (UTC) | change | decode | source |
|---|---|---|---|
| 2026-10-11 17:45 | first end-to-end full-512 run, 17 whole layers resident, 1.6 GiB RAM tier | 4.4 / 5.6 / 5.4 | `mini-shim-logs/split_F1_full512_a.log` |
| 2026-10-11 18:20 | frequency-ranked placement, every layer rings | 5.2 / 7.4 / 5.7 | `mini-shim-logs/split_F4_full512_d_freq.log` |
| 2026-10-11 18:50 | + no prefill borrow (servable config) | 4.9 / 7.4 / 5.5 | `mini-shim-logs/split_F5_full512_e_freq_noborrow.log` |
| 2026-10-11 19:55 | pre-armed USB handshake + deferred pull + whole-blob preads (`HS_PRE`, `HS_DEFER`, `MAC_PULL`), texts byte-identical to the row above | 9.3 / 12.4 / 8.5 | `mini-shim-logs/split_H4_pull.log`, `split_H9_final.log` |
| 2026-10-11 21:00 | Metal lane with an LRU of 1,000 registered slots | 10.3 / 14.7 / 10.6 | `mini-shim-logs/split_WF6_m_final.log` |
| 2026-10-12 00:05 | + 200K k8v4 KV and vision (7,820 slots, -1,288) | 9.0 / 12.3 / 9.4 | ledger row 2026-10-12T00:05Z |
| 2026-10-12 02:07 | in-place/miss split job + early start (`METAL_SPLIT`, `METAL_EARLY`, `MAC_PULL_EARLY`) | 10.4 / 17.2 / 13.8 and 10.1 / 17.2 / 12.1 | `mini-shim-logs/dec_final.log` |
| 2026-10-12 05:28 | G=4 prefill + `--vram-reserve-mib 400` (live config before the LRU change) | 11.5 / 17.0 / 14.3 and 11.1 / 17.0 / 13.7 | `raw/lru/dd_e0.log` |
| 2026-10-12 05:33 | LRU 3,800 slots, tier 0.2 GiB, cold start | 12.7 / 17.5 / 14.3 and 12.6 / 16.4 / 14.5 | `raw/lru/dd_e1.log` |
| 2026-10-12 05:44 | LRU 4,400 + preload by profile rank | 13.7 / 17.3 / 15.9 and 13.3 / 17.9 / 15.5 | `raw/lru/dd_e2.log` |
| 2026-10-12 05:44 | LRU 5,600 + preload | 16.1 / 19.0 / 16.2 and 15.9 / 18.8 / 15.9 | `raw/lru/dd_e3.log` |
| 2026-10-12 05:59 | + SSD miss reads split in 4 parallel preads (negative, removed) | 15.5 / 17.6 / 15.1 and 15.2 / 16.9 / 15.6 | `raw/lru/dd_e5.log` |
| 2026-10-12 06:08 | + `MAC_PULL_NOCACHE=1` (**final**) | **16.9 / 19.0 / 16.3** and 16.8 / 19.0 / 15.8 in this run; **16.8 / 19.0 / 16.3** and **16.6 / 19.0 / 16.0** in `final_lru.log` | `raw/lru/dd_e7.log`, `raw/lru/final_lru.log` |

Negative results measured on the way (config, log in `results/raw/mini-shim-logs/` or `raw/decode/`): pool QoS + in-call spin, early pull started
from the CE push, 256K read chunks, look-ahead warm into a pread queue (engine died, root-caused, no gain after the fix), bigger locked RAM
tier (locked RAM displaces file cache 1:1), `--kv-resident 20480`, `--spec 6`, router look-ahead, SSD miss reads split x4, co-occurrence predictor
(9-16 % coverage, gate 80 %: not built), whole-resident-layer placement (wash at 6 layers, worse at 10+).

## 4. Prefill progression (tok/s by prompt tokens)

Source paths are relative to `results/raw/`.

| date (UTC) | change | prefill | source |
|---|---|---|---|
| 2026-10-11 17:45 | every cold expert streamed over USB per 2,048-token chunk (138 s per chunk) | 9.5 @2,562 / 10.8 @4,473 / 12.0 @6,991 | `mini-shim-logs/pf_F1.log`, `pf_F1_8k.log` |
| 2026-10-11 21:00 | same, after the decode work | 14.4 @3,467 | `mini-shim-logs/pf_WF6_m.log` |
| 2026-10-11 22:10 | Mac Metal cold-expert prefill lane (`STRATA_PF_MAC`) | 128.6 @3,467 | ledger row 2026-10-11T22:10Z (engine log lines not kept) |
| 2026-10-11 23:55 | fp16 partial sums, F_NOCACHE preads, per-prompt slot sets | 50 / 109 / 139 / 156 @569 / 2,753 / 4,473 / 7,348 | ledger row 2026-10-11T23:55Z |
| 2026-10-12 00:10 | pipelined USB transfers (h2d 228 -> 353 MB/s, d2h 154 -> 328 MB/s) | 47 / 121 / 140 / 154-160 @569 / 2,909 / 4,473 / 7,348 (SSD-bound) | `mini-shim-logs/final_pp.out`, ledger row 2026-10-12T00:10Z |
| 2026-10-12 01:10 | layer-major chunk-group lockstep G=3 | 231 @3,783, 178 @8,218, 363 @17,610 | `mini-shim-logs/ls_final.log` |
| 2026-10-12 03:22 | even chunks, cross-group read-ahead, fp16-I/O Metal prefill | 39 @569, 260-279 @3,783, 364 @6,991, 401 @7,905, 311 @8,218, 383 @17,610 | `mini-shim-logs/pf500_final.log` |
| 2026-10-12 04:26 | G=4 + `--vram-reserve-mib 400` | 259-287 @3,783, 373-414 @7,905, 316-318 @8,218, 385 @17,610 | `raw/g4/g4v_fix1.log`, `raw/g4/dc_g4c.log` |
| 2026-10-12 06:12 | final (LRU hits read in place by the prefill lane, prompt sets borrowed from the LRU) | 277-330 @3,783, 383-422 @7,905, 346 @8,218, 392 @17,610, 378 @33,253 | `raw/lru/final_lru.log` |

Limits found with measurements (`results/raw/duplex/`, `results/raw/pf500/`, plan entries 2026-10-12T00:10Z and 03:40Z): the USB link is half
duplex through the bridge (simultaneous arm drops the upload; 104.0 vs 104.2 MB/s sequential); prompts <= 4K tokens are SSD-bound
(477 MB of cold blobs per layer at ~2.0 of 2.41 GB/s); at G=3 one chunk-layer costs 96 ms (link 58, front 29, host ~8) against the 85 ms that 500 tok/s needs.

## 5. Memory and placement

| item | value | source |
|---|---|---|
| experts | 24,576 (48 layers x 512), 1,862,400 B each = 45,770,342,400 B, `exl3-k3-mul1-v1` | `results/pack/exl3_pack.json` |
| expert slots in VRAM | 7,400 (12.84 GiB), pre-filled from the profile in 64.1 s | `raw/lru/final_lru.log` |
| VRAM free after load | 716 MiB (`--vram-reserve-mib 400`) | same |
| Metal LRU | 5,600 slots x 1,867,776 B = 10.46 GB, mlocked, preloaded by profile rank in 4.8 s | same |
| static RAM tier | 0.2 GiB (115 blobs), was 5.1 GiB | plan entry 06:25Z |
| Metal working-set cap on the mini | 11.84 GiB (`recommendedMaxWorkingSetSize`); this, not RAM, bounds the LRU | `raw/lru/ws.m`, plan entry 06:25Z |
| engine RSS / kernel wired at READY | 10.57 GB RSS (`ps`), 11.6 GB wired | `raw/lru/final_lru.log` |
| 200K KV, k8v4 | 816 B/cell/layer, 2.61 GB; costs 1,308 of the 9,108 slots that fit without it | `results/campaign/GOAL.md` (Vision + 200K plan) |

## 6. Kernel and library receipts (card-free unless marked)

| component | result | source |
|---|---|---|
| EXL3 decode (reconstruct) on the 3090 | bit-exact, 50/50 | `results/kernels/m1_receipt.json` |
| exllamav3 GEMV via tgcu | 27/27, rel RMS <= 5.6e-4 (m 1/3/8); GEMM 69/69, rel RMS <= 1.1e-3 | `gemv_receipt.json`, `gemm_receipt.json` |
| fused MoE decode (`exl3_moe_coop`) | 11/11, rel RMS <= 8.1e-4; 108.8 us per layer at bsz 1 | `moe_receipt.json` |
| expert pack -> VRAM MoE | 12/12 (rel RMS <= 6.2e-4); device plan kernel 14/14 | `gpu_receipt.json`, `plan_receipt.json` |
| Strata EXL3 GPU arms (decode, prefill) | see receipts | `exl3s_receipt.json`, `exl3s_pf_receipt.json` |
| NEON CPU tier (`libexl3cpu`) | 106/106: decode + had_in bit-exact (K 1-8 x 3 codebooks), GEMV rel RMS <= 5.6e-7 | `cpu_receipt.json`, `cpu_run.log` |
| Metal lib (`libexl3metal`) vs NEON | all pass: rel RMS 4.5e-5 to 7.3e-5 (gate 1e-3), negatives detected, deterministic, thread-safe | `metal_test_run.log`, `metal_test_receipt.json` |
| Metal throughput | decode wall m=1 E=10/13/20: 0.659 / 0.793 / 1.107 ms; prefill 129 ms/layer @2,048 (1.56 TFLOPS), 344 ms @8,192 (2.34 TFLOPS) | `metal_bench_run.log` |

Those receipts were produced on 2026-10-10 / 11 by the scripts in `exl3/` and `exl3/strata_m4/` on the original machine. The GPU ones need the card
and were not re-run for this repo; the NEON and Metal ones were re-run on 2026-10-12, see section 8.

## 7. The weights

`results/pack/exl3_pack.json` holds, per layer, the sha256 of that layer's 953,548,800-byte section of `experts.bin`; all 48 layers are marked
`turboderp/Qwen3.8-Flash-Next-exl3@69e33439 3.05bpw_h5_ng5`. `results/pack/td_verify.log` is the full `pack_td.py verify` run of 2026-10-11 (every
layer: 6 experts re-fetched from Hugging Face and byte-compared, section sha256 equal to the manifest; final line `VERIFY OK`).

## 8. Reproducibility checks run on 2026-10-12 (card-free)

Run by `scripts/verify-card-free.sh` on the mini; raw output in [`results/verification/`](../results/verification/).

| check | result |
|---|---|
| Strata patches 0001-0004 `git apply --check` + apply, in order, on a fresh `git archive v0.1.41` (`fb58e0db`) | all four apply; resulting files byte-identical to the deployed trees (24/24 files) |
| the same series on a fresh public clone of `Niko1221/Strata` tag `v0.1.41` (driver Mac) | tag resolves to `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`; vision patch applies |
| `libexl3metal.dylib` rebuilt from this repo (driver Mac, `exl3/metal/build.sh`) | sha256 `de43149055d8...` = the deployed dylib, byte for byte |
| `libexl3metal` test on the mini with the rebuilt dylib + rebuilt `libexl3cpu` | ALL PASS (same checks as the original receipt) |
| `libexl3cpu.dylib` rebuilt (`exl3/cpu/build.sh`, SDK 26.5 or 26.0) | sha256 `9fe9cab5491e...` = the deployed dylib, byte for byte (an absolute `-o` path changes the bytes: clang puts it in the install name; the script builds with a relative name; SDK 15.x differs in 52 bytes) |
| `test_exl3_cpu.py` with the rebuilt dylib | ALL PASS 106/106 (the real-expert cases read 6 experts of layer 24 from the pack instead of the retired encoder fixture; `results/verification/cpu_receipt_rebuilt.json`) |
| `configs/render.sh` on the three config templates with the mini's `$HOME` | byte-identical to the live `~/shim/strata-exl3-full512-{final-g4-lru,final-g4,final}.json` |
| `pack_td.py index` | regenerated headers byte-identical to the existing index (sha256 `2dc6b0ce...`, `0ea6ec8c...`) |
| `pack_td.py sync 24` into a temp pack (0.95 GB fetched from the pinned revision, 76 s) | section sha256 `80eb9002da51...` = manifest layer 24; NEON forward vs reference rel RMS 5.2e-4 |
| `exl3_strata_pf.fatbin`, `exl3_strata_recon.fatbin` rebuilt with nvcc 13.4 in the container | sha256 identical to the deployed fatbins |
| `exl3_strata.fatbin` rebuilt | its id depends on the build path: built from `/Users/sero/exl3` it is **byte-identical** to the deployed one (`d19f0e17...`), from `/build/exl3` it is `82e480e5...` (same size 662,480 B; the earlier three different hashes were three different directories). The Docker build uses `/Users/sero/exl3` ([reproduce.md](reproduce.md#10-one-click-what-was-run)) |
| `libstrata_exl3.dylib` host library rebuilt (`exl3/strata_m4/build.sh`) | same size (4,450,936 B) and exported symbols as the deployed one; sha differs because it embeds the fatbin above |
| `tgcu_builtin.fatbin` / `tgcu_libc.cubin` rebuilt | **byte-identical** to `shim/prebuilt/` and the deployed files ([reproduce.md](reproduce.md#10-one-click-what-was-run)): `nvcc -std=c++17 -O3 -Xptxas -v --fatbin -arch=sm_86 -no-compress -x cu tgcu_builtin.cu` (an earlier try without `-Xptxas -v` differed only in the command line stored in the fatbin header, not in the path) and `ptxas -arch=sm_86 -c tgcu_libc.ptx` |
| `libcudart.dylib` shim compiled from `shim/tgcudart.cpp` + `tgcublas.cpp` | compiles; exported symbols identical to the deployed one (136); 121,488 B vs 138,944 B (build flags not recorded; about 15 flag sets tried, none matched the size). The source is the `hs2` snapshot of `tgcudart.cpp`: the copy left in the mini's `~/shim` predates the pre-armed handshake and lacks six exported `tgcuHs*` symbols the deployed dylib has |
| `strata-vision-metal` built from the public tag + `patches/vision` + llama.cpp `3cf03257f` (38 s) | builds; same size as the deployed binary (8,077,952 B), sha256 differs (signature); **not run** against the mmproj |
| full engine rebuild (CMake configure in the container, `macbuild.py`, ninja) | device half re-run in the Docker container, host half on a driver Mac (not the mini); never run: see [reproduce.md section 10](reproduce.md#10-one-click-what-was-run). Redone for config C on the mini itself: section 8b |
| Docker build bundle (`docker compose build --no-cache build`, 3 min 4 s, then `docker compose run --rm build`) | EXL3 fatbins + `tgcu_builtin.fatbin` + `tgcu_libc.cubin` byte-identical to the deployed files; Strata engine device code: 63 fatbins, 59 byte-identical to the deployed ones in every one of four builds, 4 vary run to run (nvcc anonymous-namespace ids) |
| `scripts/test-gates.sh` | ALL GATES OK: wrong hash, corrupted byte, unknown command and a failing `verify-card-free.sh` step all exit nonzero (the old `pack_td.py verify` printed `VERIFY FAILED` and exited 0) |
| `./run.sh --check` on a driver Mac (empty install root) | 53 ok, 13 warnings, 0 failures; engine host build with LLVM 22 only (Apple clang 17 fails) |

### 8b. Config C: rebuilt from the repo's patches and compared with the deployed files (2026-10-15)

Full record: [`results/verification/config-c-repro.md`](../results/verification/config-c-repro.md). Patches `0001`-`0019` applied to a pristine `v0.1.41`, the container build, then `./run.sh --check` with a scratch install root on the mini (Apple clang 21, SDK 26.5; the live trees and the server untouched) and on the driver Mac.

| check | result |
|---|---|
| 20 source / shim / data files of config C (`config-c-sha256.txt`), `git apply` of 19 Strata + 5 shim patches | sha256 equal to the deployed sources (also `cmp`-ed against the mini's source trees); `patches/shim/base` + patches reproduce `tgcu.py`, `tgcu_api.py`, `tgcudart.cpp` |
| `libcudart.dylib` rebuilt on the mini (`shim/build-libcudart.sh`) | **byte-identical** (sha256 `20e8167d271b...`, md5 `ec0517ec`) |
| `libexl3cpu.dylib` on the mini / `libexl3metal.dylib` from `--build-metal` | **byte-identical** (`9fe9cab5491e...` / `e38a011c7993...`) |
| `libstrata_exl3.dylib` on the mini | identical except 48 bytes (`LC_UUID`, code signature) |
| Strata engine on the mini (scratch root `/tmp/recipe`, 135 host rules + link) vs the deployed `strata_d15a` (md5 `fc6e4ddd`) | all 9,372 symbols present with the same sizes; every data and text section identical except `__text` (70 of 1,111,789 words: `adrp`/`add` of globals that moved) and the embedded CUDA fatbins: 52 of 56 byte-identical; the other four (`kernels`, `fused_gr`, `sampler`, `moe_mmq`) are the fatbins nvcc names differently on every build (identical instruction streams except `sampler`: 3,970 of 16,416 differ in register allocation). **Not byte-identical as a file** (113,402,072 B vs 113,402,080 B) |
| device code in the container | 5 of 5 EXL3 / shim blobs byte-identical; 59 of 63 engine fatbins identical to `~/strata-mac` (the four above) |
| found and fixed | `docker/build-engine.sh` lacked `-DSTRATA_ISA_FLOOR=none` of the deployed CMake cache: without it `cpu_avxvnni_ok()` differed (56 vs 132 B) |
| `./run.sh --check` on the mini | 59 ok, 8 warnings (no Metal compiler, no models, no comgr under the scratch root), 0 failures; the new prerequisite section reads the mini's real wired limits (14336 / 15032385536) and `caffeinate` |
| `scripts/verify-card-free.sh patches`, `scripts/test-gates.sh` | pass |
| not reproduced | the bytes of four nvcc fatbins; anything needing the card (no number was re-measured); a model fetch |

## 9. Exact, rounding-level, approximate

- **Exact:** the weights (every routed expert of every layer: the checkpoint's trellis and scale tensors byte for byte, the three `mul1` scalars one common value (checked for all 73,728
  tensors); nothing pruned or masked); the GPU
  EXL3 kernels (bit-exact decode, GEMV/MoE within the gates in section 6); the NEON path; text parity between decode transport variants
  (`HS_PRE`, `HS_DEFER`, `MAC_PULL`, early pull: byte-identical outputs on 6/6 requests, ledger rows 2026-10-11T18:55Z to 20:50Z); the elastic K/V, the aligned reads and the staged refills move the same bytes to other places (their only effect on text is the fp32 association below).
- **Rounding-level:** with `STRATA_EXL3_METAL_SPLIT=1` (in every config since the split job) a layer's experts are summed as `y = A + B` in fp32, where the
  in-place/miss partition depends on the LRU contents and the slot count, so greedy text depends on LRU state and every change of the slot count (Decode7 .. Decode16) moves it: the Decode13 6,400 vs 6,200 slot arms matched md5 on 1 of 10 prompts (same-config reruns 10 of 10), config C vs its predecessor on 5 of 8 teacher-forced prompts; all coherent, same
  acceptance. The variant without `SPLIT` is bit-exact and ~8-11 % slower. The Mac prefill lane returns fp16-rounded partial sums
  (`STRATA_PF_MAC_Y16=1`); an earlier A/B had the fp16 lane diverge from the fp32 lane at character 127 of a 48-token comparison.
- **Approximate (config C):** `STRATA_PF_XPORT8=1` sends the lane's x and y rows as int8 + one fp16 scale per 32 values: rel RMS 0.55-0.65 % round trip on real MoE inputs, 0.80-0.83 % on the cold partial sum (Xport8 card-free tests, plan entry 2026-10-12T14:35Z). It is gated by the teacher-forced test of section 0.3 (top-1 98.01 %, mean KL 0.0041, p99 KL 0.058 against the
  stack with it off). The lossy `STRATA_MISS_TOL` is in the engine source and off.
- **Not measured:** KL or top-1 agreement against stock exllamav3 for any of these configs; GPQA Diamond; tool-calling benchmarks; concurrency above 1 with the shipped engine (section 0.4 measured the batch engine, which is slower); the host-tail config on the 190K needle
  (config C: found, 189,481 tokens); a vision run after the LRU change of the host-tail step as a raw log (config C: image PASS in `d15_final.log`, `newA.image.log`, `newB.image.log`).
