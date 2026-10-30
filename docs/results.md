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

## 0.000000000 Latest configuration: `configs/strata-exl3-full512-lq.json.tmpl` (config LQ, chain82-84, 2026-10-30)

Config **LQ** = config L without `0031`: config AM (section 0.0000000) + `STRATA_PF_MAC_QMIN_TOK=3000` (`patches/strata/0032`), engine `strata_o37j` (the config L build,
patches 0001-0032, sources `results/verification/config-l-sha256.txt`) with `STRATA_KVT_LAZY` unset, so `0031` does nothing. Production on the mini since 2026-10-30T18:52Z.
Prompts under 3000 tokens read only the experts they route to, on demand at each layer, instead of the three-layers-ahead read-ahead (section 0.00000000, "0032").
After a long prompt the trimmed K/V's slots refill before the next request starts, as in AM, which is what keeps LQ inside the quality gate where L fails (0.00000000.5).

**Source of these numbers:** the campaign log (chain82, chain83, chain84; 2026-10-30), transcribed; raw logs (`q21_lq81.log`, `q21_lq.log`, `abba_o83_*`, `final83.log`,
`soak83.log`, `d84_*.jsonl`, `chain84.summary`) are on the mini and not yet in `results/raw/`.

### 0.000000000.1 Quality

| check | LQ | AM | L |
|---|---|---|---|
| `q23.sh`, fresh restart: top-1 / mean KL / p99 KL | 98.01 % / 0.006181 / 0.0955 PASS | 98.16 % / 0.006144 / 0.0955 PASS | 98.01 % / 0.006181 / 0.0955 PASS |
| `q81_lp.py`, each panel prompt right after a fresh 33K prompt | 97.45 % / 0.005850 / **0.0655 PASS** | 0.0779, 0.0527 PASS (2 runs) | **0.1063, 0.1123 FAIL** (2 runs) |

### 0.000000000.2 ABBA83 (AM LQ LQ AM, fresh restart per arm; prefill tok/s run1 / run2)

| arm | 2.5K | 8K | 18K | 33K | decode realistic | TTFT | probes | agent turns (`small_sum`) |
|---|---|---|---|---|---|---|---|---|
| Aa (AM, `strata_o37f`) | 227.5/253.7 | 601.7/710.5 | 716.0/786.7 | 832.3/952.5 | 33.33 | 14.76 | 29.04 | 54.32 s |
| Qa (**LQ**, `strata_o37j`) | 224.9/251.4 | 605.3/708.3 | 718.4/787.7 | 830.9/953.5 | 33.50 | 14.76 | 29.07 | 51.66 s |
| Qb (**LQ**) | 226.6/253.6 | 604.0/710.4 | 715.2/786.6 | 832.0/946.8 | 32.68 | 14.74 | 29.05 | 51.62 s |
| Ab (AM) | 222.8/251.2 | 602.1/709.2 | 715.6/787.6 | 833.4/951.7 | 33.67 | 14.72 | 29.15 | 53.90 s |
| **LQ vs AM (mean of 2 arms, run 2)** | 252.5 vs 252.5 | 709.4 vs 709.9 | 787.2 vs 787.2 | 950.2 vs 952.1 | 33.09 vs 33.50 | 14.75 vs 14.74 | 29.06 vs 29.10 | **51.64 vs 54.11 s (-4.6 %)** |

The 1000-token prompt is not in this ABBA; `STRATA_PF_MAC_QMIN_TOK=3000` alone took it from 8.51 to 7.19 s (+18 %, section 0.00000000.1). 0 GPU timeouts, 0 fit fallbacks.

### 0.000000000.3 Decode ABBA84 (AM LQ LQ AM, fresh restart per arm, the 29-prompt realistic set twice per arm)

| arm | pass 1 tok/s (TTFT) | pass 2 tok/s (TTFT) |
|---|---|---|
| Aa (AM) | 33.42 (14.74) | 33.10 (14.35) |
| Qa (**LQ**) | 33.05 (14.74) | 33.20 (14.36) |
| Qb (**LQ**) | 33.58 (14.73) | 33.86 (14.37) |
| Ab (AM) | 32.89 (14.77) | 33.08 (14.43) |
| **LQ vs AM** | **33.42 vs 33.12 tok/s**; paired per prompt: mean +0.90 %, median +1.01 %, LQ slower on 10 of 29 | |

ABBA83's decode column (-1.2 %) and the regression's realistic set (33.27 vs AM's 33.71 in chain57) had pointed the other way; with 4 passes per config there is no decode cost.
0 GPU timeouts.

### 0.000000000.4 Regression and soak (`final21.sh` + 14-prompt soak, chain83)

| check | LQ | AM (chain57) |
|---|---|---|
| realistic decode, 29 requests | 33.27 tok/s, TTFT 14.73 s | 33.71, 14.76 |
| tool reuse / image | PASS / PASS | PASS / PASS |
| 190K needle (depth 50 %) | FOUND, 256 s | FOUND, 257 s |
| 5 x 32K needles | FOUND 51/48/32/37/36 s | FOUND 51/48/32/37/36 s |
| soak: 14 x 32K + 190K | 14/14 FOUND (21-63 s), 190K FOUND 232 s, engine alive | 14/14, 231 s |
| decode probe after the soak (mean of 6) | 28.7 tok/s | 28.8 |
| swap after prompt 14 (engine MALLOC_LARGE swapped) | 1502 MB (588 MB) | 1462 MB (587 MB) |
| GPU timeouts | 0 | 0 |

## 0.00000000 Withdrawn as the default: `configs/strata-exl3-full512-l.json.tmpl` (config L, Opt73, 2026-10-30)

**Withdrawn as the default on 2026-10-30 (13:52Z; the mini went back to config AM, then to config LQ at 18:52Z, section 0.000000000): requests that follow a long prompt fail the quality gate under L (section 0.00000000.5).**

Config **L** = config AM (section 0.0000000) + `STRATA_KVT_LAZY=0` (`patches/strata/0031`) + `STRATA_PF_MAC_QMIN_TOK=3000` (`patches/strata/0032`), engine `strata_o37j`,
production on the mini 2026-10-30T11:49Z-13:52Z. Both knobs are inert unless set; nothing else changes (no weight, context, vision, MTP or transport change).
- **0031, lazy K/V-trim refill.** A long prompt grows the elastic K/V into the expert cache; the next request trims it back and, in AM, copies every freed slot's expert
  back to the GPU before that request starts (747-792 slots after a 128K prompt, ~7 s). With 0031 the freed slots join the D18 pending refill that the prompt loan's
  slots already use (idle time, decode pieces); until they are back those experts are computed by the Mac's Metal lane, as for the loan's pending slots.
- **0032, read-ahead threshold.** For chunks of >= 512 tokens the Mac lane queues the reads of every expert the GPU does not hold, three layers ahead. A 1000-token chunk
  routes to ~219 such experts per layer, about half of them already in the Mac's RAM LRU, but the read-ahead read all ~223 the LRU lacked (9,830 blobs, ~18 GB per prompt,
  on an SSD that tops out at ~2.3 GB/s for any thread count). Under 3000 prompt tokens L reads only the routed experts, on demand at each layer.

**Source of these numbers:** the campaign log (plan entries 2026-10-30T08:40Z .. 13:50Z, Opt63 .. Opt74), transcribed; raw logs (`kvt63_*`, `q72_*` rows, `abba_o73_*`,
`final74.log`, `soak74.log`, `q21_dd.log`, `q21_kv0.log`) are on the mini and not yet in `results/raw/`.

### 0.00000000.1 What each change does (single arms, fresh restart per arm)

| measurement | AM | L's change | arm |
|---|---|---|---|
| first 94-token request of another conversation right after a 128K prompt: prompt time (2 reps) | 11.5 / 12.2 s (11.0 / 11.65 s in the closing AM arm) | **4.4 / 4.6 s** | `STRATA_KVT_LAZY=0` (300: 7.0 / 7.4 s) |
| ... that request's decode / the next request's decode | 32.0 / 33.0 tok/s | 30.7 / 30.2 tok/s (-4 % / -7 %, until the idle refill: 7.6-10 s) | same |
| 1000-token prompt (2 runs) | 8.54 / 8.48 s (117.6 tok/s) | **7.18 / 7.20 s (139.2 tok/s, +18 %)** | `STRATA_PF_MAC_QMIN_TOK=3000` |
| 2.5K / 400 / 8K prompt | 10.17 / 4.52 / 10.87 s | 10.04 (+1.2 %) / 4.49 / 10.86 s | same; 2048 instead of 3000: 2.5K unchanged, 1000 tok 7.28 s |

0 GPU timeouts in every arm. Reader threads (`STRATA_PF_MAC_THREADS` 4 / 16 / 32 vs 8) change nothing: the SSD is the limit for 2.5K-8K prompts (~19 GB of expert reads per lockstep group pass).

### 0.00000000.2 ABBA73 (AM L L AM, fresh restart per arm; prefill tok/s run1 / run2)

| arm | 2.5K | 8K | 18K | 33K | decode realistic | TTFT | probes | agent turns (`small_sum`) |
|---|---|---|---|---|---|---|---|---|
| Aa (AM, `strata_o37f`) | 223.0/251.7 | 599.0/710.0 | 716.5/787.2 | 830.9/952.8 | 33.48 | 14.76 | 29.08 | 54.36 s |
| La (**L**, `strata_o37j`) | 224.9/250.3 | 596.9/714.7 | 714.9/787.5 | 829.8/949.7 | 33.09 | 14.74 | 28.72 | 52.23 s |
| Lb (**L**) | 228.4/252.3 | 600.7/708.7 | 717.4/787.1 | 830.2/952.5 | 33.30 | 14.70 | 28.95 | 52.60 s |
| Ab (AM) | 226.7/251.5 | 602.3/710.5 | 715.4/788.0 | 832.4/949.8 | 33.23 | 14.74 | 29.30 | 54.11 s |
| **L vs AM (mean of 2 arms)** | 251.3 vs 251.6 | 711.7 vs 710.3 | 787.3 vs 787.6 | 951.1 vs 951.3 | **33.20 vs 33.36 (-0.5 %)** | 14.72 vs 14.75 | **28.84 vs 29.19 (-1.2 %)** | **52.42 vs 54.24 s (-3.4 %)** |

The standard rows are unchanged (the ABBA has no 1000-token row). Agent turns (short requests after the 33K rows) are 3.4 % faster; decode is within AM's own spread
(the two AM arms differ by 0.75 %); the probes, run right after the long rows, are 1.2 % slower while the trimmed slots still wait for the idle refill. 0 GPU timeouts, 0 fit fallbacks.

### 0.00000000.3 Quality

705-position teacher-forced gate (`q23.sh`, fresh restart) vs p-off: **L top-1 98.01 %, mean KL 0.006181, p99 0.0955 - PASS** (after a long prompt L fails: section 0.00000000.5) (AM 98.16 % / 0.006144 / 0.0955); `STRATA_KVT_LAZY=0` alone 98.16 % / 0.006072 / 0.0955.
Rejected on this gate the same day: `STRATA_D18_K=200` (fewer slots refilled before the first token; single arms 2.5K +7 %): p99 0.154, FAIL - more experts on the Metal lane in the panel's decode.

### 0.00000000.4 Regression and soak (`final21.sh` + 14-prompt soak, chain74)

| check | L | AM (chain57) |
|---|---|---|
| realistic decode, 29 requests | 33.17 tok/s, TTFT 14.71 s | 33.71, 14.76 |
| tool reuse / image | PASS / PASS | PASS / PASS |
| 190K needle (depth 50 %) | FOUND, 256 s | FOUND, 257 s |
| 5 x 32K needles | FOUND 53/48/31/37/34 s | FOUND 51/48/32/37/36 s |
| soak: 14 x 32K + 190K | 14/14 FOUND (22-56 s), 190K FOUND 232 s, engine alive | 14/14, 231 s |
| decode probe after the soak (mean of 6) | 27.6 tok/s | 28.8 |
| swap after prompt 14 (engine MALLOC_LARGE swapped) | 1286 MB (674 MB) | 1462 MB (587 MB) |
| GPU timeouts | 0 | 0 |

### 0.00000000.5 Quality after a long prompt (chain81 / chain82): why L was withdrawn

The quality gate (`q23.sh`, 705 teacher-forced positions vs p-off) restarts the server and scores the panel from that fresh state. A server in use is often in another state:
the request before was a long prompt, so the K/V has just been trimmed back and the expert slots the prompt path borrowed are still being refilled. `q81_lp.py` scores
the same panel with a fresh ~33K-token prompt of another conversation sent before each of the 8 panel prompts (same lpcmp vs p-off, same thresholds).

| config | panel state | top-1 | mean KL | p99 KL | gate (p99 < 0.1) |
|---|---|---|---|---|---|
| AM | fresh restart (section 0.0000000) | 98.16 % | 0.006144 | 0.0955 | PASS |
| L | fresh restart (0.00000000.3) | 98.01 % | 0.006181 | 0.0955 | PASS |
| AM | after a 33K prompt, run 1 / run 2 | 97.59 / 98.30 % | 0.004726 / 0.005848 | **0.0779 / 0.0527** | PASS / PASS |
| L | after a 33K prompt, run 1 / run 2 | 97.45 / 98.01 % | 0.008550 / 0.007063 | **0.1063 / 0.1123** | **FAIL / FAIL** |
| AM + `0032` only (`STRATA_PF_MAC_QMIN_TOK=3000`, no `0031`) | after a 33K prompt | 97.45 % | 0.005850 | 0.0655 | PASS |

The cause is `0031` (`STRATA_KVT_LAZY=0`): after a long prompt the expert slots the trimmed K/V gives back are left to the idle refill, so the next requests decode
those experts on the Mac's Metal lane (other rounding) on top of the slots the prompt's own loan left pending; AM refills them before the next request starts.
`0032` alone (a change to which blobs are read, not to what is computed) passes in that state. 0 GPU timeouts in every arm. From now on a candidate runs this check
as well as `q23.sh` before it becomes the default.

## 0.0000000 Previous configuration (the default again 2026-10-30 13:52Z-18:52Z): `configs/strata-exl3-full512-am.json.tmpl` (config AM, Opt57, 2026-10-30)

Config **AM** = config S2 (section 0.000000) + `STRATA_PF_ALIGN=131072` with `STRATA_PF_ALIGN_MAXT=2048` (`patches/strata/0030`, engine `strata_o37f`), production on the mini since 2026-10-30T05:10Z.
The prompt path borrows GPU expert-cache slots for its buffers (1,862,400 B each, not page-aligned), so every buffer's page offset - and the shim's 256 KiB lazy-download rounds - moved with
the loan's first slot. AM starts every borrowed buffer of >= 2 MiB on a 128 KiB boundary for 2048-token chunks (prompts < 28K tokens); the 3072-token chunks of >= 28K prompts keep S2's layout.
Nothing else changes (no decode, weight, context, vision or MTP change).

**Source of these numbers:** the campaign log (plan entries 2026-10-30T00:30Z .. 05:30Z, Opt52 .. Opt57), transcribed; raw logs (`abba_o55_*`, `final57.log`, `soak57.log`, `q21_am.log`) are on the mini and not yet in `results/raw/`.

### 0.0000000.1 Prefill rows, ABBA55 (S AM AM S, fresh restart per arm; tok/s, run1 / run2)

| arm | 2.5K | 8K | 18K | 33K |
|---|---|---|---|---|
| Sa (S2, `strata_o37a`) | 227.6/250.7 | 598.2/703.9 | 697.7/765.7 | 831.2/949.8 |
| Aa (**AM**, `strata_o37f`) | 224.8/250.9 | 604.3/710.2 | 714.7/788.1 | 832.0/947.2 |
| Ab (**AM**) | 227.3/253.6 | 596.0/709.8 | 713.9/787.5 | 833.8/948.1 |
| Sb (S2) | 227.4/250.5 | 599.3/703.7 | 699.1/768.4 | 831.0/951.9 |
| **AM vs S2, warm (mean of 2 arms)** | **252.3 vs 250.6 (+0.7 %)** | **710.0 vs 703.8 (+0.9 %)** | **787.8 vs 767.1 (+2.7 %)** | **947.7 vs 950.9 (-0.3 %)** |

33K: within S2's own spread (ten S2 arms on 2026-10-29/30 at 33K warm: mean 951.2, sd ~2); the 3072-token chunks are laid out as in S2. 0 GPU timeouts, 0 fit fallbacks in every arm.
Live check right after the promotion (fresh restart, then the rows; not comparable with ABBA arms, which run 29 decode requests and probes first - fresh-restart rows are higher for every config, S2 18K ~793): 2.5K 220.2/248.2, 8K 604.9/734.3, 18K 750.6/791.7, 33K 831.1/962.4.
Before it (same day): ALIGN 64 KiB on every chunk 33K -0.6 %; ALIGN 128 KiB on every chunk 8K +0.8 %, 18K +2.4 %, 33K -0.4 % (ABBA52) -> MAXT 2048.

### 0.0000000.2 Decode, TTFT, quality

| arm | decode realistic (tok/s) | TTFT (s) | probes (tok/s) |
|---|---|---|---|
| Sa / Aa / Ab / Sb | 33.55 / 33.40 / 33.32 / 33.26 | 14.77 / 14.73 / 14.74 / 14.81 | 29.28 / 28.88 / 28.84 / 28.87 |

Quality (705-position teacher-forced gate, `q23.sh`): AM vs p-off top-1 98.16 %, mean KL 0.006144, p99 0.0955 - PASS, the same as S2 vs p-off (98.16 % / 0.006096 / 0.0955); AM vs S2 top-1 99.72 %, mean KL 0.000135.
The p99 of 0.0955 is near the 0.1 gate for every 5-bit-transport config (S2 and `xb5` give the same value); it comes from the transport, not from 0030.

### 0.0000000.3 Regression and soak (`final21.sh` + 14-prompt soak, chain57)

| check | AM | S2 (chain42) |
|---|---|---|
| realistic decode, 29 requests | 33.71 tok/s, TTFT 14.76 s | 33.50, 14.80 |
| tool reuse / image | PASS / PASS | PASS / PASS |
| 190K needle (depth 50 %) | FOUND, 257 s | FOUND, 255 s |
| 5 x 32K needles | FOUND 51/48/32/37/36 s | FOUND 51/49/32/38/36 s |
| soak: 14 x 32K + 190K | 14/14 FOUND, 190K FOUND 231 s, engine alive | 14/14, 230 s |
| decode probe after the soak (mean of 6) | 28.8 tok/s | 28.3 |
| swap after prompt 14 (engine MALLOC_LARGE swapped) | 1462 MB (587 MB) | 1282 MB (587 MB) |
| GPU timeouts | 0 | 0 |

### 0.0000000.4 What did not work (2026-10-29/30, all on S2)

| experiment | result |
|---|---|
| `STRATA_PF_YRING` (two y landing buffers) | no effect (exact; the loan size it needed was the only difference) |
| `STRATA_D18_K=0` (no refill before the first token) | decode -4.3 % |
| `STRATA_PF_MAC_DEMOTE` 1-in-8 / 1-in-4 (GPU-resident experts moved to the Mac during prefill) | 33K -2.7 % / -3.1 %, 18K -5.1 % / -5.5 %: the 3090/Mac split is balanced at 33K |
| lockstep group 7 / 8 (`STRATA_PF_GROUP` + `STRATA_EXL3_METAL_JOBS`) | 7 = S2; 8: 33K -1.4 %, decode -2 %, more swap, fails the p-off gate (p99 0.131) |

## 0.000000 Previous configuration: `configs/strata-exl3-full512-s2.json.tmpl` (config S2, Opt42, 2026-10-29)

Lineage since config G: **H** (section 0.00000: G + the idle-only SSD-parked conversation cache, `patches/strata/0025`, engine `strata_o24b`, `--ssd-park-mib 8192`) -> **P35** (2026-10-29T13:25Z) = H + `--prefill 3072` with `STRATA_PF_BIG_MIN=28000` (Prefill25 / Opt26 incl. the `lend()` cap, `0026`) + `STRATA_PF_GRP_FIT=1` (Opt29, `0028`) + the 5-bit lane transport `STRATA_PF_XPORT_BITS=5` (Opt28, `0027`), engine `strata_o29a` md5 `29cd1bbe`, live config md5 `60d3be11` -> **S2** (2026-10-29T16:02Z) = P35 + `STRATA_PF_SPLIT2=1` (Opt37, `0029`), engine `strata_o37a`, live config md5 `6e897d7a687fcf161e576dcd8bc40084`. Every new knob is inert unless set; decode, the weights (3.05 bpw, all 512 experts, no pruning), 200K context, vision and MTP are unchanged.
**Source of these numbers:** the campaign log (plan entries 2026-10-29T03:40Z .. 16:05Z, Opt26 .. Opt42), transcribed. Unlike the sections below, the raw logs (`abba_o26_*`, `abba_o31_*`, `abba_o32_*`, `abba_o40_*`, `final35.log`, `final42.log`, `soak37.log`, `soak42.log`, the `q21_*` / `xq30` quality logs, `pf*33_33000.*` profiles) are still on the mini and are **not yet in this repo**. Config H's own measurements against G are in section 0.00000 (raw: `results/raw/opt24/`); below, H is the control arm of the 10-29 ABBAs.

Method: as above; every ABBA arm starts from a fresh server restart (`arm23.sh`); a row is the mean of the two arms of a config; "run1 / run2" = the first (cold) and second (warm) request of the same prompt size; prefill prompt sizes 2,565 / 8,018 / ~18K / ~33K tokens.

### 0.000000.1 Prefill rows, ABBA (tok/s, run1 / run2)

| ABBA | arms | 2.5K | 8K | 18K | 33K |
|---|---|---|---|---|---|
| ABBA26 (H I26 I26 H) | H | 230.9/261.2, 225.9/252.6 | 566.4/652.3, 565.4/651.8 | 672.6/734.6, 671.6/733.4 | 731.8/795.6, 731.9/795.7 |
| | I26 (H + `--prefill 3072`, BIG_MIN 28000, `strata_o26a`) | 229.9/258.9, 230.1/257.2 | 556.6/638.1, 556.4/639.6 (-1.7/-2.1 %) | 666.8/726.9, 665.5/728.3 (-0.9 %) | 756.9/858.3, 760.0/861.1 (+3.6/+8.0 %) |
| ABBA31 (H XB XM XM XB H) | H | 225.4/250.6 | 566.0/653.0 | 672.2/734.5 | 732.3/792.9 |
| | XB (H + 5-bit, `strata_o28b`) | 227.0/252.0 (+0.7/+0.5 %) | 585.0/682.8 (+3.4/+4.6 %) | 688.8/755.8 (+2.5/+2.9 %) | 764.3/831.6 (+4.4/+4.9 %) |
| | XM (XB + M27 Metal lib) | - | 683.5/683.2 (warm, per arm) | - | 832.3/833.8 (warm, per arm): = XB |
| ABBA32 (H IG CX CX IG H) | H | 225.8/251.0 | 561.8/654.9 | 672.4/735.5 | 731.9/794.5 |
| | IG (I26 + GRP_FIT, `strata_o29a`) | 225.9/251.3 | 569.0/661.5 (+1.3/+1.0 %) | 672.5/734.0 (0/-0.2 %) | 759.8/861.2 (+3.8/+8.4 %) |
| | CX (IG + 5-bit + M27 lib; = P35 numerics + M27) | 225.8/251.3 | 592.0/689.7 (+5.4/+5.3 %) | 691.8/759.8 (+2.9/+3.3 %) | 815.4/931.9 (+11.4/+17.3 %) |
| ABBA40 (P S S P) | P35 | 225.4/249.9 | 587.1/686.0 | 687.8/756.0 | 811.5/928.5 |
| | **S2** (P35 + SPLIT2, `strata_o37a`) | 225.2/251.6 (0/+0.7 %) | 596.1/702.2 (+1.5/+2.4 %) | 697.8/767.5 (+1.4/+1.5 %) | 833.0/950.0 (+2.7/+2.3 %) |

ABBA40 per arm at 33K warm: Pa 928.1, Pb 928.9, Sa 947.9, Sb 952.0 (no overlap). Cumulative H -> S2 (warm): 8K 654.9 -> 702.2 (+7.2 %), 33K 794.5 -> 950.0 (+19.6 %); 2.5K unchanged. 0 GPU timeouts in every arm.

### 0.000000.2 Decode, TTFT, agent turns (same ABBAs)

| ABBA | config | decode realistic (tok/s, arm A / arm B) | TTFT (s) | agent-turn set `small_sum` (s) | probes (tok/s) |
|---|---|---|---|---|---|
| ABBA26 | H / I26 | 33.83/33.41 / 33.50/33.63 | 15.03/15.07 / 15.18/15.15 | 52.71/53.74 / 54.28/54.09 | noise |
| ABBA31 | H / XB | 33.38/33.31 / 33.47/33.35 | 15.13/15.12 / 14.90/14.91 | 53.99/53.90 / 53.64/53.53 | 28.88/29.33 / 28.65/29.31 |
| ABBA32 | H / IG / CX | 33.28/33.23 / 33.66/33.76 / 33.29/33.24 | 15.11/15.12 / 15.07/15.06 / 14.80/14.82 | 53.98/54.11 / 54.53/54.05 / 54.03/54.07 | 29.10/29.08 / 29.18/28.53 / 28.50/29.20 |
| ABBA40 | P35 / S2 | 33.57/32.87 / 33.44/33.22 | 14.89/14.90 / 14.81/14.81 | 54.23/54.35 / 54.01/54.00 | noise |

Decode is unchanged within noise along the whole lineage (the changes are prefill-only). IG's +0.55 s on the agent-turn set (ABBA32) is a refill debt: the agent turns run right after the 33K rows, and a prompt >= BIG_MIN lends 1,375 VRAM slots (H: 884), so ~1,075 slots are still pending refill when the next turns arrive; the long prompt itself saved ~3 s.

### 0.000000.3 Quality

Teacher-forced gate (705 positions, vs the saved baseline with the lane transport off, gate: top-1 >= 97 %, mean KL < 0.01, p99 KL < 0.1):

| arm | top-1 | mean KL | p99 KL | verdict |
|---|---|---|---|---|
| H (int8 transport, = G) | 97.73 % | 0.005474 | 0.081680 | PASS |
| x6 emulation (`STRATA_PF_XQ_EMU=6`, double quantisation) | 98.72 % | 0.004660 | 0.063791 | PASS |
| x5 emulation (`STRATA_PF_XQ_EMU=5`) | 97.73 % | 0.007146 | 0.087304 | PASS |
| xb5 (real 5-bit codec, `strata_o28b`) | 98.01 % | 0.006181 | 0.095506 | PASS, thin p99 margin |
| hg (H + `strata_o29a` + GRP_FIT) vs config F | 100.00 % | 0 | 0 | EXACT |
| cx (I26 + o29a + GRP_FIT + 5-bit + M27) | 98.16 % | 0.006072 | 0.095506 | PASS |
| p37n (P35 on `strata_o37a`, SPLIT2 off) | 98.01 % | 0.006181 | 0.095506 | identical to xb5 |

Because the 705-position p99 of xb5 sat at 0.0955, a wider panel was run before promotion (Opt33: 24 prompts = 8 kinds x 3 variants with new seeds and shifted excerpts, **2,139 scored positions**, reference = the fp16 lane `STRATA_PF_XPORT8=0`; qualification bar: mean KL <= 1.5 x int8's):

| transport | vs fp16: top-1 / mean KL / p99 KL | vs int8: top-1 / mean KL / p99 KL |
|---|---|---|
| int8 (H, P35's predecessor) | 98.50 % / 0.003590 / 0.055040 | - |
| 6-bit | 98.27 % / 0.003765 / 0.057354 | 98.55 % / 0.003698 / 0.044137 |
| **5-bit (P35, S2)** | **98.27 % / 0.004077 / 0.045874** (1.14 x int8 mean KL) | 98.55 % / 0.003938 / 0.056647 |
| X6/Y5 | 97.85 % / 0.004667 / 0.065392 | 98.50 % / 0.004084 / 0.040101 |

All pass; 5-bit qualifies (bar 0.005385) and has a lower p99 than int8 on 2,139 positions, so the 0.0955 on the narrow gate was a single-panel tail. X6/Y5 scoring worse than 5/5 shows the panel's noise floor (~+-0.0005 nats between widths of near-equal error).
S2 vs P35 (`lpcmp` p37n vs s2, Opt38): top-1 99.86 %, mean KL 0.000145, p99 0.002632. Per prompt: code-cpp (3,610 tokens), code-py, agentic (4,236), long-9K (9,542), chat, math, code-cu (3,552) bit-exact; docs (3,068 tokens) 98.96 % / 0.001063: it is cut into 2 chunks, below `STRATA_PF_SPLITF_MIN` 4, so the SPLIT2 order is never reached there and the drift is run-to-run residency (refill timing decides GPU vs Metal rounding). Every prompt that takes the SPLIT2 order is exact. (A gate run of S2 itself against the transport-off baseline is not recorded separately; S2's numerics are P35's.)
Note on cx: the M27 Metal library is bit-exact on the `pfx_test4` bench but **not** inside the engine (p37n vs cx 99.86 % / 0.000204 / 0.002723); it was never promoted.

### 0.000000.4 Regression and soak

| | P35 (`final35.log`, Opt35) | S2 (`final42.log`, Opt42) |
|---|---|---|
| realistic 29 prompts: tok/s / TTFT / wall | 32.94 / 14.89 s / 23.81 s | 33.50 / 14.80 s / 23.55 s |
| decode probes (mean) | 29.37 | 29.25 |
| deterministic probe md5s | repeat (8af670ad, f60e112b) | post-soak probes repeat (f60e112b x2) |
| tool reuse / image | PASS / PASS | PASS / PASS |
| prefill rows (run1/run2) | 2.5K 237.8/238.4, 8K 594.8/683.8, 17K 650.9/710.6, 33K 808.0/928.0 | (see ABBA40) |
| ~190K needle | FOUND, 260 s (config F 352 s) | FOUND, 255 s |
| 5 x 32K needles | FOUND, 51/49/32/39/37 s | FOUND, 51/49/32/38/36 s |
| GPU timeouts | 0 | 0 |
| swap (2 GB) at the end | 1.41 GB (798 MB -> 1,414 MB, step at the 190K needle) | 1.48 GB |
| soak (14 x 32K needles + 190K) | `soak37`: 14/14 FOUND (21-63 s), 190K FOUND 234 s, 0 GPU timeouts, wired 12.7-13.0 GB flat, swap 0.88 -> 1.40 GB then flat | `soak42`: 14/14 FOUND (22-63 s), 190K FOUND 230 s, 0 GPU timeouts, swap held 1.28 GB |

Both regressions and both soaks PASS. Watch item: ~0.5-0.7 GB of swap headroom left after a 190K prompt.

### 0.000000.5 Where the 33K period goes, and what did not work

- Opt34 (33K, `TGCU_PROF` + pf-mac stats, H vs XB5, per chunk-layer, 748 chunk-layers): front 32.7 -> 32.9 ms, x fetch 14.7 -> 9.5 (-35 %, the byte cut), y push 20.4 -> 19.1 (-6 % only), period 49.9 -> 45.7 (-8.4 %). The y push span includes waiting for the card behind the front, so the byte cut shows up mostly on x. The model that predicted +13-25 % for 5-bit assumed the link was the binding resource; measured +3-5 % (2048 chunks) and +8 % (3072 chunks).
- Opt41 (33K, 3072-token chunks, 528 chunk-layers): period P35 58.8 ms vs S2 57.1 (-2.9 %, = the ABBA40 +2.3 %); x fetch 13.5 / 13.5, y push 24.9 / 25.6, link x+y 38.4 / 39.1 ms (68 % of the period). ~12 ms of each y push is a wait on the card through the single y landing buffer (present with and without SPLIT2).
- **SPLIT2 was modelled at +15-25 % (card front of chunk i+1 under chunk i's x download) and measured +2.3 %**: the period is host-serial (y push ~19 ms + x fetch 9.5 ms + ~17 ms host work), not card-bound. Kept because it is exact and a small real win.
- **M27 Metal kernel (super-chunk GEMM + aliased epilogue tile): neutral twice** (ABBA27 H M27 M27 H: 33K 794 vs 793 warm; ABBA31 XM vs XB: equal), although -22.6 % on the M1 Max bench and -4.5..-6.8 % on the M4 bench: Metal is ~66 % busy and not the binding term in situ. Also not bit-exact inside the engine (above). Not shipped; its library is not in this repo.
- Prefill25 alone (config I, 10-20: `--prefill 3072` + BIG_MIN without the table fit) lost 2-3 % on 8K/19.5K; the Opt26 `lend()` fix did not change that (I26 = I within 0.3 %); the cause was the mapped routing-group table sized for 3072-token chunks and pulled whole by the shim after every synchronize (+1.0 ms per chunk-layer at 8K), fixed by `STRATA_PF_GRP_FIT=1`.
- `STRATA_PREFILL_TIMING=1` is not faithful over tgcu (its per-phase events serialise the stream over USB: 33K wall 95 s vs ~42 s); only its shares were used.
- Next experiment (not part of S2): `STRATA_PF_YRING` (two y landing buffers, engine `strata_o37b`) against the single-buffer y push wait.

## 0.00000 Previous configuration: `configs/strata-exl3-full512-h.json.tmpl` (config H, Opt24, 2026-10-19)

= config G (section 0.0000) + patch `strata/0025` (SSD-parked conversation cache, `--ssd-park-mib 8192 --ssd-park-dir`; see [`park-cache.md`](park-cache.md)). Engine `strata_o24b` md5 `b3a275ad2feb61495f019e30ac104d41`, live config md5 `fa0860bc9774632d0b3f8093d5b3a720`; all other artifacts are config G's. Raw: [`results/raw/opt24/`](../results/raw/opt24/); hashes: [`results/verification/config-h-sha256.txt`](../results/verification/config-h-sha256.txt). Note: the patch was generated from the Opt24 sources against the v0.1.41 + `0001`..`0024` tree, which reconstructs the live G sources exactly (md5 `dd67d1a1` for `generate.cpp`); applying it reproduces the sources of `strata_o24b` (md5 `80e56c46`, `3ae62b93`). The engine binary itself was built on the mini with `build24.sh`, not re-built from the repo.
Switch-back (A1 cold, pause 15 s, B1 other document, A2 = A1's prefix + a new question; `results/raw/opt24/logs/switch/`):

| prompt | A1 cold | B1 | A2 with park (H) | A2 without park (full re-prefill) | A2 text md5 |
|---|---|---|---|---|---|
| 30K | 49.1 s | 45.2 s | **3.08 s** (read 330 ms, restore 1.47 s, 24 tokens prefilled) | 41-49 s | `7cd8fb6e` (= no-switch control) |
| 100K | 132.9 s | 146.5 s | **5.95 s** (read 823 ms, restore 4.36 s, 27 tokens prefilled) | 133-146 s | `7cd8fb6e` |

A shorter pause (4 s): the idle write of A was aborted by B1, A2 re-prefilled in full (41.7 s) - idle-only trades parking of very quick switches for zero cost on the request path. The earlier variant H0 (`strata_o24a`, outgoing conversation saved synchronously on the request path) restored A2 in 5.2 s at 30K but paid 2-5 s per switch and raised the realistic TTFT (16.05 s vs 15.09 s, `abba_o24_H0a/b` vs `abba_o24_G`), so it was not promoted.
ABBA (G H1 H1 G, fresh restart per arm, 29 realistic prompts, 300 tokens + 12 probes + prefill rows x2; `abba/abba_o24_*`): decode (mean tok/s over the 29 requests) G 33.39 / 33.61 (mean 33.50), H 33.55 / 33.39 (mean 33.47): neutral; TTFT 15.09 s both; probe means G 28.89 / 29.41, H 29.62 / 29.32; prefill (tok/s, run 1 / run 2) 2,565 tokens G 228.6 / 252.4, H 231.1 / 256.0; 8,018 566.6 / 654.2 vs 567.2 / 653.6; 19,547 672.7 / 736.0 vs 672.3 / 735.6; 33,842 733.7 / 796.9 vs 732.9 / 796.0; 0 GPU timeouts, no foreign requests.
Regression (`reg24_h1.log` vs `d23_final_G.log`; realistic 29, probes, prefill rows x2, tool reuse, image, 190K needle, 5 x 32K): H realistic 33.51 tok/s vs G 33.43, TTFT 15.06 vs 15.08 s; tool reuse PASS (3,379 tokens reused) both; image PASS both; 190K needle FOUND 273 s vs 276 s; 5 x 32K FOUND 52/49/32/41/37 s vs 53/49/33/41/38 s; prefill 240/236, 564/648, 637/693, 742/793 vs 236/233, 564/650, 635/692, 741/795 tok/s; 27/29 realistic outputs md5-identical to G (p19 and p08 differ at fp level on the MTP/draft path; G itself varies run to run on the probes). Quality (teacher-forced, 705 positions, `q24_h1.log`): top-1 100.00 %, mean KL 0, p99 KL 0 (PASS, gate < 0.1). Guarded 5 x 32K soak (`soak24_h1.log`): 5/5 found, engine alive, 0 GPU timeouts.
Not covered: parking is lost on a server restart; a switch inside ~1.5 s + the write time is not parked (above); one park cache entry per conversation, 8 GiB bound (about 400K tokens of 19 KiB each).

## 0.0000 Previous configuration: `configs/strata-exl3-full512-g.json.tmpl` (config G, Opt23, 2026-10-18)

= config F (section 0.000) + patch `strata/0024` (`STRATA_O23=3`: the slim doorbell publishes into per-ring payload slots, x is pulled as fp16, and the device skips the wait for the mini's rows when no routed expert of the ring is outside VRAM; kernels `exl3/exl3_o23.cu` -> `libstrata_o23.dylib`, `STRATA_O23_LIB`) + the fused-decode Metal prefill GEMM (`EXL3M_PF_FD=2`, `exl3/metal`: the EXL3 decode happens in the GEMM's B-tile load, no 1.9 GB decoded-weight round trip through DRAM). Both are bit-exact: the Metal library against the previous one (39/39 `cmplibs` cases and 7/7 zipf-routed cases bit-identical), the doorbell against F (teacher-forced log-probabilities identical, below). Live config md5 `ff78909a73ed45f5984c7156e59adb2f`, engine `strata_o23b` md5 `05ec22e54048b896f83372eb7c430feb`, libraries `libstrata_o23.dylib` md5 `6a4e09a7`, `libexl3metal_g.dylib` md5 `c49abac8` (sha256 `53d4deab...`, rebuilt byte-identical from `exl3/metal/build.sh`). Raw: [`results/raw/opt23/`](../results/raw/opt23/); hashes: [`results/verification/config-g-sha256.txt`](../results/verification/config-g-sha256.txt), rebuild comparison: [`results/verification/config-g-repro.md`](../results/verification/config-g-repro.md).
Profile of F (decode, `HS_TRACE` + `HSPROF`, 79,680 rings), per cold ring in ms: wait-for-ring 0.304, x pull copy 0.033, mini pool (Metal job A/B + SSD reads) 1.069, y push 0.248, flag 0.021; the 5 % zero-cold rings paid a 0.470 tail; 44 rings x ~1.6 ms are ~71 of ~76 ms per window. Prefill Metal job microbench (192 experts, 27 rows each, previous library): 52.4 ms = decode 26.4 (DRAM-write bound) + GEMM 23.4 (re-reads the decoded weights) + 1.8; fused decode: 27 ms (27 rows/expert), 45 (53 rows), 83 (107 rows).
First screen (ABBA F O O F, 10 prompts + 3 probes): zero-cold skip + fp16 x pull +2.63 % decode (paired 9/10), ring wait 0.305 -> 0.271 ms, zero-cold tail 0.470 -> 0.307 ms.
ABBA (F G G F, fresh restart per arm; `abba_summary.txt`): decode realistic F 32.64, G 33.38 tok/s (**+2.33 %**, paired 26/29 prompts faster, se 0.51 %); probes F 28.19, G 28.91 (+2.56 %); prefill (4 runs each, tok/s) 2,565 tokens F 220 247 222 247 (234), G 228 252 228 250 (240); 8,018 F 465 527 467 528 (497), G 562 651 561 651 (606); 19,547 F 543 581 544 580 (562), G 671 735 672 734 (703); 33,842 F 601 640 601 642 (621), G 732 790 733 792 (762) (+2.6 / +22 / +25 / +22.7 %). Agent turns (new tokens: wall s, F / G, mean of 2): 60: 3.64 / 3.09, 100: 4.69 / 3.99, 150: 5.29 / 4.54, 200: 5.71 / 4.83, 300: 6.84 / 5.87, 400: 6.94 / 5.92, 600: 9.17 / 9.04, 1000: 9.31 / 9.30.
Regression (final config, 75 requests; `d23_final.log`): realistic 33.43 tok/s over 29 prompts (TTFT 15.1 s, F 19.7 s), probes (tok/s, 12 requests) 27.4 / 33.4 / 23.3, 27.8 / 33.3 / 25.9, 27.7 / 33.1 / 25.4, 28.3 / 33.9 / 25.1; prefill 235.8 / 233.0, 563.9 / 650.3, 635.3 / 691.6, 740.9 / 795.0 tok/s (2,565 / 8,018 / 18,330 / 33,842 tokens, 2 runs each); tool reuse PASS (3,379 tokens reused), image PASS, ~190K needle FOUND (189,450 tokens, 276 s; F 352 s), 5 x 32K needles FOUND (33-53 s), 0 GPU timeouts. Quality (teacher-forced, 705 positions, `q21_G23.log`) vs the saved baseline: top-1 97.73 %, mean KL 0.005474, p99 KL 0.081680 (PASS, gate < 0.1); vs config F and vs the E-era log-probabilities: top-1 100 %, KL 0 (identical). Guarded soak (5 successive ~33K needles): 5/5 found, 0 GPU timeouts, no foreign request in the engine log (`soak23.out`).
Not shipped, measured in the same campaign: ring trims (arm / hold-poke are hidden behind the GPU chain: not a lever); fp16 y push (`STRATA_O23` bit 2, rounding-level, built: wait 0.271 -> 0.266 ms, +4.24 % vs F in a single 10-prompt arm, noise-level next to the +2.63 % of the exact path, so left out); 3-chunk Metal jobs (prefill is host-bound at 8K+ after the fused decode: not measured).

## 0.000 Latest configuration: `configs/strata-exl3-full512-f.json.tmpl` (config F, Prefill22, 2026-10-17)

= config E (section 0.00) + patches `strata/0022`, `0023` + `shim/0006` + `shim/tgusb.c` (`libtgusb.dylib`) with `TGCU_USB_NPIPE=1`, `TGCU_USB_SUBP=1`, `TGCU_USB_FUSE=1`, `TGCU_USB_FENCE=11`. Live config md5 `b8d72f718d1101508ccd5fbc15255b76`, engine `strata_p22` (= `strata_p21b`) md5 `4180585c847520eadb3cecd70a17bc34`, shim dir with `tgcu.py` md5 `1d8a4179`, `libcudart.dylib` md5 `00771147`, `libtgusb.dylib` md5 `7e3e04fd`. Raw: [`results/raw/prefill22/`](../results/raw/prefill22/).
In-server screen (flag file per request, ABBA by prompt, every prompt carries a unique head so no request is a prefix-cache hit; pool + npipe + subp + fuse vs none, 29 realistic prompts + 24 probes): decode 34.74 vs 32.30 tok/s (+7.76 %, 18/29 faster, se 2.67 %), probes 27.90 vs 26.77 (+3.9 %).
PLE first-group warm-ahead on fresh text with equal prompt token counts (count_tokens, 4 repeats, ABBA): f0 / f1 (plew=1) tok/s 8K 450 / 431, 18K 482 / 483, 33K 546 / 547 -> not shipped.
ABBA (E F F E, fresh restart per arm): decode realistic E 30.54 / 30.83, F 32.74 / 32.85 tok/s (the two arms of each side: E 30.69, F 32.80, +6.79 %, paired 28/29 faster, se 0.78 %); probes E 26.74 / 27.01 (26.88), F 28.12 / 27.62 (27.88); prefill (4 runs each, tok/s) 2,565 tokens E 208 233 208 232, F 219 246 221 249; 8,018 E 456 511 455 513, F 469 526 469 527; 19,547 E 517 551 517 550, F 543 582 542 579; 33,842 E 580 615 580 617, F 602 641 600 640. Agent turns (new tokens: wall s, E / F, mean of 4): 30: 2.76 / 2.66, 60: 3.92 / 3.62, 100: 4.97 / 4.67, 150: 5.75 / 5.30, 200: 6.11 / 5.71, 250: 6.26 / 5.83, 300: 7.29 / 6.83, 400: 7.29 / 6.96, 600: 9.30 / 9.33, 1000: 9.46 / 9.38.
Regression (final config, ~110 requests): realistic 32.62 tok/s (TTFT 19.7 s), prefill 231.5 / 225.6, 469.7 / 526.5, 515.1 / 551.0, 606.2 / 639.1 tok/s (2.5K / 8K / 18K / 33K), tool reuse PASS (3,379 reused), image PASS, ~190K needle FOUND (352 s), 5 x 32K needles FOUND, 0 GPU timeouts. Quality gate (teacher-forced, 705 positions) vs the saved baseline: top-1 97.73 %, mean KL 0.005474, p99 KL 0.0817 (PASS); vs config E: top-1 100 %, KL 0.
Soak (5 successive ~33K needles, stop at the first failure or GPU timeout): 5/5 found, 0 GPU timeouts. CPU-load USB check (server down, 8 CPU hogs, `stress21.py`: random sizes / offsets / patterns, host-to-device then device-to-host compare, fence 11): seed 1 10,482 iterations 26.3 GB, seed 2 10,544 iterations 26.4 GB, 0 mismatches.
Pool finding: with `TGCU_LAZY_POOL=3` (the lazy shadow pool) the teacher-forced quality flow (`xp8_quality.py lp`, its 5th prompt) crashed the engine twice in a row (`bulk IN short read: 94728/102912 bytes` in the decode ring's `hs_wait`). Bisect on the first 5 prompts of that flow: in the flag-file server (pool + npipe + subp + fuse, no log-probability env) every arm passed; with `STRATA_LOGPOS` set, the pool present and any one of npipe / subp / fuse removed still crashed, `STRATA_LOGPOS` alone with the pool crashed, the pool removed passed, `STRATA_DEC_ROUTE_DUMP` alone with the pool passed: the trigger is the pool together with the log-probability dump. The first F candidate (with the pool) had passed ABBA (decode +6.50 %), the regression and the soak; only the quality gate caught it, so the pool is not in config F and the quality gate is part of every promotion.

## 0.00 Latest configuration: `configs/strata-exl3-full512-e.json.tmpl` (config E, FewerTrips19, 2026-10-17)

= config D (section 0.0) + patch `strata/0021` (`STRATA_D19_HOIST=1`). Live config md5 `078235c5bfec8bac85dc0a39d562485d`, engine `strata_d19b` md5 `8b11081ba95a61982e5247e7e09c4aa9`; shim, Metal library and every other knob as config D. Raw: [`results/raw/fewertrips19/`](../results/raw/fewertrips19/).
ABBA (C D E E D C, fresh restart per arm, 29 realistic prompts + 12 probes): decode realistic C 31.03 / 31.24, D 29.83 / 29.80, E 31.00 / 30.88 tok/s (E vs D +3.84 %, paired 24/29 faster, se 0.78 %; E vs C -0.52 %); probes C 26.83 / 26.93, D 26.26 / 26.28, E 26.71 / 26.23; TTFT D 20.2 s, E 20.2 s, C 23.8 s.
USB / ring time per window (config D, `STRATA_HS_TRACE`, per cold ring, 44 rings per window): wait-for-ring 0.319 -> 0.282 ms with E, x pull 0.036, y push 0.247, flag push 0.020, pool (Metal job + SSD reads) 1.21 ms, host busy 1.47 ms: ~27 ms of a ~80-90 ms window is USB + card chain, ~53 ms is the mini's pool.
Regression (final config, ~110 requests): realistic 30.85 tok/s (TTFT 20.2 s), prefill 230 / 224, 459 / 512, 491 / 520, 588 / 621 tok/s (2.5K / 8K / 18K / 33K), tool reuse PASS (3,379 reused), image PASS, ~190K needle FOUND (356 s), agent turns 2.7-9.4 s, 5 x 32K needles FOUND, 0 GPU timeouts. Quality gate (teacher-forced, 705 positions) vs the saved baseline: top-1 97.73 %, mean KL 0.005474, p99 KL 0.0817 (PASS); vs the config D run: p99 KL 0.1267 (informational).
Negative / not built (simulations on the Decode7 routing dump, `results/raw/fewertrips19/sim/table19_*.out`): mini-side layer blocks -0.6..-11 %; dynamic zero-cold ring skip +2.7 % at best; card-only self-speculation (teacher-forced top-1 agreement 92.3 % with the cold experts skipped, 88.2 % with `STRATA_EXL3_RESIDENT_ONLY` routing; one card-only token costs 22-25 ms, break-even <= 16-18 ms); prefetch from the card-only routing (recall 0.87 of the exact experts, -70 % misses) cannot pay for the card-only pass; `--spec 5/6` 30.63 / 29.67 vs 29.82 on 10 prompts (no gain).

## 0.0 Latest configuration: `configs/strata-exl3-full512-d.json.tmpl` (config D, Prefill18, 2026-10-15)

= config C (section 0) + patch `strata/0020` (lazy loan refill; `STRATA_D7_MODE=15`, `STRATA_D18_K=300`). Live config md5 `b481bfe4dd4f299be24c59c70f4e12cc`, engine `strata_d18d` md5 `b5b0e1bcc566424353e37625aba1c7d8`; shim and Metal library as config C. Raw: [`results/raw/prefill18/`](../results/raw/prefill18/).
Prefill (2 runs each, tok/s): 2,565 tokens 229.1 / 226.0; 8,018 457.2 / 510.5; 18,330 489.1 / 520.9; 33,842 588.5 / 618.6 (config C: 176 / 383-418 / 446-470 / 555-584). Realistic 29 prompts 29.70 tok/s, TTFT 20.2 s (config C 31.03, 23.75 s); 12 probes 25.88 (26.84).
Controlled in-server A/B (flag per request, 29 prompts x 2): prefill TTFT 23.07 -> 19.63 s, decode -1.5 % (se 1.6 %). Quality gate vs the saved baseline: top-1 97.59 %, mean KL 0.005303, p99 KL 0.056416 (PASS); image, tool reuse, 190K needle (357 s), 5 x 32K needles PASS, 0 GPU timeouts.
Sweeps (decode paired vs control): K=0 -4.7 %, K=300 -1.5 %, 300 slots pending (K=584) +0.9 %, G=9 K=300 -3.2 %, refill between decode windows -15.3 %. 2.5K prompts read 18.3 GB per request (SSD-bound, ~280 tok/s ceiling).

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
- **Approximate (config S2):** `STRATA_PF_XPORT_BITS=5` narrows that lane transport to 5-bit codes + one fp16 scale per 32 values (rel RMS 4.69 % on real x rows, int8 0.56 %); gated on the 705-position test and on a 2,139-position panel against the fp16 lane (section 0.000000.3: mean KL 0.004077 vs int8's 0.003590, p99 0.0459 vs 0.0550). `STRATA_PF_GRP_FIT` (a table size) and `STRATA_PF_SPLIT2` (the turn order) are exact; `--prefill 3072` with `STRATA_PF_BIG_MIN=28000` changes the chunking of prompts of >= 28K tokens only, which is rounding-level like any chunk-size change (the fp32 association and residency effects above).
- **Not measured:** KL or top-1 agreement against stock exllamav3 for any of these configs; GPQA Diamond; tool-calling benchmarks; concurrency above 1 with the shipped engine (section 0.4 measured the batch engine, which is slower); the host-tail config on the 190K needle
  (config C: found, 189,481 tokens); a vision run after the LRU change of the host-tail step as a raw log (config C: image PASS in `d15_final.log`, `newA.image.log`, `newB.image.log`).
