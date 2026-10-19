# SSD-parked conversation cache (config H, Opt24)

Patch [`patches/strata/0025`](../patches/strata/0025-opt24-ssd-parked-conversation-cache.patch), engine `strata_o24b`, flags `--ssd-park-mib 8192 --ssd-park-dir $HOME/strata-park` in
[`configs/strata-exl3-full512-h.json.tmpl`](../configs/strata-exl3-full512-h.json.tmpl).

**Problem.** The server holds one live conversation (KV + recurrent state). Switching to another conversation (another agent, another chat) and back means re-prefilling the first
one: 41-49 s for a 30K prompt, 133-146 s for 100K at this rig's ~640-740 tok/s.

**What it does.** When a request has finished and the engine has been idle for `--ssd-park-idle-ms` (1.5 s), the finished conversation (>= 6,144 tokens) is written to a session
file in the park directory (the engine's own session format: running state, the deepest checkpoint, every attention layer's K/V, the draft ring; ~19 KiB per token, 592 MiB for 31K
tokens, 1.4 GiB for 97K). Later, a request whose prompt starts with a parked conversation that the live session does not hold reads that file back (validate, H2D) and prefills only the new
tokens, if that gives back at least 4,096 more tokens than the live path. The parked files are bounded by `--ssd-park-mib` (8 GiB in config H, LRU over conversations; a conversation larger
than the bound is not parked), must leave 6 GiB of the volume free, and the index lives in RAM only: the directory is emptied at every server start (parked conversations do not survive a restart).

**Idle-only trade-off.** The write happens only while idle (a 30K conversation: ~2-5 s, in 16 MiB blocks). Any request that arrives stops the write at the next block, whether or not
it continues the conversation. So: a switch away *before* the idle write finishes (about 1.5 s idle + 2-5 s write after the last token) is **not parked**, and switching back re-prefills
as before (measured: a 4 s pause lost the park, A2 re-prefilled in 41.7 s). In exchange the request path pays nothing. The first variant (H0, `strata_o24a`) also saved the outgoing
conversation synchronously on the request path when another conversation arrived: that was parked reliably but cost 2-5 s of every switch and showed as a TTFT regression
(realistic TTFT 16.05 s vs 15.09 s for G in `results/raw/opt24/logs/abba/`). The synchronous save is still available as `--ssd-park-switch-save` (off in config H).

**Measured (Opt24, 2026-10-19, `results/raw/opt24/`).**

| switch-back (A1, pause 15 s, B1, A2; A2 text md5 `7cd8fb6e` = no-switch control) | A2 with park | A2 re-prefill |
|---|---|---|
| 30K | 3.1 s (file read 330 ms, restore 1.47 s, 24 tokens read) | 41-49 s |
| 100K | 5.95 s (read 823 ms, restore 4.36 s, 27 tokens read) | 133-146 s |

Decode / prefill are unchanged (ABBA G H1 H1 G, fresh restart per arm, 29 realistic prompts: decode 33.50 (G) vs 33.47 tok/s (H); prefill rows identical within noise; the
park code is idle-time only). Regression equal to G: 190K needle FOUND 273 s (G 276 s), 5 x 32K needles, image, tool reuse. Teacher-forced quality gate: 705/705 positions, top-1
100 %, p99 KL 0. Guarded 5 x 32K soak PASS, 0 GPU timeouts.

**Disk and privacy.** At most `--ssd-park-mib` (8 GiB) in `$HOME/strata-park`, plus the 6 GiB free floor. Files hold the conversation's KV state (derived from the prompts) unencrypted
with the user's file permissions; the directory is emptied at every server start. Needs `--prompt-cache > 0` and no `--batch` / layer split, else the tier switches itself off with a log line.
`--ssd-park-dir` unset = off = config G behaviour.
