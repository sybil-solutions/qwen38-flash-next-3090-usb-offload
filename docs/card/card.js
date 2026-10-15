// Card text: measured numbers from the live config C (Decode16, 2026-10-15; docs/results.md section 0). Render: ./render.sh -> ../exl3-card.png
var CARD = {
  word: 'Sybil AI',
  credit: 'art · @samdape',
  tag: 'Measured',
  model: 'Qwen3.8-Flash-Next · EXL3 3bpw',
  sub: 'all 512 experts · 1× RTX 3090 over USB + Mac mini M4 16 GB · Strata + Metal',
  stats: [['31', 'tok/s decode c1 · 26.8 short'], ['584', 'tok/s prefill @33K · 418 @8K'], ['200K', 'context · needle ok'], ['vision', 'image in · Metal encoder']],
  detail: ['turboderp 3.05bpw', 'MTP spec 4', 'k8v4 elastic KV', '9,958 experts in VRAM', '6,400 in mini RAM', 'rest on NVMe', 'p99 KL 0.058 vs reference run'],
  row: ['PCIe Gen3 x2 behind USB 3.2 10 Gb/s', 'config C · updated 2026-10-15', 'targets 33 / 600 not reached'],
  foot: '@0xSero',
};
