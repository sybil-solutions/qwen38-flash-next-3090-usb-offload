// tgusb21.c (Prefill21): libusb asynchronous transfer executor for the tgcu prompt lane.
//
// One USB transaction on the ASM2464 bridge costs ~90 us of round-trip LATENCY (libusb sync control or bulk, from C or Python alike), not throughput: a bulk OUT of 192 KB takes 0.245 ms (800 MB/s) but the
// pipelined upload loop paid 0.53 ms per round because every control transfer (arm, semaphore poll = request + completion) is its own 90 us round.  Here the transfers of one step are queued asynchronously
// (up to ~4 control transfers, or a control + the bulk it announces, finish in ONE round) and the whole message loops (tgu_h2d_pipe / tgu_d2h_pipe) run without returning to the interpreter.
// Order on the wire = order of submission within an endpoint; control transfers (EP0) vs bulk (EP2 / EP1 IN) are ordered by FENCES (wait for everything in flight) where a later transfer must not overtake an earlier
// one: g_fence (set by tgu_fence) bit 0: a fence before every control transfer that follows a bulk, bit 1: a fence between an arm and its bulk.  libusb ships no header on the mini: the entry points are passed in by
// the loader (tgu_init); the transfer struct is the 1.0.2x layout.
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
typedef void H; typedef void CTX;
struct xfer {
  H* dev_handle; uint8_t flags; unsigned char endpoint; unsigned char type; unsigned timeout; int status; int length; int actual_length;
  void (*callback)(struct xfer*); void* user_data; unsigned char* buffer; int num_iso_packets;
};
static int (*p_submit)(struct xfer*);
static int (*p_events)(CTX*, struct timeval*);
static struct xfer* (*p_alloc)(int);
static H* g_h; static CTX* g_ctx;
#define MAXOPS 24
static struct xfer* g_t[MAXOPS];
static unsigned char g_cb[MAXOPS][8 + 16];
static volatile int g_inflight;
static int n_ops;
static int g_fence = 2;
static int g_last_status = 0;
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static void cb(struct xfer* t) { (void) t; --g_inflight; }
int tgu_init(void* alloc, void* submit, void* events, H* h, CTX* ctx) {
  p_alloc = (struct xfer* (*)(int)) alloc; p_submit = (int (*)(struct xfer*)) submit; p_events = (int (*)(CTX*, struct timeval*)) events; g_h = h; g_ctx = ctx;
  for (int i = 0; i < MAXOPS; ++i) if (!(g_t[i] = p_alloc(0))) return -1;
  return 0;
}
void tgu_fence(int mode) { g_fence = mode; }
int tgu_last_status(void) { return g_last_status; }
static void begin(void) { n_ops = 0; g_inflight = 0; }
static int drain(double timeout_ms) {
  double t0 = now_ms();
  while (g_inflight > 0) {
    struct timeval tv = {0, 100000}; p_events(g_ctx, &tv);
    if (now_ms() - t0 > timeout_ms) return -2;
  }
  return 0;
}
static int add_ctrl(uint8_t bm, uint8_t req, uint16_t wv, uint16_t wi, const void* data, int len) {   // bm 0x40 = vendor OUT with `len` data bytes, 0xC0 = vendor IN reading `len` bytes
  if (n_ops >= MAXOPS || len > 16) return -9;
  struct xfer* t = g_t[n_ops]; unsigned char* b = g_cb[n_ops];
  b[0] = bm; b[1] = req; memcpy(b + 2, &wv, 2); memcpy(b + 4, &wi, 2); uint16_t l = (uint16_t) len; memcpy(b + 6, &l, 2); if (data && len) memcpy(b + 8, data, len);
  t->dev_handle = g_h; t->flags = 0; t->endpoint = 0; t->type = 0; t->timeout = 5000; t->status = -1; t->length = 8 + len; t->actual_length = 0; t->callback = cb; t->user_data = 0; t->buffer = b;
  ++n_ops; ++g_inflight;
  if (p_submit(t)) { --g_inflight; --n_ops; return -1; }
  if ((g_fence & 4) && drain(6000)) return -2;
  return 0;
}
static int add_bulk(unsigned char ep, void* data, int len, unsigned timeout_ms) {
  if (n_ops >= MAXOPS) return -9;
  struct xfer* t = g_t[n_ops];
  t->dev_handle = g_h; t->flags = 0; t->endpoint = ep; t->type = 2; t->timeout = timeout_ms; t->status = -1; t->length = len; t->actual_length = 0; t->callback = cb; t->user_data = 0; t->buffer = (unsigned char*) data;
  ++n_ops; ++g_inflight;
  if (p_submit(t)) { --g_inflight; --n_ops; return -1; }
  if ((g_fence & 4) && drain(6000)) return -2;
  return 0;
}
static int finish(int rc, double timeout_ms) {
  if (drain(timeout_ms)) return rc ? rc : -2;
  for (int i = 0; i < n_ops; ++i) {
    if (g_t[i]->status != 0) { g_last_status = g_t[i]->status; return -3 - 10 * g_t[i]->status; }
    if (g_t[i]->type == 2 && g_t[i]->actual_length != g_t[i]->length) return -4;
  }
  return rc;
}
static int fence_ctrl(void) { return (g_fence & 1) ? drain(6000) : 0; }
static void f0(uint64_t addr, uint32_t val, uint8_t* d) { uint32_t w[3] = {(uint32_t) addr, (uint32_t) (addr >> 32), val}; memcpy(d, w, 12); }
static uint16_t f0_wv(int fmt_type) { return (uint16_t) (fmt_type | 0x0F00); }
static int poke(uint64_t addr, uint32_t val) { uint8_t d[12]; f0(addr, val, d); return add_ctrl(0x40, 0xF0, f0_wv((addr >> 32) ? 0x60 : 0x40), 0, d, 12); }
static int stream_hdr(uint64_t addr, uint32_t dwords, int write) {
  uint8_t d[12]; f0(addr, dwords, d);
  int rc = add_ctrl(0x40, 0xF0, f0_wv(write ? ((addr >> 32) ? 0x60 : 0x40) : ((addr >> 32) ? 0x20 : 0x00)), write ? 1 : 2, d, 12);
  if (!rc && (g_fence & 8) && drain(6000)) return -2;   // the stream header must be through before its bulk
  return rc;
}

// ---- 32-bit PCIe memory read (the semaphore poll): request control OUT + completion control IN, queued back to back. returns 0 and *val, or <0.
static int read32_q(uint64_t addr) {   // queue the pair (ops n_ops, n_ops + 1)
  uint8_t d[12]; f0(addr, 0, d); int rc;
  if ((rc = add_ctrl(0x40, 0xF0, f0_wv((addr >> 32) ? 0x20 : 0x00), 0, d, 12))) return rc;
  return add_ctrl(0xC0, 0xF0, 0, 0, 0, 8);
}
static int read32_res(int op, uint32_t* val) {   // after finish: status bytes of the completion at op + 1
  const unsigned char* r = g_cb[op + 1] + 8;
  if (r[7] != 0 || ((r[4] >> 5) & 7) != 0) return -20 - r[7];
  uint32_t v; memcpy(&v, r, 4); *val = v; return 0;
}
int tgu_read32(uint64_t addr, uint32_t* val) {
  for (int tries = 0; tries < 11; ++tries) {
    begin(); int rc = read32_q(addr);
    if (rc) { finish(rc, 6000); return rc; }
    rc = finish(0, 6000); if (rc) return rc;
    rc = read32_res(0, val);
    if (rc == 0) return 0;
    struct timespec ts = {0, 1000000}; nanosleep(&ts, 0);
  }
  return -21;
}
static int sem_ge(uint32_t cur, uint32_t want) { return ((cur - want) & 0xffffffffu) < 0x80000000u; }
// poll `addr` until it reaches `want` (modular compare, as AsyncCopy._wait_sem). returns 0, -2 on timeout
int tgu_wait_sem(uint64_t addr, uint32_t want, double timeout_ms) {
  double t0 = now_ms(); uint32_t cur;
  for (;;) {
    int rc = tgu_read32(addr, &cur); if (rc) return rc;
    if (sem_ge(cur, want)) return 0;
    if (now_ms() - t0 > timeout_ms) { fprintf(stderr, "[tgusb] wait_sem %llx want %u cur %u timeout\n", (unsigned long long) addr, want, cur); return -2; }
  }
}

// ---- submission of a pushbuffer to a GPFIFO (ring entry, GPPut, doorbell), optionally followed by a bulk-arm (0xF2) and the bulk OUT payload it announces. *_addr are PCIe addresses.
// [stream hdr, bulk pushbuffer] [ring entry dword pokes] [GPPut poke] [doorbell poke] [arm] [bulk data]
int tgu_submit(const uint32_t* pb, int pb_dwords, uint64_t pb_addr, uint64_t entry_addr, uint32_t entry_lo, uint32_t entry_hi, int lo_changed, int hi_changed,
               uint64_t gpput_addr, uint32_t gpput_val, uint64_t db_addr, uint32_t db_val, int arm_wv, int arm_wi, void* data, int data_len, int arm_first) {
  begin(); int rc;
  if (arm_wv >= 0 && arm_first && (rc = add_ctrl(0x40, 0xF2, (uint16_t) arm_wv, (uint16_t) arm_wi, 0, 0))) return finish(rc, 6000);   // a download is armed BEFORE its copy-engine job is submitted
  if ((rc = stream_hdr(pb_addr, (uint32_t) pb_dwords, 1))) return finish(rc, 6000);
  if ((rc = add_bulk(0x02, (void*) pb, pb_dwords * 4, 5000))) return finish(rc, 6000);
  if (fence_ctrl()) return -2;
  if (lo_changed && (rc = poke(entry_addr, entry_lo))) return finish(rc, 6000);
  if (hi_changed && (rc = poke(entry_addr + 4, entry_hi))) return finish(rc, 6000);
  if ((rc = poke(gpput_addr, gpput_val))) return finish(rc, 6000);
  if ((rc = poke(db_addr, db_val))) return finish(rc, 6000);
  if (arm_wv >= 0 && !arm_first) {
    if ((rc = add_ctrl(0x40, 0xF2, (uint16_t) arm_wv, (uint16_t) arm_wi, 0, 0))) return finish(rc, 6000);
    if (data_len > 0) {
      if ((g_fence & 2) && drain(6000)) return -2;
      if ((rc = add_bulk(0x02, data, data_len, 5000))) return finish(rc, 6000);
    }
  }
  return finish(0, 6000);
}

// ---- pipelined upload (AsyncCopy._h2d_pipe's round loop).  The copy-engine jobs (one gated job per round, each waiting for its round's sentinel in SRAM) were queued by the caller with ONE pushbuffer.
// round i: payload m = min(rb, n - i * rb) bytes + the 8-byte gate sentinel at the end of a wire block of round_up(m + 8, 512) bytes, alternating between two SRAM regions (slot0 + (i & 1) * pipe_slots);
// before round i >= 2 the job of round i - 2 must have drained its region (semaphore >= sem_base + (i - 1)); at the end the last job's semaphore.
// `bulk` is the libusb buffer (>= wire bytes).  Per round the polls for round i + 1 are queued behind round i's bulk, so a poll costs no extra turn.
int tgu_h2d_pipe(const uint8_t* src, uint64_t n, uint8_t* bulk, uint64_t sem_addr, uint32_t sem_base, uint32_t gate_seq0, int slot0, int pipe_slots, int slot_size, int rb, double timeout_ms) {
  uint64_t rounds = (n + (uint64_t) rb - 1) / (uint64_t) rb;
  uint32_t cur = 0; int have = 0;   // the last semaphore value read
  double t0 = now_ms();
  for (uint64_t i = 0; i < rounds; ++i) {
    uint64_t o = i * (uint64_t) rb; int m = (int) ((n - o) < (uint64_t) rb ? (n - o) : (uint64_t) rb);
    int wire = (m + 8 + 511) / 512 * 512; int reg = (int) (i & 1);
    uint32_t seq = gate_seq0 + (uint32_t) i + 1; uint64_t gate = 0x5A17C0DE00000000ULL | (uint64_t) seq;
    if (i >= 2) {
      uint32_t need = sem_base + (uint32_t) (i - 1);   // the job of round i - 2 (sem values are sem_base + 1 + j)
      while (!(have && sem_ge(cur, need))) {
        int rc = tgu_read32(sem_addr, &cur); if (rc) return rc; have = 1;
        if (now_ms() - t0 > timeout_ms) return -2;
      }
    }
    memcpy(bulk, src + o, (size_t) m);
    memcpy(bulk + wire - 8, &gate, 8);
    begin(); int rc;
    if ((rc = add_ctrl(0x40, 0xF2, (uint16_t) (wire / 512), (uint16_t) ((slot0 + reg * pipe_slots) | (((wire + slot_size - 1) / slot_size) << 8)), 0, 0))) return finish(rc, 6000);
    if ((g_fence & 2) && drain(6000)) return -2;
    if ((rc = add_bulk(0x02, bulk, wire, 1000))) return finish(rc, 6000);
    int pq = -1;
    if (i + 1 < rounds && i + 1 >= 2) {   // the next round's poll rides behind this round's bulk
      if (!(g_fence & 1)) { pq = n_ops; if ((rc = read32_q(sem_addr))) return finish(rc, 6000); }
    }
    rc = finish(0, 6000); if (rc) return rc;
    if (pq >= 0 && read32_res(pq, &cur) == 0) have = 1; else have = 0;
  }
  uint32_t last = sem_base + (uint32_t) rounds;
  while (!(have && sem_ge(cur, last))) {
    int rc = tgu_read32(sem_addr, &cur); if (rc) return rc; have = 1;
    if (now_ms() - t0 > timeout_ms) return -2;
  }
  return 0;
}

// ---- pipelined download (AsyncCopy._d2h_pipe's round loop).  Per round (wire bytes = round_up(prefix + m, 512)): the IN arm (0xF2, value wire/512 | 0x8000), the hold word poke (releases the gated copy-engine job)
// and the bulk IN are queued together; the payload [prefix, prefix + m) of the previous round is copied out while the next round runs (two bulk buffers).  Finally the last job's semaphore.
int tgu_d2h_pipe(uint8_t* dst, uint64_t n, uint8_t* bulkA, uint8_t* bulkB, int R, int prefix, int slot_size, uint64_t hold_addr, uint32_t tok_base, uint64_t sem_addr, uint32_t sem_last, double timeout_ms) {
  uint64_t rounds = (n + (uint64_t) R - 1) / (uint64_t) R;
  uint8_t* prev = 0; uint64_t prev_o = 0; int prev_m = 0;
  for (uint64_t i = 0; i < rounds; ++i) {
    uint64_t o = i * (uint64_t) R; int m = (int) ((n - o) < (uint64_t) R ? (n - o) : (uint64_t) R);
    int wire = (prefix + m + 511) / 512 * 512; uint32_t tok = tok_base + 1 + (uint32_t) i;
    uint8_t* b = (i & 1) ? bulkB : bulkA;
    begin(); int rc;
    if ((rc = add_ctrl(0x40, 0xF2, (uint16_t) (wire / 512 | 0x8000), (uint16_t) (0 | (((wire + slot_size - 1) / slot_size) << 8)), 0, 0))) return finish(rc, 6000);
    if (fence_ctrl()) return -2;
    if ((rc = poke(hold_addr, tok))) return finish(rc, 6000);
    if ((g_fence & 2) && drain(6000)) return -2;
    if ((rc = add_bulk(0x81, b, wire, 30000))) return finish(rc, 30000);
    if (prev) memcpy(dst + prev_o, prev + prefix, (size_t) prev_m);   // the previous round's payload leaves while this round streams
    rc = finish(0, timeout_ms); if (rc) return rc;
    prev = b; prev_o = o; prev_m = m;
  }
  if (prev) memcpy(dst + prev_o, prev + prefix, (size_t) prev_m);
  return tgu_wait_sem(sem_addr, sem_last, timeout_ms);
}
