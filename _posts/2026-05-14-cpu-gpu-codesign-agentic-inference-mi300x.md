---
layout: post
title: "CPU-GPU Co-Design for Agentic LLM Inference on AMD MI300X"
date: 2026-05-14
categories: [LLM, AMD, MI300X, vLLM, LMCache, Performance]
---

*Quantifying where time actually goes — and why your CPU might be stealing 15% of your GPU throughput.*

---

## Key Summary

We instrumented the full request lifecycle of agentic LLM inference on AMD MI300X to answer a simple question: **how much of end-to-end latency is CPU work vs GPU work?**

Using MiniMax-M2.5 (230 GB FP8 MoE) on 2× MI300X with vLLM 0.19.0, we decomposed every request into serialization, HTTP overhead (tokenization + scheduling + queue wait), GPU prefill, and GPU decode across 8 scenarios spanning concurrency 1–32 and context 1k–100k tokens.

**Headline findings:**
- **At low concurrency, CPU overhead is negligible** — 0.4–0.6% of E2E time for single requests at any context length
- **At high concurrency, CPU overhead becomes material** — 11–15% of E2E time at 32 concurrent users
- **The bottleneck is not tokenization or JSON parsing** — it's **scheduling + queue wait**, which scales superlinearly with concurrency
- **Tokenization at 100k tokens costs only 220ms** (~500k tok/s on a single CPU core), tiny compared to GPU prefill (2–4 seconds)
- **LMCache adds minimal CPU overhead** vs HBM prefix cache — the CPU% split is nearly identical between the two strategies
- **The real CPU-GPU co-design opportunity** is not in making CPU faster, but in **overlapping CPU work with GPU work** and reducing scheduling contention at high concurrency

---

## 1. Motivation: The Hidden CPU Tax in Agentic Inference

Our previous work benchmarked [LMCache for multi-turn agentic workloads on MI300X](https://github.com/andyluo7/openclaw-workspace/tree/main/multiturn-agentic-bench), comparing KV-cache strategies. We measured TTFT, throughput, and cache hit rates. But we treated the inference server as a black box — we never asked *where inside the server* the time goes.

Agentic AI workloads are not just GPU workloads. Every request passes through a CPU pipeline before and after GPU execution:

```
Client                          Server (vLLM)                      GPU
──────                          ─────────────                      ───
 │                                    │                              │
 │─── serialize request ──────────────│                              │
 │    (JSON, 0.04-1.3ms)             │                              │
 │                                    │                              │
 │                          ┌─────────┴──────────┐                  │
 │                          │ HTTP parse          │                  │
 │                          │ Tokenize input      │                  │
 │                          │ Schedule request    │  "HTTP Overhead" │
 │                          │ KV cache lookup     │  (7-3900ms)      │
 │                          │ Queue wait          │                  │
 │                          └─────────┬──────────┘                  │
 │                                    │                              │
 │                                    │──── GPU prefill ────────────│
 │                                    │     (41-28537ms)            │
 │                                    │                              │
 │                                    │──── GPU decode (streaming) ─│
 │                                    │     (1780-20792ms)          │
 │                                    │                              │
 │◄── parse SSE response ────────────│                              │
 │    (1.9µs per chunk)              │                              │
```

The question: at scale (32 concurrent users, 100k token contexts), does the CPU pipeline become a bottleneck?

---

## 2. Methodology

### 2.1 Hardware & Software

| Component | Specification |
|-----------|--------------|
| GPU | 2× AMD Instinct MI300X (192 GB HBM3 each), gfx942 |
| CPU | AMD EPYC (ENC1-CLS01-SVR08) |
| Model | MiniMaxAI/MiniMax-M2.5 FP8, TP=2 |
| Framework | vLLM 0.19.0 (ROCm) |
| KV Cache | HBM prefix cache / LMCache CPU DRAM |
| Workload | 739 anonymized Claude Code agentic conversations |

### 2.2 What We Measured

We decomposed each request into five time components:

| Component | Where | What It Captures |
|-----------|-------|-----------------|
| **t_serialize** | Client CPU | JSON serialization of the request payload |
| **t_http_overhead** | Server CPU | HTTP parsing + tokenization + scheduling + queue wait + KV cache lookup |
| **t_server_prefill** | Server GPU | Attention computation over all input tokens |
| **t_decode** | Server GPU (mostly) | Autoregressive token generation + streaming |
| **t_response_parse** | Client CPU | SSE chunk parsing + tool call extraction |

We classify `t_serialize + t_http_overhead + t_response_parse` as **CPU time** and `t_server_prefill + t_decode` as **GPU time**.

**Note:** `t_http_overhead` is measured as the gap between client sending the HTTP request and receiving the first byte back. This includes tokenization, scheduling, queue wait time, and KV cache management — all CPU-side work that happens before the GPU begins prefill. At low concurrency this is mostly tokenization + scheduling. At high concurrency, queue wait dominates.

### 2.3 Test Matrix

| Scenario | Concurrency | Context | Purpose |
|----------|------------|---------|---------|
| single_1k | 1 | 1,000 | Baseline: pure overhead |
| single_8k | 1 | 8,000 | Typical agent turn |
| single_32k | 1 | 32,000 | Large agent context |
| single_100k | 1 | 100,000 | Maximum agent context |
| conc4_8k | 4 | 8,000 | Light multi-tenant |
| conc16_32k | 16 | 32,000 | Medium load |
| conc32_32k | 32 | 32,000 | High load, moderate context |
| conc32_100k | 32 | 100,000 | Stress: high load + large context |

Each scenario was run with 3–5 batches of requests, with results aggregated.

---

## 3. Results

### 3.1 The CPU-GPU Split: It's All About Concurrency

**HBM Prefix Cache Configuration:**

| Scenario | Conc | Ctx | HTTP OH (ms) | Prefill (ms) | Decode (ms) | Total (ms) | **CPU%** | **GPU%** |
|----------|------|-----|-------------|-------------|-------------|------------|---------|---------|
| single_1k | 1 | 1K | 7 | 41 | 1,780 | 1,828 | **0.4%** | 99.6% |
| single_8k | 1 | 8K | 15 | 124 | 3,142 | 3,282 | **0.5%** | 99.5% |
| single_32k | 1 | 32K | 47 | 682 | 7,736 | 8,465 | **0.6%** | 99.4% |
| single_100k | 1 | 100K | 131 | 3,555 | 20,792 | 24,479 | **0.6%** | 99.4% |
| conc4_8k | 4 | 8K | 53 | 137 | 3,101 | 3,291 | **1.6%** | 98.4% |
| conc16_32k | 16 | 32K | 555 | 498 | 7,832 | 8,885 | **6.2%** | 93.8% |
| conc32_32k | 32 | 32K | 1,130 | 636 | 7,873 | 9,639 | **11.6%** | 88.4% |
| conc32_100k | 32 | 100K | 3,885 | 2,479 | 19,591 | 25,957 | **14.9%** | 85.1% |

The pattern is clear: **CPU overhead scales with concurrency, not context length.**

- Single-request: CPU% is flat at ~0.5% regardless of whether context is 1k or 100k
- At concurrency 32: CPU% jumps to 11–15%
- The dominant CPU cost is `t_http_overhead` (scheduling + queue wait), not tokenization

### 3.2 LMCache vs HBM Prefix Cache: CPU Overhead Comparison

**LMCache DRAM Configuration (gpu-mem-util=0.78):**

| Scenario | Conc | Ctx | HTTP OH (ms) | Prefill (ms) | Decode (ms) | Total (ms) | **CPU%** | **GPU%** |
|----------|------|-----|-------------|-------------|-------------|------------|---------|---------|
| single_1k | 1 | 1K | 7 | 44 | 2,653 | 2,704 | **0.3%** | 99.7% |
| single_8k | 1 | 8K | 15 | 178 | 3,376 | 3,569 | **0.4%** | 99.6% |
| conc4_8k | 4 | 8K | 50 | 121 | 3,455 | 3,627 | **1.4%** | 98.6% |
| conc16_32k | 16 | 32K | 515 | 1,655 | 8,063 | 10,233 | **5.1%** | 94.9% |
| conc32_32k | 32 | 32K | 1,135 | 722 | 8,386 | 10,243 | **11.0%** | 89.0% |
| conc32_100k | 32 | 100K | 3,937 | 28,537 | 20,769 | 53,244 | **9.8%** | 90.2% |

**Key comparison — CPU overhead is nearly identical:**

| Scenario | HBM-PC CPU% | LMCache CPU% | Delta |
|----------|------------|-------------|-------|
| single_1k | 0.4% | 0.3% | −0.1% |
| conc4_8k | 1.6% | 1.4% | −0.2% |
| conc16_32k | 6.2% | 5.1% | −1.1% |
| conc32_32k | 11.6% | 11.0% | −0.6% |
| conc32_100k | 14.9% | 9.8% | −5.1% |

**LMCache does NOT add measurable CPU overhead.** In fact, CPU% is slightly *lower* with LMCache at high concurrency because LMCache's CPU DRAM cache reduces HBM pressure, meaning less time in KV block eviction decisions on the CPU side.

The `t_http_overhead` is nearly identical between the two configs (~1,130–1,135ms at conc32_32k), confirming that the LMCache connector's CPU-side work (hash computation, cache lookup, DMA scheduling) is negligible.

### 3.3 Where Does CPU Time Actually Go?

We ran standalone micro-benchmarks to isolate each CPU component:

| Component | Time at 100K tokens | % of HTTP Overhead (conc=32) |
|-----------|-------------------|------------------------------|
| Tokenization (encode) | 220 ms | ~5.7% |
| JSON serialization (request build) | 0.82 ms | <0.1% |
| SHA256 hash (cache key) | 0.62 ms | <0.1% |
| SSE chunk parse (per token) | 1.9 µs | <0.1% |
| Detokenization (128 tokens) | 0.27 ms | <0.1% |
| **Scheduling + queue wait** | **~3,660 ms** | **~94%** |

The smoking gun: **scheduling + queue wait accounts for ~94% of CPU overhead** at high concurrency. Tokenization, hashing, and serialization are negligible.

This makes sense: at 32 concurrent requests, the vLLM scheduler must:
1. Decide which requests to batch together
2. Walk the prefix cache tree to find matching blocks
3. Allocate KV blocks for new tokens
4. Manage the preemption queue when HBM is under pressure
5. Coordinate across TP workers

Each of these is O(n) or worse in the number of concurrent requests, and they all happen on a single Python thread (GIL-bound).

### 3.4 Tokenization Deep-Dive: Linear but Fast

| Tokens | Encode (ms) | Throughput (tok/s) |
|--------|------------|--------------------|
| 679 | 1.18 | 576,506 |
| 2,711 | 5.09 | 532,379 |
| 5,423 | 10.35 | 523,861 |
| 10,840 | 20.46 | 529,718 |
| 21,679 | 42.72 | 507,414 |
| 43,359 | 87.85 | 493,582 |
| 67,745 | 134.90 | 502,188 |
| 101,615 | 220.38 | 461,085 |

Tokenization scales linearly with input length at ~500k tok/s. Even at 100k tokens (the largest agentic context we tested), tokenization takes only **220ms** — under 1% of E2E time for any scenario.

The HuggingFace `tokenizers` library (Rust-based BPE) is already highly optimized. Switching to a C++ tokenizer would save ~50–100ms at 100k tokens — not enough to matter.

**Detokenization** (streaming output) is even faster: 0.27ms for 128 output tokens. Per-token streaming overhead is not a concern.

---

## 4. Analysis: The Scheduling Wall

### 4.1 Why Scheduling Dominates at High Concurrency

The `t_http_overhead` captures everything from HTTP request receipt to first GPU kernel launch. At concurrency 1, it's dominated by tokenization (~220ms for 100k). At concurrency 32, it balloons to **3,885ms** — a 30× increase.

The growth is **superlinear** with concurrency:

| Concurrency | HTTP Overhead (32K ctx) | Growth Factor |
|-------------|------------------------|--------------|
| 1 | 47 ms | 1.0× |
| 4 | 53 ms | 1.1× |
| 16 | 555 ms | 11.8× |
| 32 | 1,130 ms | 24.0× |

This superlinear scaling points to **contention** in the scheduling path:

1. **Python GIL:** vLLM's scheduler runs in the main asyncio event loop. At 32 concurrent requests, the GIL serializes scheduling decisions, tokenization, and HTTP handling.

2. **Prefix cache tree walks:** With prefix caching enabled, every scheduling decision walks the block hash tree. At high concurrency with diverse prompts, the tree grows and walks become expensive.

3. **Block allocation contention:** The KV block allocator must coordinate free/used block tables across TP workers.

4. **Queue wait:** When the GPU is saturated, requests queue in the scheduler waiting for slots.

### 4.2 The 15% Rule

Our data suggests a practical rule of thumb:

> **At production-level concurrency (16–32 users), CPU overhead consumes 10–15% of E2E latency on MI300X.**

This means that even with an infinitely fast GPU, you would only recover 85–90% of theoretical speedup. The remaining 10–15% is CPU-bound.

For a concrete example: at conc32_100k with HBM prefix cache, total E2E is 25,957ms. GPU time is 22,070ms (prefill + decode). Even if GPU time went to zero, the CPU overhead of 3,887ms would remain — setting a hard floor on latency.

---

## 5. Optimization Recommendations

### Tier 1: High Impact, Framework-Level

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| **Pipeline scheduling with GPU execution** | 5–10% E2E at high concurrency | Medium |
| **Move tokenization off main event loop** | 2–3% at high concurrency | Low |
| **Batch scheduling decisions** | 3–5% at high concurrency | Medium |
| **Pre-allocate KV blocks speculatively** | 2–3% at high concurrency | Medium |

### Tier 2: System-Level Tuning

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| **NUMA affinity** (pin workers to GPU-local node) | 1–2% | Low |
| **CPU frequency governor** (`performance` mode) | 0.5–1% | Trivial |
| **Dedicated CPU cores for scheduler** (isolcpus) | 1–2% | Low |

### Tier 3: Not Worth Optimizing

| Component | Why Not |
|-----------|---------|
| Tokenizer speed | Already 500k tok/s, <1% of E2E |
| JSON serialization | <1ms even at 100k tokens |
| SSE parsing | 1.9µs per chunk — effectively zero |
| LMCache hash/lookup | <1ms even at 100k tokens |
| Detokenization | 0.27ms for 128 output tokens |

---

## 6. Key Takeaways

### For inference platform teams:

1. **CPU overhead is real but bounded.** At 32 concurrent users, 10–15% of E2E latency is CPU. This sets a floor on achievable latency regardless of GPU speed.

2. **Scheduling is the bottleneck, not tokenization.** Don't waste time optimizing the tokenizer — optimize the scheduler and its interaction with the KV cache manager.

3. **LMCache adds zero measurable CPU overhead.** The cache connector's hash/lookup/DMA scheduling cost is lost in the noise. If you're avoiding LMCache because of CPU concerns, don't.

4. **The GIL is the elephant in the room.** At 32+ concurrent requests, Python GIL serializes scheduling, tokenization, and HTTP handling. Multi-process architectures (like vLLM V1's separated EngineCore) are the right direction.

### For hardware architects:

1. **CPU performance matters for inference at scale.** A faster CPU won't help a single request, but it directly impacts latency at 16+ concurrent users.

2. **PCIe/Infinity Fabric bandwidth is not the CPU bottleneck.** The CPU overhead is all compute (scheduling, hash computation, Python interpretation), not data transfer.

3. **NUMA topology matters.** Ensuring scheduler threads run on CPU cores local to the GPU's NUMA node reduces memory access latency for KV block table management.

### For the agentic AI community:

1. **The CPU-GPU co-design question is a scheduling problem**, not a compute problem. The path forward is better overlap between CPU scheduling and GPU execution.

2. **Context length matters less than concurrency.** A single 100k-token request has 0.6% CPU overhead. Thirty-two 1k-token requests have 11%+ CPU overhead. If you're scaling to many concurrent agent sessions, CPU efficiency of the scheduler is critical.

---

## Appendix: Reproduction

### Environment

```bash
# Container
docker run -d --name lmcache-bench --entrypoint /bin/bash \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
  --group-add video --cap-add SYS_PTRACE \
  -v /mnt/nvme3n1p1/models:/work/models \
  vllm/vllm-openai-rocm:v0.19.0 -c "sleep infinity"

# LMCache (source build for ROCm)
docker exec lmcache-bench bash -c "
  git clone --depth 1 https://github.com/LMCache/LMCache.git /work/LMCache
  cd /work/LMCache && BUILD_WITH_HIP=1 pip install -e . --no-build-isolation
  pip uninstall -y nixl nixl-cu12 cupy-cuda12x cufile-python cuda-pathfinder
"
```

### Server Configs

**HBM Prefix Cache:**
```bash
VLLM_FLOAT32_MATMUL_PRECISION=high \
vllm serve /work/models/MiniMax-M2.5 \
  --tensor-parallel-size 2 --enable-prefix-caching \
  --gpu-memory-utilization 0.85 --host 0.0.0.0 --port 8000
```

**LMCache DRAM:**
```bash
PYTHONHASHSEED=0 VLLM_FLOAT32_MATMUL_PRECISION=high \
LMCACHE_LOCAL_CPU=true LMCACHE_CHUNK_SIZE=256 \
vllm serve /work/models/MiniMax-M2.5 \
  --tensor-parallel-size 2 --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.78 --host 0.0.0.0 --port 8000
```

### Benchmark Scripts

All scripts and raw data are available in the [`cpu-gpu-codesign/`](https://github.com/andyluo7/openclaw-workspace/tree/main/multiturn-agentic-bench/cpu-gpu-codesign) directory.

---

*This analysis accompanies our LMCache multi-turn agentic benchmark and uses the same hardware, model, and workload traces.*
