<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# JSEW: an external cache-aware, load-aware router for Dynamo

**Join-the-Shortest-Effective-Workload (JSEW)** picks, for each request, the worker that
minimizes

```text
( W_k  +  (n - hit_k)  +  E[D] ) / c_k
```

- `W_k`: work already queued at worker `k`, as the router estimates it: remaining prefill
  tokens of its in-flight requests plus the expected *remaining* decode of each, conditional on
  the tokens it has produced so far (learned from completions; no per-request prediction).
- `n - hit_k`: prompt tokens worker `k` would have to prefill, from a per-worker **shadow prefix
  index** (an LRU over block hashes sized to the worker's KV cache, updated at routing time).
- `E[D]`: mean decode length; `c_k`: relative capacity weight.

A hysteresis band keeps a session on its hash **home** worker unless another worker is better by
more than a fraction `eta`. The policy is throughput-optimal for fleets of prefix-caching
backends; this directory is its reference implementation as an **external router**.

## Files

| File | What it is |
|---|---|
| `jsew_router.py` | The policy: `block_hashes`, `ShadowWorker`, `RemainingCurve`, `JsewRouter`. No Dynamo dependency. |
| `jsew_proxy.py` | OpenAI-compatible proxy (`/v1/completions`, `/v1/chat/completions`) that routes with `JsewRouter` and forwards to a Dynamo frontend with the `x-dynamo-worker-instance-id` header. |
| `test_jsew_router.py` | Unit tests for the policy. |

## Run

```bash
# workers as usual, e.g. one vLLM per GPU with prefix caching on
python -m dynamo.vllm --model Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching --block-size 64

# the frontend must be in direct mode: it honours the pin header and does not route itself
python -m dynamo.frontend --http-port 8000 --router-mode direct

# the JSEW proxy in front of it
pip install -r requirements.txt
python jsew_proxy.py --frontend http://127.0.0.1:8000 --port 8010 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct --cache-tokens 2700000 --discover

curl localhost:8010/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hi"}],"stream":true}'
curl localhost:8010/jsew/stats
```

Options:

- `--workers id,id,...` or `--discover` (uses the Dynamo runtime's discovery for
  `--endpoint namespace/component/endpoint`, default `dynamo/backend/generate`, polled every
  `--discovery-interval` seconds; `--discovery-backend file|etcd|kubernetes|mem`).
- `--cache-tokens`: KV tokens per worker (read `GPU KV cache size` from the worker log). Sizes the
  shadow index; too small underestimates hits, too large overestimates them.
- `--block-size`: tokens per hash block, match the workers' `--block-size`.
- `--tokenizer`: HF tokenizer for requests that do not carry `nvext.token_data`. Pre-tokenized
  requests skip this.
- Session key for the hash home: `x-jsew-session` header, else `nvext.cache_salt`, else the
  hash of the prompt's second block (the first block is usually a shared system prompt).

The proxy adds `x-jsew-worker: <instance id>` to every response so pinning can be checked
against `nvext.extra_fields: ["worker_id"]`.

## Why direct mode

In `round-robin`, `kv` and the other frontend router modes the frontend ignores
`x-dynamo-worker-instance-id` and routes on its own. Only `--router-mode direct` forwards to the
pinned instance. Direct mode also rejects unpinned requests, so the proxy learns worker ids
from discovery rather than by probing.

## What it gives

Measured on one node with eight Blackwell GPUs, Dynamo 1.4.2, vLLM 0.26, calibrated load sweeps
from 0.7 to 1.1 of fleet capacity, 300 s per point, against Dynamo's round-robin and KV router:

- Claude Code trace (8 x Qwen2.5-7B, 128k context, 120 agentic sessions): cached-token fraction
  0.82 vs 0.65 (KV router) vs 0.37 (round-robin); P50 TTFT 0.19 s vs 0.40-0.63 s vs 1.9-13.6 s;
  round-robin saturates at 4.2 req/s while both cache-aware routers sustain the offered 4.8 req/s.
- Mooncake conversation trace (32 x Qwen2.5-1.5B under CUDA MPS): cached fraction 0.30-0.33 vs
  0.21-0.22 vs 0.08; P50 TTFT 0.09-0.15 s vs 0.11-0.22 s vs 0.15-0.44 s; both cache-aware routers
  complete 84 req/s at 110% load where round-robin completes 74.

The gap over the KV router comes from keeping a session's follow-up turns home: the KV router's
score trades cached blocks against load (short prompts lose), and its index only learns a
worker's blocks after the worker publishes KV events (bursts of turns get spread). JSEW records
the prefix in its shadow index at routing time.

## Notes

- A Rust in-process variant as a custom worker-selection policy (see
  [`../custom-policy-example`](../custom-policy-example/README.md)) is the natural next step;
  Dynamo exposes device/host/disk overlap, active prefill tokens and decode cost per candidate,
  which cover the score above except the attained-service decode estimator.
- The proxy tokenizes once more than the frontend does when requests are not pre-tokenized.
