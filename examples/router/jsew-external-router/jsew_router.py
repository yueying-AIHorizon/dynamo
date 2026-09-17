# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Join-the-Shortest-Effective-Workload (JSEW) routing policy.

JSEW is a cache-aware, load-aware worker-selection rule for a fleet of LLM
backends that each keep a prefix (KV) cache. For a request with block hashes
``h`` and ``n`` input tokens it picks

    argmin_k  ( W_k + (n - hit_k(h)) + E[D] ) / c_k

where ``W_k`` is the router's estimate of the work already queued at worker
``k`` (remaining prefill tokens of in-flight requests plus the expected
remaining decode of each, conditional on the tokens it has produced so far),
``hit_k(h)`` is the number of prompt tokens the router believes worker ``k``
already holds in its prefix cache, ``E[D]`` is the mean decode length, and
``c_k`` is a relative capacity weight. A hysteresis band keeps a request on
its session's home worker unless another worker is better by more than a
fraction ``eta``.

The router keeps a *shadow* prefix index per worker: an LRU over block hashes
sized to the worker's KV cache, updated at routing time (not when the worker
reports the blocks as stored), so a burst of turns from one session sees its
home immediately. Decode lengths are never predicted per request; the
estimator uses the empirical conditional mean of remaining output given
attained output, refreshed from completed requests.

The module has no Dynamo dependency; :mod:`jsew_proxy` wires it to a Dynamo
frontend running in ``--router-mode direct``.
"""

from __future__ import annotations

import bisect
import hashlib
import zlib
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

DEFAULT_BLOCK_SIZE = 64


def block_hashes(
    tokens: Sequence[int], block_size: int = DEFAULT_BLOCK_SIZE
) -> list[int]:
    """Prefix-chained block hashes of a token sequence.

    Block ``i`` hashes its own tokens together with the hash of block ``i-1``,
    so two sequences share block id ``i`` only if they share the whole prefix
    up to and including block ``i``. A trailing partial block is hashed too.
    """
    out: list[int] = []
    prev = b""
    for start in range(0, len(tokens), block_size):
        chunk = tokens[start : start + block_size]
        h = hashlib.blake2b(digest_size=8)
        h.update(prev)
        h.update(",".join(map(str, chunk)).encode())
        prev = h.digest()
        out.append(int.from_bytes(prev, "big"))
    return out


def _block_tokens(index: int, input_len: int, block_size: int) -> int:
    return max(0, min(block_size, input_len - index * block_size))


class RemainingCurve:
    """``r(s) = E[D - s | D > s]`` on a grid, learned from completed requests."""

    def __init__(self, prior_mean: float = 300.0, grid: int = 32, window: int = 5000):
        self.prior = prior_mean
        self.grid = grid
        self.window = window
        self.done: list[int] = []
        self.table: list[float] | None = None
        self.mean = prior_mean

    def add(self, decode_len: int) -> None:
        self.done.append(int(decode_len))
        if len(self.done) % 200 == 0:
            self.refresh()

    def refresh(self) -> None:
        if len(self.done) < 50:
            return
        d = sorted(self.done[-self.window :])
        n = len(d)
        self.mean = sum(d) / n
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + d[i]
        table: list[float] = []
        for s in range(0, d[-1] + self.grid, self.grid):
            i = bisect.bisect_right(d, s)
            cnt = n - i
            if cnt >= 5:
                table.append(max(suffix[i] / cnt - s, 1.0))
            else:
                table.append(table[-1] if table else self.prior)
        self.table = table

    def __call__(self, attained: int) -> float:
        if self.table is None:
            return max(self.prior - attained, 1.0)
        return self.table[min(attained // self.grid, len(self.table) - 1)]


@dataclass
class InFlight:
    prefill_remaining: int
    out_est: float
    decoded: int = 0


@dataclass
class ShadowWorker:
    """What the router can know about one worker."""

    worker_id: int
    cache_tokens: int
    weight: float = 1.0
    block_size: int = DEFAULT_BLOCK_SIZE
    lru: OrderedDict[int, int] = field(default_factory=OrderedDict)
    used: int = 0
    inflight: dict[str, InFlight] = field(default_factory=dict)

    def match(self, hashes: Sequence[int]) -> int:
        """Length of the longest cached prefix, in blocks."""
        k = 0
        for h in hashes:
            if h in self.lru:
                k += 1
            else:
                break
        return k

    def hit_tokens(self, hashes: Sequence[int], input_len: int) -> int:
        k = self.match(hashes)
        return sum(_block_tokens(i, input_len, self.block_size) for i in range(k))

    def insert(self, hashes: Sequence[int], input_len: int) -> None:
        """Record that this worker will hold the prefix (LRU, roots newest)."""
        k = self.match(hashes)
        need = sum(
            _block_tokens(i, input_len, self.block_size) for i in range(k, len(hashes))
        )
        while self.used + need > self.cache_tokens and self.lru:
            _, t = self.lru.popitem(last=False)
            self.used -= t
        for i in range(len(hashes) - 1, -1, -1):
            t = _block_tokens(i, input_len, self.block_size)
            if t > 0 and hashes[i] not in self.lru:
                self.lru[hashes[i]] = t
                self.used += t
        for i in range(k - 1, -1, -1):
            if hashes[i] in self.lru:
                self.lru.move_to_end(hashes[i])

    def workload(self, rcurve: RemainingCurve) -> float:
        w = 0.0
        for st in self.inflight.values():
            if st.decoded == 0:
                w += st.prefill_remaining + st.out_est
            else:
                w += rcurve(st.decoded)
        return w


class JsewRouter:
    """JSEW worker selection with a shadow prefix index per worker."""

    def __init__(
        self,
        worker_ids: Iterable[int],
        cache_tokens: int,
        hysteresis: float = 0.05,
        weights: dict[int, float] | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        prior_decode_mean: float = 300.0,
    ):
        self.cache_tokens = cache_tokens
        self.eta = hysteresis
        self.block_size = block_size
        self.weights = dict(weights or {})
        self.rcurve = RemainingCurve(prior_mean=prior_decode_mean)
        self.workers: dict[int, ShadowWorker] = {}
        self._rid_worker: dict[str, int] = {}
        for w in worker_ids:
            self.add_worker(int(w))

    # ---- fleet membership -------------------------------------------------
    def add_worker(self, worker_id: int) -> None:
        if worker_id not in self.workers:
            self.workers[worker_id] = ShadowWorker(
                worker_id,
                self.cache_tokens,
                self.weights.get(worker_id, 1.0),
                self.block_size,
            )

    def remove_worker(self, worker_id: int) -> None:
        self.workers.pop(worker_id, None)

    def sync_workers(self, worker_ids: Iterable[int]) -> None:
        live = {int(w) for w in worker_ids}
        for w in list(self.workers):
            if w not in live:
                self.remove_worker(w)
        for w in live:
            self.add_worker(w)

    # ---- routing ----------------------------------------------------------
    def scores(self, hashes: Sequence[int], input_len: int) -> dict[int, float]:
        out = {}
        for w, sw in self.workers.items():
            uncached = input_len - sw.hit_tokens(hashes, input_len)
            out[w] = (
                sw.workload(self.rcurve) + uncached + self.rcurve.mean
            ) / sw.weight
        return out

    def home(self, session_key: object) -> int | None:
        if not self.workers:
            return None
        ids = sorted(self.workers)
        return ids[zlib.crc32(str(session_key).encode()) % len(ids)]

    def route(
        self,
        hashes: Sequence[int],
        input_len: int,
        session_key: object | None = None,
    ) -> int:
        """Pick a worker; ``session_key`` gives the request a hash home."""
        if not self.workers:
            raise RuntimeError("JsewRouter has no workers")
        sc = self.scores(hashes, input_len)
        best = min(sc, key=sc.get)
        if session_key is None:
            return best
        home = self.home(session_key)
        if home is not None and sc[home] <= sc[best] + self.eta * max(sc[best], 1.0):
            return home
        return best

    def admit(
        self, rid: str, worker_id: int, hashes: Sequence[int], input_len: int
    ) -> None:
        """Account the request at ``worker_id`` and record its prefix."""
        sw = self.workers[worker_id]
        sw.inflight[rid] = InFlight(
            prefill_remaining=input_len - sw.hit_tokens(hashes, input_len),
            out_est=self.rcurve.mean,
        )
        sw.insert(hashes, input_len)
        self._rid_worker[rid] = worker_id

    def on_token(self, rid: str) -> None:
        w = self._rid_worker.get(rid)
        if w is None:
            return
        sw = self.workers.get(w)
        if sw is not None and rid in sw.inflight:
            sw.inflight[rid].decoded += 1

    def complete(self, rid: str, decode_len: int | None = None) -> None:
        w = self._rid_worker.pop(rid, None)
        if w is None:
            return
        sw = self.workers.get(w)
        if sw is None:
            return
        info = sw.inflight.pop(rid, None)
        n = decode_len if decode_len is not None else (info.decoded if info else 0)
        if n:
            self.rcurve.add(n)

    # ---- introspection ----------------------------------------------------
    def stats(self) -> dict[str, object]:
        return {
            "workers": {
                str(w): {
                    "inflight": len(sw.inflight),
                    "workload_tokens": round(sw.workload(self.rcurve)),
                    "shadow_cache_tokens": sw.used,
                    "shadow_blocks": len(sw.lru),
                }
                for w, sw in sorted(self.workers.items())
            },
            "decode_mean": round(self.rcurve.mean, 1),
            "completed": len(self.rcurve.done),
        }
