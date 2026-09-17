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

"""Unit tests for the JSEW policy (``pytest examples/router/jsew-external-router``)."""

import random

import pytest
from jsew_router import JsewRouter, RemainingCurve, ShadowWorker, block_hashes


def test_block_hashes_are_prefix_chained():
    a = list(range(1, 300))
    b = list(range(1, 300))
    c = list(range(1, 200)) + [999] + list(range(201, 300))
    ha, hb, hc = block_hashes(a, 64), block_hashes(b, 64), block_hashes(c, 64)
    assert ha == hb
    assert ha[:3] == hc[:3]  # first three 64-token blocks agree
    assert ha[3:] != hc[3:]  # divergence at block 3 changes every later id


def test_shadow_worker_lru_evicts_leaves_first():
    sw = ShadowWorker(1, cache_tokens=128, block_size=64)
    sw.insert([1, 2], 128)
    assert sw.match([1, 2]) == 2 and sw.used == 128
    sw.insert([1, 3], 128)  # shares the root, needs one more block: evicts leaf 2
    assert sw.match([1, 3]) == 2
    assert sw.match([1, 2]) == 1


def test_same_prefix_routes_to_same_worker_under_light_load():
    r = JsewRouter([10, 20, 30, 40], cache_tokens=10_000, block_size=64)
    h = block_hashes(list(range(640)), 64)
    first = r.route(h, 640, session_key="s1")
    r.admit("a", first, h, 640)
    r.complete("a", 50)
    longer = block_hashes(list(range(640)) + list(range(64)), 64)
    assert r.route(longer, 704, session_key="s1") == first


def test_overloaded_home_is_left():
    r = JsewRouter([1, 2], cache_tokens=10_000, hysteresis=0.05, block_size=64)
    h = block_hashes(list(range(640)), 64)
    home = r.route(h, 640, session_key="s")
    for i in range(50):  # pile work on the home worker
        other_prompt = block_hashes(list(range(5000 + i, 5000 + i + 640)), 64)
        r.admit(f"q{i}", home, other_prompt, 640)
    other = ({1, 2} - {home}).pop()
    assert r.route(h, 640, session_key="s") == other


def test_hysteresis_keeps_home_when_close():
    r = JsewRouter([1, 2], cache_tokens=10_000, hysteresis=0.5, block_size=64)
    h = block_hashes(list(range(640)), 64)
    home = r.home("s")
    other = ({1, 2} - {home}).pop()
    r.admit("x", home, block_hashes(list(range(9000, 9064)), 64), 64)
    assert r.route(h, 640, session_key="s") == home
    assert r.scores(h, 640)[other] < r.scores(h, 640)[home]


def test_remaining_curve_is_conditional_mean():
    rc = RemainingCurve(prior_mean=100.0, grid=32)
    rng = random.Random(0)
    for _ in range(600):
        rc.add(int(rng.expovariate(1 / 200.0)) + 1)
    rc.refresh()
    assert abs(rc.mean - 200) < 30
    # near-memoryless: remaining after 320 attained is still of order the mean
    assert 100 < rc(320) < 400


def test_sync_workers_adds_and_removes():
    r = JsewRouter([1, 2, 3], cache_tokens=100)
    r.sync_workers([2, 3, 4])
    assert sorted(r.workers) == [2, 3, 4]
    with pytest.raises(RuntimeError):
        JsewRouter([], cache_tokens=100).route([1], 64)
