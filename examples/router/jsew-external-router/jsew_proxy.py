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

"""OpenAI-compatible proxy that routes with JSEW and pins requests to workers.

    client -> jsew_proxy (this file) -> dynamo.frontend --router-mode direct

For every /v1/completions or /v1/chat/completions request the proxy tokenizes
the prompt (or takes ``nvext.token_data``), computes prefix-chained block
hashes, asks :class:`jsew_router.JsewRouter` for a worker, and forwards the
request to the Dynamo frontend with the ``x-dynamo-worker-instance-id``
header. The frontend must run in ``--router-mode direct``: in the other
modes the header is ignored and the frontend routes on its own.

Streaming responses are relayed chunk by chunk; the proxy counts produced
tokens to feed the router's attained-service estimator and records the
completion when the stream ends. Worker membership comes from ``--workers``
or, with ``--discover``, from the Dynamo runtime's discovery
(``instance_ids()`` of the backend generate endpoint), polled periodically.

Example::

    python -m dynamo.frontend --http-port 8000 --router-mode direct
    python jsew_proxy.py --frontend http://127.0.0.1:8000 --port 8010 \
        --tokenizer Qwen/Qwen2.5-7B-Instruct --cache-tokens 2700000 --discover
    curl localhost:8010/v1/completions -H 'content-type: application/json' \
        -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "Hi", "stream": true}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid

import aiohttp
from aiohttp import web
from jsew_router import JsewRouter, block_hashes

log = logging.getLogger("jsew_proxy")
PIN_HEADER = "x-dynamo-worker-instance-id"
SESSION_HEADER = "x-jsew-session"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "trailer",
    "upgrade",
    "host",
    "content-length",
}


class Proxy:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.router = JsewRouter(
            [int(w) for w in args.workers.split(",") if w],
            cache_tokens=args.cache_tokens,
            hysteresis=args.hysteresis,
            block_size=args.block_size,
        )
        self.tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer  # optional dependency

            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.session: aiohttp.ClientSession | None = None
        self._runtime = None
        self._client = None

    # ---- worker discovery -------------------------------------------------
    async def start_discovery(self) -> None:
        from dynamo.runtime import DistributedRuntime

        loop = asyncio.get_running_loop()
        self._runtime = DistributedRuntime(
            loop, self.args.discovery_backend, self.args.request_plane
        )
        endpoint = self._runtime.endpoint(self.args.endpoint)
        self._client = await endpoint.client()
        await self._client.wait_for_instances()
        self.router.sync_workers(self._client.instance_ids())
        log.info("discovered workers: %s", sorted(self.router.workers))
        asyncio.create_task(self._poll_discovery())

    async def _poll_discovery(self) -> None:
        while True:
            await asyncio.sleep(self.args.discovery_interval)
            try:
                ids = self._client.instance_ids()
                if set(ids) != set(self.router.workers):
                    self.router.sync_workers(ids)
                    log.info("workers now: %s", sorted(self.router.workers))
            except Exception as exc:  # noqa: BLE001
                log.warning("discovery poll failed: %r", exc)

    # ---- tokenization -----------------------------------------------------
    def tokens_for(self, path: str, body: dict) -> list[int]:
        nvext = body.get("nvext") or {}
        if nvext.get("token_data"):
            return [int(t) for t in nvext["token_data"]]
        if self.tokenizer is None:
            raise web.HTTPBadRequest(
                text="request has no nvext.token_data and no --tokenizer is set"
            )
        if path.endswith("/chat/completions"):
            msgs = body.get("messages") or []
            if getattr(self.tokenizer, "chat_template", None):
                return list(
                    self.tokenizer.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=True
                    )
                )
            text = "\n".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in msgs
            )
        else:
            p = body.get("prompt", "")
            text = p if isinstance(p, str) else "\n".join(map(str, p))
        return list(self.tokenizer.encode(text))

    # ---- request handling -------------------------------------------------
    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        tokens = self.tokens_for(request.path, body)
        hashes = block_hashes(tokens, self.args.block_size)
        nvext = body.get("nvext") or {}
        session_key = request.headers.get(SESSION_HEADER) or nvext.get("cache_salt")
        if session_key is None and hashes:
            # Default home: the second block; the first is often a shared
            # system prompt.
            session_key = hashes[1] if len(hashes) > 1 else hashes[0]
        rid = uuid.uuid4().hex
        worker = self.router.route(hashes, len(tokens), session_key)
        self.router.admit(rid, worker, hashes, len(tokens))

        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP
        }
        headers[PIN_HEADER] = str(worker)
        headers["content-type"] = "application/json"
        url = self.args.frontend.rstrip("/") + request.path
        ntok: int | None = None
        try:
            async with self.session.post(
                url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=None),
            ) as upstream:
                resp = web.StreamResponse(status=upstream.status)
                for k, v in upstream.headers.items():
                    if k.lower() not in HOP_BY_HOP:
                        resp.headers[k] = v
                resp.headers["x-jsew-worker"] = str(worker)
                await resp.prepare(request)
                ntok = 0
                if body.get("stream"):
                    async for raw in upstream.content:
                        await resp.write(raw)
                        line = raw.strip()
                        if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                            ntok += _tokens_in_chunk(line[5:])
                            self.router.on_token(rid)
                else:
                    data = await upstream.read()
                    await resp.write(data)
                    try:
                        usage = json.loads(data).get("usage") or {}
                        ntok = int(usage.get("completion_tokens", 0))
                    except Exception:  # noqa: BLE001
                        ntok = 0
                await resp.write_eof()
                return resp
        finally:
            self.router.complete(rid, ntok)

    async def passthrough(self, request: web.Request) -> web.Response:
        url = self.args.frontend.rstrip("/") + request.path_qs
        async with self.session.get(url) as upstream:
            return web.Response(
                status=upstream.status,
                body=await upstream.read(),
                content_type=upstream.content_type,
            )

    async def stats(self, request: web.Request) -> web.Response:
        return web.json_response(self.router.stats())


def _tokens_in_chunk(data: bytes) -> int:
    try:
        j = json.loads(data)
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for ch in j.get("choices") or []:
        if ch.get("text") or (ch.get("delta") or {}).get("content"):
            n += 1
    return n


def build_app(proxy: Proxy) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/v1/completions", proxy.handle)
    app.router.add_post("/v1/chat/completions", proxy.handle)
    app.router.add_get("/v1/models", proxy.passthrough)
    app.router.add_get("/health", proxy.passthrough)
    app.router.add_get("/jsew/stats", proxy.stats)

    async def on_startup(app: web.Application) -> None:
        proxy.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
        if proxy.args.discover:
            await proxy.start_discovery()
        if not proxy.router.workers:
            raise SystemExit("no workers: pass --workers ids or --discover")

    async def on_cleanup(app: web.Application) -> None:
        await proxy.session.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--frontend",
        default="http://127.0.0.1:8000",
        help="Dynamo frontend URL (must run with --router-mode direct)",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--workers", default="", help="comma-separated worker instance ids")
    ap.add_argument(
        "--discover",
        action="store_true",
        help="discover workers via the Dynamo runtime",
    )
    ap.add_argument(
        "--discovery-backend",
        default="file",
        choices=["file", "etcd", "kubernetes", "mem"],
    )
    ap.add_argument("--request-plane", default="tcp", choices=["tcp", "nats"])
    ap.add_argument(
        "--endpoint",
        default="dynamo/backend/generate",
        help="namespace/component/endpoint of the workers",
    )
    ap.add_argument("--discovery-interval", type=float, default=5.0)
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer for requests without nvext.token_data",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="tokens per hash block (match the workers' KV block size)",
    )
    ap.add_argument(
        "--cache-tokens",
        type=int,
        required=True,
        help="KV cache tokens per worker (shadow index size)",
    )
    ap.add_argument(
        "--hysteresis",
        type=float,
        default=0.05,
        help="stay home unless another worker is better by this fraction",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    web.run_app(build_app(Proxy(args)), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
