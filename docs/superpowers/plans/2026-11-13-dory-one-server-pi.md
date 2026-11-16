# Dory One-Server pi Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pi's coding agent runs on Dory through one llama-server that takes the loop count per request (picked by the trained text decision model) and self-speculates with a loops=1 draft on the same weights.

**Architecture:** Patch NVIDIA's llama.cpp Dory branch so a context allocates cache for `n_loops_max` loops and runs `n_loops` per decode, add a per-request `n_loops` field and a same-weights loops=1 draft to llama-server, then put a small Python gateway in front that picks k from the last user message, adds `n_loops`, and streams the reply back to pi.

**Tech Stack:** C++ (llama.cpp PR #30306 head 1c3cdd2, Metal), Python 3.13 `.venv` (stdlib `http.server`, existing `dory.policy` / `dory.features`), pi coding agent (`~/.pi/agent/models.json`).

**Spec:** `docs/superpowers/specs/2026-11-13-dory-one-server-pi-design.md`

## Global Constraints

- Work in worktree `.worktrees/pi` on branch `pi` from `main`. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- llama.cpp changes: branch `dory-runtime-loops` from 1c3cdd2 in a git worktree of the vendored repo at `vendor/llama.cpp-loops` (main repo path, not inside `.worktrees/pi`), built into `vendor/llama.cpp-loops/build`. Never rebuild or modify `vendor/llama.cpp` (the paused weight-space study uses its binary).
- Python points at the patched binary via `DORY_LLAMA_BIN=<repo>/vendor/llama.cpp-loops/build/bin`.
- Weights: `models/gguf/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (stock, `num_loops` = 4). Never delete anything in `~/.cache/huggingface`, never change `iogpu.wired_limit_mb`.
- At most one llama-server at a time for any measurement; stop servers you start. Don't kill processes you didn't start.
- A request without `n_loops` behaves exactly as today (the GGUF's `num_loops`).
- Defaults: `-np 1`, `-c 16384`, f16 KV, `--jinja`, `--spec-type draft-simple --spec-draft-n-max 4`, thinking disabled (`chat_template_kwargs: {"enable_thinking": false}` unless the client sets it), gateway port 8083, pi provider `dory`, model id `dory`.

## Review Focus

- A k switch on a slot that still holds cache from another k: the slot must reprocess the whole prompt at the new k (stale loop or output-layer KV gives silently wrong text). Pinned by Task 2's `test_switch_matches_fresh`.
- A request with `n_loops` > `n_loops_max`, 0, 3 or a non-integer: the server rejects it with HTTP 400 and does not crash. Pinned by Task 2's `test_bad_n_loops`.
- The server's prompt cache (`--cache-ram` / slot save) restoring an entry made at another k: the entry must be ignored. Pinned by Task 2's `test_prompt_cache_keyed_by_loops`.
- pi sends `content` as a list of parts, or the last message is a `tool` message: the decision must still read the last `user` message's text. Pinned by Task 3's `test_last_user_text_shapes`.
- The gateway with the client disconnecting mid-stream: the upstream request is closed and the gateway keeps serving. Pinned by Task 3's `test_client_disconnect`.

---

### Task 1: Runtime loop count in libllama

**Files:**
- Create (outside our repo): branch + worktree `vendor/llama.cpp-loops`
- Modify (in `vendor/llama.cpp-loops`): `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-graph.h` (`allow_reuse`), `src/llama-model.cpp` (Dory KV filter in `create_memory`), `src/models/dory.cpp`, `src/models/models.h`
- Create: `vendor/llama.cpp-loops/tests/test-dory-loops.cpp` (+ `tests/CMakeLists.txt` entry)

**Interfaces:**
- Produces (C API):
  - `llama_context_params.n_loops_max` (`int32_t`, 0 = model's `num_loops`). The Dory KV filter skips virtual layers whose loop index ≥ `n_loops_max`.
  - `int32_t llama_set_n_loops(struct llama_context * ctx, int32_t n_loops)`: returns 0, or -1 if `n_loops < 1 || n_loops > n_loops_max` or the model is not Dory. It does not touch memory; the caller clears sequences when it changes k.
  - `int32_t llama_get_n_loops(const struct llama_context * ctx)`.
- Dory graph: walks input layers, the first `n_loops` recurrent blocks, then the output layers (virtual indices `n_input + n_recurrent*model.n_loops ...`). Each virtual layer keeps its own `cache_layer[il]`.
- Graph reuse: `llm_graph_params::allow_reuse` returns false when `n_loops` differs.

- [ ] **Step 1: Set up the branch and worktree**

```bash
cd vendor/llama.cpp && git branch dory-runtime-loops 1c3cdd2 && git worktree add ../llama.cpp-loops dory-runtime-loops
cd ../llama.cpp-loops && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
```

- [ ] **Step 2: Write the failing test** `tests/test-dory-loops.cpp`. It takes the GGUF path as argv[1], uses a fixed 64-token prompt, and for each k in {1, 2, 4}:
  - context A: model loaded with KV override `dory.num_loops=int:k`, `n_loops_max` 0;
  - context B: stock model (`num_loops` 4), `llama_set_n_loops(ctx, k)`.
  It checks that the last-token logits from one decode of the prompt match (max abs diff ≤ 1e-3), then that 32 greedy tokens are identical. It also checks that `llama_set_n_loops(ctx, 3)` returns 0 (3 ≤ 4 is allowed in the C API; the server restricts k), `llama_set_n_loops(ctx, 5)` returns -1, and `llama_set_n_loops(ctx, 0)` returns -1. Last, context C with `n_loops_max = 1` reports a KV buffer smaller than B's (`llama_state_get_size` or the memory breakdown).

- [ ] **Step 3:** `cmake --build build -j 10 --target test-dory-loops && build/bin/test-dory-loops <repo>/models/gguf/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`. Expected: build error (`llama_set_n_loops` undeclared).

- [ ] **Step 4:** Implement per the Interfaces block. Add `n_loops` and `n_loops_max` to `llama_cparams`. `llama_model_dory` keeps `n_loops` as the allocated maximum and the graph reads the context's current `n_loops`. Only the stock GGUF needs to work. Whether `cparams` reaches the graph builder through `llm_graph_params` is the implementer's call; keep it in one place.

- [ ] **Step 5:** Run the test again. Expected: `PASS` for k = 1, 2, 4. Then run `build/bin/llama-cli -m <stock gguf> -p "2+2=" -n 8 -no-cnv --temp 0` and check it matches `vendor/llama.cpp/build/bin/llama-cli` output (no `n_loops` set = unchanged behavior).

- [ ] **Step 6: Commit** in `vendor/llama.cpp-loops` on `dory-runtime-loops`. Then, in our repo (`.worktrees/pi`), write `patches/llama-dory-runtime-loops.patch` (`git -C vendor/llama.cpp-loops diff 1c3cdd2..HEAD`) and commit it.

### Task 2: llama-server per-request `n_loops` and same-weights self-draft

**Files:**
- Modify (in `vendor/llama.cpp-loops`): `tools/server/server-task.cpp` (params from JSON), `tools/server/server-context.cpp` (slot loop count, switch, prompt cache key, `/props`), `common/common.h`, `common/arg.cpp` (`--spec-draft-loops N`), `common/speculative.cpp` (same-weights draft context)
- Create (our repo): `tests/test_dory_server_loops.py` (integration; skips unless `DORY_LLAMA_BIN` points at a binary whose `/props` has `n_loops_max`)
- Modify (our repo): `scripts/setup_llama.sh` (optional second build: `--loops` applies `patches/llama-dory-runtime-loops.patch` into `vendor/llama.cpp-loops`), `dory/server.py` (`server_args(..., runtime_loops: bool = False, spec_draft_loops: int | None = None)`)

**Interfaces:**
- Consumes: Task 1 `llama_set_n_loops`, `llama_get_n_loops`, `n_loops_max`.
- Produces (HTTP):
  - `n_loops` (int, optional) on `/completion`, `/v1/completions` and `/v1/chat/completions`. Allowed values are 1, 2 and 4, each ≤ `n_loops_max`; anything else gets HTTP 400 `{"error": {"message": "n_loops must be one of 1, 2, 4 and <= n_loops_max"}}`.
  - The response's `timings` object gains `"n_loops": <int>`.
  - `/props` gains `"n_loops_max": <int>`.
- Produces (CLI): `--spec-draft-loops N` creates the draft context on the target model with `n_loops_max = n_loops = N` (no `-md` file) and uses the draft-simple drafter. Speculation is skipped for a slot whose `n_loops <= N`.
- Server behavior:
  - Each slot records the `n_loops` its cached tokens were computed at.
  - On a request whose `n_loops` differs: clear that slot's sequence in the target memory (draft memory is kept), reprocess the full prompt, set the slot's loop record.
  - A request whose `n_loops` differs from the context's current value waits until no other slot is mid-decode, then calls `llama_set_n_loops`. Batches never mix k.
  - Prompt-cache entries store `n_loops`; a mismatch is a cache miss.
- Produces (Python): `server_args(gguf, k, port, ..., runtime_loops=True, spec_draft_loops=1)` omits `--override-kv`, adds `--jinja --spec-type draft-simple --spec-draft-n-max 4 --spec-draft-loops 1`.

- [ ] **Step 1: Write the failing tests** in `tests/test_dory_server_loops.py` (one server for the module: patched binary, stock Q5_K_M, `-np 1 -c 16384 --jinja`, no draft unless stated):
  - `test_exact_vs_per_k(k)` for k in 1, 2, 4: for the first 20 items of `evalsets.load_split("wsp_screen")`, the greedy output with `n_loops=k` equals a reference run of the stock `vendor/llama.cpp/build/bin/llama-server` at `-np 1` with `--override-kv dory.num_loops=int:k`. Run the reference before the patched server starts, one server at a time, and cache it in `results/pi/ref_k{k}.jsonl`. Don't reuse the old sweep files: they ran at 4 slots, which is not bit-deterministic. Use the same `max_tokens` (`MAX_TOKENS[source]`) and `enable_thinking: false`. All 20 must be identical.
  - `test_switch_matches_fresh`: send prompt P at k=1, then the same P plus an appended assistant turn and a new user turn at k=4. The second output must equal a fresh server's output for the second request at k=4.
  - `test_bad_n_loops`: `n_loops` in (0, 3, 5, "2") → 400, and the server still answers a valid request afterwards.
  - `test_prompt_cache_keyed_by_loops`: with `--cache-ram 2048`, run P at k=2, then Q at k=2 (evicts P from the slot), then P at k=4. The k=4 output must equal a fresh k=4 run (no reuse of the k=2 entry).
  - `test_spec_identical`: restart with `--spec-type draft-simple --spec-draft-n-max 4 --spec-draft-loops 1`. 10 screen items at k=4 and 10 at k=2 must match the non-speculative outputs, and `timings.draft_n > 0` for the k=4 ones.
  - `test_timings_and_props`: `timings.n_loops == k`; `/props` `n_loops_max == 4`.
  - `test_memory`: with the spec server at `-c 16384`, after one k=4 request, the server's resident set (`ps -o rss=`, which includes the mapped weights and Metal buffers on unified memory) is ≤ 9 GB. Print it alongside the log's `llama_kv_cache` and `compute buffer` sizes.

- [ ] **Step 2:** `DORY_LLAMA_BIN=$PWD/../../vendor/llama.cpp-loops/build/bin .venv/bin/python -m pytest tests/test_dory_server_loops.py -v`. Expected: the HTTP tests FAIL (`n_loops` ignored, no `n_loops_max` in `/props`).

- [ ] **Step 3:** Implement per the Interfaces block. Draft context: follow the existing MTP branch in `common_speculative_init_result` (`llama_init_from_model(model_tgt, cparams)`), with `cparams.n_loops_max = N` and then `llama_set_n_loops(ctx_dft, N)`. If draft-simple assumes a separate `model_dft` (vocab checks), point those checks at the target model.

- [ ] **Step 4:** Rebuild `llama-server` in `vendor/llama.cpp-loops/build` and rerun the tests. Expected: all PASS; `test_memory` prints ≤ 9 GB.

- [ ] **Step 5: Commit** the llama.cpp branch, regenerate `patches/llama-dory-runtime-loops.patch`, and commit it in our repo with the tests, `setup_llama.sh` and `dory/server.py`. The full `.venv` suite must pass, with the integration tests skipping when `DORY_LLAMA_BIN` is unset.

### Task 3: Gateway (OpenAI-compatible, decision model in front)

**Files:**
- Create: `dory/gateway.py`, `tests/test_gateway.py`
- Modify: `dory/policy.py` (extract `decide_k(policy, text) -> int`, reused by `Router._decide`), `dory/router.py`

**Interfaces:**
- Consumes: Task 2 HTTP `n_loops`, `server_args(..., runtime_loops=True, spec_draft_loops=1)`; `dory.policy.load_policy`, `choose_k`, `proba_for_ks`; `dory.features.text_features`; `dory.server.Server`.
- Produces:
  - `decide_k(policy: dict, text: str) -> int` in `dory/policy.py` (text features → k with the policy's tau and code floor).
  - `last_user_text(messages: list[dict]) -> str` in `dory/gateway.py` (content as a string or a list of `{"type": "text", "text": ...}` parts; "" if there is no user message).
  - `python -m dory.gateway [--port 8083] [--policy results/policy_text.joblib] [--gguf models/dory-q5.gguf] [--ctx 16384]`. It starts the one llama-server (`-np 1`, runtime loops, self-draft) and serves:
    - `GET /v1/models` → `{"data": [{"id": "dory", "object": "model"}]}`
    - `POST /v1/chat/completions`, stream and non-stream. It sets `n_loops = decide_k(policy, last_user_text(messages))`, adds `chat_template_kwargs.enable_thinking=false` if absent, and forwards to the server. The response body or SSE is passed through unchanged.
  - Decisions are memoized by text (`functools.lru_cache(maxsize=256)` on the text), so tool steps skip the ~60 ms classifier.
  - One log line per request to stderr and `results/logs/gateway.log`: time, k, prompt tokens, generated tokens, tok/s, draft acceptance.
  - SIGINT/SIGTERM stop the llama-server.

- [ ] **Step 1: Write the failing tests** (`tests/test_gateway.py`, no GPU: monkeypatch `Server` and point the upstream at a stub HTTP server that echoes the body it received):
  - `test_last_user_text_shapes`: a string content; a parts list; a last message with role `tool` after a user message (returns that user text); no user message → "".
  - `test_k_follows_last_user_message`: with `decide_k` monkeypatched to `lambda p, t: 4 if "hard" in t else 1`, a request whose last user message contains "hard" reaches the upstream with `n_loops == 4`. A follow-up request with an extra assistant `tool_calls` turn and a `tool` result, same last user message, also gets 4.
  - `test_enable_thinking_default`: the upstream sees `chat_template_kwargs.enable_thinking == false` when absent, and the client's value when present.
  - `test_stream_passthrough`: an upstream SSE stream of 3 chunks + `[DONE]` reaches the client byte-identical.
  - `test_client_disconnect`: the client closes mid-stream; the gateway closes the upstream response and then serves a second request.
  - `test_models_endpoint`: `/v1/models` lists `dory`.
  - `test_decide_k_matches_router` (in `tests/test_policy.py`, skips without the classifier): for 5 train prompts, `decide_k` equals `Router._decide`'s k with the same policy.

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_gateway.py -v`. Expected: FAIL (`dory.gateway` missing).

- [ ] **Step 3:** Implement with `http.server.ThreadingHTTPServer` and `urllib.request`. Stream by reading the upstream in chunks and writing each chunk with a flush. No new dependencies.

- [ ] **Step 4:** Run the tests and the full `.venv` suite. Expected: PASS.

- [ ] **Step 5: Commit.**

### Task 4: pi provider and end-to-end check

**Files:**
- Create: `scripts/dory_pi.sh` (starts the gateway with `DORY_LLAMA_BIN` set; refuses to start if port 8083 is taken or another llama-server is running), `docs/dory-pi.md` (start/stop, what k means, the log line)
- Modify (outside the repo, with a timestamped backup next to it): `~/.pi/agent/models.json`, adding provider `dory`:

```json
"dory": {
  "baseUrl": "http://127.0.0.1:8083/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "compat": { "supportsDeveloperRole": false },
  "models": [{ "id": "dory", "contextWindow": 16384, "maxTokens": 4096 }]
}
```

**Interfaces:**
- Consumes: Task 3 `python -m dory.gateway`.

- [ ] **Step 1: Tool-call smoke test (no pi):** start `scripts/dory_pi.sh`. POST a chat request with one tool (`write_file(path, content)`) and the user message "Create hello.txt containing hi". Expected: the response has `choices[0].message.tool_calls[0].function.name == "write_file"` and JSON arguments with `path` and `content`. If llama-server returns the XML as plain `content` instead, fix the template detection in Task 2's server (Dory's template uses the Qwen3-Coder `<tool_call><function=...>` format) before going on.

- [ ] **Step 2: Add the pi provider:** back up `~/.pi/agent/models.json` to `models.json.bak-dory-<timestamp>`, add the block above, and validate the JSON (`python -m json.tool`).

- [ ] **Step 3: End to end:** in a temp dir, run pi non-interactively with provider `dory` and model `dory` (see `pi --help` for the print/one-shot flag) on: "Create a file add.py with a function add(a, b) that returns a + b, then run `python -c 'import add; print(add.add(2, 3))'`". Expected: `add.py` exists, the command printed 5, and `results/logs/gateway.log` shows the requests with one k across the task's steps.

- [ ] **Step 4:** Stop the gateway (check no llama-server is left running) and commit `scripts/dory_pi.sh`, `docs/dory-pi.md`, plus a transcript excerpt of Steps 1 and 3 in `docs/dory-pi.md`.
