# Dory in pi: one server, per-request loops — Design

## Intent

Use Dory1.0-4B as the model behind the pi coding agent (file edits, shell commands), with the trained text
decision model choosing the loop count k ∈ {1, 2, 4} per request and self-speculative decoding speeding up
k = 2 and k = 4. Run all of it from one llama-server process holding one copy of the weights.

## Decisions (the user's)

- Agentic coding in pi: tool calls for edits and shell, not chat only.
- The trained text decision model (`results/policy_text.joblib`, tau 0.4326, code floor) picks k. Not the
  two-rule heuristic, not fixed k. The user accepts that its picks on coding sessions are untested.
- Cache option 1 only: each pi session stays on the same slot so the prompt cache is kept; no background
  warm-up of other loop counts.
- One server, not one per k. NVIDIA's branch fixes the loop count at load, so llama.cpp gets patched to take
  the loop count per request.
- Other model servers (dotsLab, Qwen on 8080, moestream, dllm) are off while Dory runs.
- Option 3 (one cache shared across loop counts) is a later project, not this one.

## Decisions (the controller's, during design)

- **Decision rule:** k is a function of the text of the last `user` message in the request. Tool-result steps
  inside one task carry the same last user message, so they get the same k with no state. A changed k only
  happens when the user types.
- **k switch:** the server clears that slot's cache and reprocesses the session at the new k. Correct but slow
  for long sessions; accepted (switches only happen on a new user message).
- **Loop count in llama.cpp:** the context allocates cache for `n_loops_max` loops (default: the GGUF's
  `num_loops`, 4 for the stock file) and runs `n_loops ≤ n_loops_max` per decode. Requests carry `n_loops`.
- **Self-draft:** the draft context is created on the target model's weights with loops = 1 (precedent: the
  MTP draft context in `common/speculative.cpp`). Used only when the slot's k > 1. The draft's cache does not
  depend on the target's k, so it survives k switches.
- **Exactness bar:** at each k, greedy outputs from the patched server with per-request loops must match the
  current per-k setup (stock GGUF with `--override-kv dory.num_loops=int:k`) on the 60-item screen set at
  `-np 1`. Speculative outputs must match non-speculative outputs.
- **Defaults:** stock Q5_K_M GGUF, `-np 1`, 16384-token context, f16 KV, `--jinja`, draft-simple n_max 4,
  thinking disabled (matches all our evaluations), gateway on port 8083, pi provider id `dory`, model id
  `dory`.
- **Packaging:** the llama.cpp change lives on branch `dory-runtime-loops` in a separate git worktree of the
  vendored repo (`vendor/llama.cpp-loops`, own build dir) so the paused weight-space work keeps its binary;
  it is exported as `patches/llama-dory-runtime-loops.patch`.

## Goals

- pi can run a coding task with Dory end to end (tool calls parsed into OpenAI `tool_calls`).
- One llama-server process; resident memory at 16k context ≤ 9 GB (vs ~17 GB for the multi-server version).
- Outputs identical to the per-k setup at the same k.
- The k used for each request is logged.

## Non-goals

- Sharing one cache across loop counts (option 3).
- Background warm-up of other loop counts.
- Re-tuning the decision model for coding sessions.
- Concurrent requests with different k at full speed (a request with a different k waits for the others).

## Anti-goals

- Silently wrong outputs after a k switch (stale cache entries from another k).
- Breaking the stock behavior: a request without `n_loops` must behave exactly as today.
- Changing the binary used by the paused weight-space study.
