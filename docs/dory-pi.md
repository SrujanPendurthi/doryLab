# Dory in pi

Run from a checkout with `.venv`, the text policy at
`results/policy_text.joblib`, the stock Dory Q5 weights linked as
`models/dory-q5.gguf`. Before first use, build the patched binary at
`vendor/llama.cpp-loops/build/bin`:

```sh
./scripts/setup_llama.sh --loops
```

Then start the gateway:

```sh
./scripts/dory_pi.sh
```

The launcher works from any current directory. It resolves `vendor` to the
main repository when used in a worktree and exports the absolute
`DORY_LLAMA_BIN` path. It refuses to start if any `llama-server` is running or
port 8083 has a listener; the error names the reason and exits nonzero.
Keep other model servers off while using Dory. Model and classifier loading
take time. Once ready, `curl http://127.0.0.1:8083/v1/models` lists `dory`.

The gateway runs in the foreground. Stop it with Ctrl-C, or send SIGTERM to
its PID if you started it in the background. Both signals stop its child
`llama-server`. Check for leftovers with:

```sh
pgrep -fl llama-server
pgrep -fl 'python.*dory[.]gateway'
```

No matches is the expected stopped state. Only stop processes you started.

## pi configuration

Add this provider inside `providers` in `~/.pi/agent/models.json`, retaining
the other providers. First copy the file to a timestamped
`models.json.bak-dory-<timestamp>` beside it.

```json
"dory": {
  "baseUrl": "http://127.0.0.1:8083/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "compat": { "supportsDeveloperRole": false },
  "models": [{ "id": "dory", "contextWindow": 24576, "maxTokens": 4096 }]
}
```

Validate with `.venv/bin/python -m json.tool ~/.pi/agent/models.json`. In
another terminal, start pi with `pi --provider dory --model dory`. For a
single prompt use `--print`; `--mode json` includes tool calls and results.
The configured `apiKey: "none"` supplies the custom provider's placeholder
key without a separate environment variable.

For commands that use `python`, put an appropriate virtualenv's `bin`
directory on pi's `PATH` before starting pi. The check below used this
repository's `.venv/bin`. Otherwise this machine has `python3` but no
unqualified `python` command.

## Loop selection and logs

`k` is the number of Dory recurrent loops for the target model: 1, 2, or 4.
The text policy chooses it from the last user message. Decisions are cached
by that text, so tool results and assistant turns keep the same `k` until
the user text changes. One server accepts the loop count per request;
its draft context always uses one loop and shares the model weights.
Defaults are one slot, 24576 context tokens, f16 KV, `--jinja`, draft-simple
with at most four draft tokens, and thinking disabled unless the client
explicitly requests otherwise.

Each completion request writes one line to stderr and
`results/logs/gateway.log`: UTC time, `k`, prompt tokens, generated tokens,
generation tokens/second, and accepted/proposed draft-token ratio. `-` means
the upstream response did not supply a metric (or proposed no draft tokens).
Both streaming and non-streaming requests use the same logging format.

## Context and memory validation (2026-11-16 UTC)

The gateway, launcher, and provider now use **24576** context tokens.
Pi compacts when `contextTokens > contextWindow - reserveTokens`; its default
`reserveTokens` is 16384. The former 16384-token window left a zero threshold
and compacted after every turn. The chosen window leaves an **8192-token**
threshold, allowing the short tool task to finish without rewriting history.
No pi compaction settings were changed.

Measured one gateway and one patched llama-server at a time, with the normal
single-slot, f16 KV, loops=1 self-draft configuration. After one real policy
`k=4` request (406 prompt tokens, 64 generated tokens), `ps -o rss=` reported
8959104 KiB (**9.174122496 GB**) at 32768 context. That exceeds the 9 GB
limit, so the specified 24576 fallback was measured: 7887696 KiB
(**8.077000704 GB**). GB here is decimal (`RSS KiB * 1024 / 1e9`).
Both measurement servers were stopped before the next server started.
The measurement harness added only `--verbosity 4` to expose buffer lines:

```text
0.00.504.202 I llama_kv_cache:       MTL0 KV buffer size =  2304.00 MiB
0.00.527.024 I llama_kv_cache:       MTL0 KV buffer size =   432.00 MiB
0.00.554.398 I sched_reserve:       MTL0 compute buffer size =   187.73 MiB
0.00.554.400 I sched_reserve:        CPU compute buffer size =    45.53 MiB
0.00.749.036 I llama_kv_cache:       MTL0 KV buffer size =   864.00 MiB
0.00.758.156 I llama_kv_cache:       MTL0 KV buffer size =   162.00 MiB
0.00.771.831 I sched_reserve:       MTL0 compute buffer size =   182.84 MiB
0.00.771.833 I sched_reserve:        CPU compute buffer size =    45.53 MiB
```

The first two KV buffers and compute pair belong to the target; the next
two KV buffers and compute pair belong to the same-weights draft. No rebuild
was needed for the subsequent comment-only llama.cpp clarification.

Provider backup: `/Users/srujanpendurthi/.pi/agent/models.json.bak-dory-20261116T001803678869Z`.
Only `providers.dory.models[0].contextWindow` changed; all other file text
was preserved, and `python3 -m json.tool` passed.

### pi end-to-end after the context fix

Used a fresh temporary directory outside the repository, prepending the
repository's `.venv/bin` to `PATH`. The launcher ran from that directory;
its one child used `-np 1 -c 24576 --jinja --spec-draft-loops 1`.
The command was `pi --provider dory --model dory --print --mode json --no-session`
with the same exact prompt as the earlier check:

```text
Create a file add.py with a function add(a, b) that returns a + b, then run `python -c 'import add; print(add.add(2, 3))'`
```

Trimmed JSON transcript:

```json
{"type": "tool_execution_start", "toolName": "write", "args": {"path": "add.py", "content": "def add(a, b):\n    return a + b\n"}}
{"type": "tool_execution_end", "toolName": "write", "result": {"content": [{"type": "text", "text": "Successfully wrote 32 bytes to add.py"}]}, "isError": false}
{"type": "tool_execution_start", "toolName": "bash", "args": {"command": "python -c 'import add; print(add.add(2, 3))'"}}
{"type": "tool_execution_end", "toolName": "bash", "result": {"content": [{"type": "text", "text": "5\n"}]}, "isError": false}
```

Pi exited **0**, with empty stderr, and wrote the exact 32-byte `add.py`
shown in the earlier transcript. The JSON transcript contains **zero
`compaction_start` events**. All three requests (write, bash, final response)
kept `k=2`, with no extra `k=4` compaction request:

```text
time=2026-11-16T00:18:27.998962+00:00 k=2 prompt_tokens=1664 generated_tokens=66 tok/s=16.907261330596263 draft_acceptance=1.0
time=2026-11-16T00:18:30.898620+00:00 k=2 prompt_tokens=1756 generated_tokens=39 tok/s=15.677550737504992 draft_acceptance=0.9375
time=2026-11-16T00:18:33.936280+00:00 k=2 prompt_tokens=1813 generated_tokens=46 tok/s=17.242832729195563 draft_acceptance=1.0
```

SIGTERM stopped the owned gateway and its child. Final process checks found
no gateway, no llama-server, and no listener on 8083. The gateway's stderr
also included the classifier's existing tokenizer-regex warning; classifier
configuration was left unchanged in this fix wave.

## Earlier transcript at 16384 context (2026-11-16 UTC)

The launcher used one patched `llama-server`, `-np 1`, `-c 16384`,
`--jinja`, and `--spec-draft-loops 1`. The provider backup was
`~/.pi/agent/models.json.bak-dory-20261116T000108443870Z`.

### Tool-call smoke test, without pi

POST `/v1/chat/completions` with `model: "dory"`, `stream: false`, a
`write_file(path, content)` tool schema, and the exact user message:

```text
Create hello.txt containing hi
```

Raw response excerpt (the `choices` value):

```json
[{"finish_reason":"tool_calls","index":0,"message":{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"write_file","arguments":"{\"path\":\"hello.txt\",\"content\":\"hi\"}"},"id":"kh1bSZppAeO2tU5zuHWmVUtqEBxy4xrx"}]}}]
```

The function name and parsed JSON arguments matched the requirement. The
tool call was structured, with empty `content`.

```text
time=2026-11-16T00:00:40.567359+00:00 k=1 prompt_tokens=294 generated_tokens=36 tok/s=14.834877220198322 draft_acceptance=-
```

### pi end-to-end: creates and runs add.py

From a fresh temporary directory outside the repository, the exact prompt
was passed as one argument to
`pi --provider dory --model dory --print --mode json --no-session`:

```text
Create a file add.py with a function add(a, b) that returns a + b, then run `python -c 'import add; print(add.add(2, 3))'`
```

The first run wrote the correct file, attempted the requested command, got
`/bin/bash: python: command not found` (exit 127), and automatically ran
`python3 -c 'import add; print(add.add(2, 3))'`, which printed `5`. Pi exited
0. Its four task requests all used `k=2`:

```text
time=2026-11-16T00:01:27.726113+00:00 k=2 prompt_tokens=1700 generated_tokens=66 tok/s=16.991838166596345 draft_acceptance=1.0
time=2026-11-16T00:01:30.657834+00:00 k=2 prompt_tokens=1792 generated_tokens=39 tok/s=15.932486507699585 draft_acceptance=0.9375
time=2026-11-16T00:01:34.284033+00:00 k=2 prompt_tokens=1865 generated_tokens=46 tok/s=14.811333237224238 draft_acceptance=0.875
time=2026-11-16T00:01:39.689830+00:00 k=2 prompt_tokens=1929 generated_tokens=69 tok/s=13.91289830019165 draft_acceptance=0.828125
```

The check was repeated in another fresh directory with the repository's
`.venv/bin` prepended to `PATH`, without changing the prompt, model, or pi
settings. Trimmed pi JSON transcript:

```json
{"type":"tool_execution_start","toolName":"write","args":{"path":"add.py","content":"def add(a, b):\n    return a + b\n"}}
{"type":"tool_execution_end","toolName":"write","result":{"content":[{"type":"text","text":"Successfully wrote 32 bytes to add.py"}]},"isError":false}
{"type":"tool_execution_start","toolName":"bash","args":{"command":"python -c 'import add; print(add.add(2, 3))'"}}
{"type":"tool_execution_end","toolName":"bash","result":{"content":[{"type":"text","text":"5\n"}]},"isError":false}
```

Pi exited 0 with no stderr. The resulting `add.py` was:

```python
def add(a, b):
    return a + b
```

All three task requests used `k=2`:

```text
time=2026-11-16T00:02:36.096532+00:00 k=2 prompt_tokens=1703 generated_tokens=66 tok/s=16.776804235601052 draft_acceptance=1.0
time=2026-11-16T00:02:39.088288+00:00 k=2 prompt_tokens=1795 generated_tokens=39 tok/s=15.067167848249829 draft_acceptance=0.9375
time=2026-11-16T00:02:42.262846+00:00 k=2 prompt_tokens=1852 generated_tokens=46 tok/s=16.309181489296467 draft_acceptance=1.0
```

**Earlier compaction issue (fixed by the 24576 context above):** after each completed task, pi emitted
`{"type":"compaction_start","reason":"threshold"}` and made an additional
gateway request. Its installed defaults enable compaction with
`reserveTokens: 16384`, equal to this provider's entire context window;
the threshold is `contextWindow - reserveTokens`. The compaction prompt
is a new user message, so the policy can choose a different `k`. These
extra requests used `k=4`:

```text
time=2026-11-16T00:01:55.290626+00:00 k=4 prompt_tokens=284 generated_tokens=132 tok/s=12.645698232710577 draft_acceptance=0.8416666666666667
time=2026-11-16T00:02:53.249011+00:00 k=4 prompt_tokens=284 generated_tokens=132 tok/s=13.04392027356785 draft_acceptance=0.8416666666666667
```

The task's write, execution, and final response retained one `k`; the full
pi invocation includes this extra compaction work. No compaction settings
were changed. Both runs used the exact requested provider configuration.
After the check, SIGTERM stopped the gateway and its single child server;
process checks found neither remaining.
