# Dory Loop-Count Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve NVIDIA Dory1.0-4B on this Apple M4 (24 GB) with a controller that picks the loop count k ∈ {1,2,4} per request, plus measured batching, KV-cache and speculative-decoding choices.

**Architecture:** llama.cpp (Metal, PR #30306 fork, pinned) runs one `llama-server` per k. Every server shares the same mmap'd Q5_K_M GGUF and has batched slots, so mixed-k traffic batches by k. A Python package `dory/` does four things:
- benchmarks single-stream and batched speed;
- runs fixed-k sweeps: a lazy sweep for training labels, and the full grid for holdout and final-test evaluation;
- trains two controllers, a text baseline and a k=1 hidden-state probe;
- evaluates them offline with *modeled* latency (token counts ÷ single-stream speed), then routes live requests.

Each phase ends in a measured go/no-go gate recorded in `results/decisions.md`.

**Tech Stack:** Python 3.13 (`/usr/local/bin/python3`, venv `.venv`), llama.cpp Metal build, numpy, scikit-learn, joblib, datasets, huggingface_hub, torch (MPS) + transformers (NVIDIA classifier only), pytest. HTTP is stdlib `urllib`. Process RSS comes from stdlib `subprocess` + `ps`.

**Spec:** `reports/Dory loop count controller.md` (research report), plus the conversation addendum copied into Global Constraints below (quantization, parallelization, speculative decoding, the speed-ups, and the validity fixes). Background notes: `research_notes/Dory loop count controller/*.md`.

## Global Constraints

- Repo root: `/Users/srujanpendurthi/Projects/Current/doryAgent`, its own git repo. `vendor/`, `models/`, `.venv/` are gitignored. `results/` is committed.
- Model: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`, Q5_K_M only, saved as `models/dory-q5.gguf`. From `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, download only `config.json`, `tokenizer_config.json` and `chat_template*` (no weights). Record revision hashes in `results/decisions.md`.
- Deferred, out of scope for this plan: Q4_K_M and mixed (recurrent Q6_K, rest Q4_K_M) quantization. If adopted later, re-check the controller on the holdout set, because labels are tied to Q5.
- llama.cpp: PR #30306 head of ggml-org/llama.cpp, built with Metal at a pinned commit SHA stored as `LLAMA_REF` in `scripts/setup_llama.sh`.
- Loop count set only via `--override-kv dory.num_loops=int:{k}`. `KS = [1, 2, 4]` everywhere. Never run k above 4. k=3 is intentionally excluded.
- KV mode: default `per_loop`. KV cache type is f16 unless Task 5 keeps `q8_0` (`-ctk q8_0 -ctv q8_0`, needs `-fa on`).
- All generation greedy: `temperature=0`, `seed=0`. Thinking mode OFF if the chat template has a toggle.
- Max new tokens: gsm8k 512, mmlu_pro 512, mbpp 768. Do not lower them: truncation corrupts labels.
- Coding prompts (classifier task type `Code Generation`, or source `mbpp`) are never routed to k=1. `MIN_CODE_K = 2`.
- **Latency is modeled, never taken from sweep wall-clock:** `seconds = n_prompt / speed[k].prompt_tps + n_gen / speed[k].gen_tps`, using single-stream (slots=1) numbers from `results/speed.json`. The probe tax uses the same formula: `n_prompt / speed[1].prompt_tps`. Text-classifier time is measured directly.
- Data splits are disjoint by id and loaded in this order, each excluding the earlier ones' ids:
  - holdout: gsm8k:test:100, mmlu_pro:test:100, mbpp:test:100
  - final: gsm8k:test:35, mmlu_pro:test:35, mbpp:test:30
  - train: gsm8k:train:600, mmlu_pro:test:600, mbpp:train:300
- Train is swept lazily. Holdout and final get the full KS grid. The final split is evaluated exactly once, in Task 10, with the already-chosen policy. Never tune on it.
- Do NOT change `iogpu.wired_limit_mb`. Ignore NVFP4/MXFP8 builds.
- Binaries: env `DORY_LLAMA_BIN`, default `vendor/llama.cpp/build/bin`.
- Integration tests: `@pytest.mark.integration`, skipped unless env `DORY_GGUF` is set. Network tests: `@pytest.mark.network`.
- Spec estimates to verify (not measured): ~11 s / 17 s / 27 s per request at k=1/2/4 (1k-token prompt, 300-token answer); probe ~2.7 s per 1k prompt tokens; Q5_K_M ~3.7 GB.

## Execution Order

- Task 1 first. Tasks 2, 4 and 7 have no model dependency, so dispatch them in parallel git worktrees while Task 1's download and build run, then merge.
- Task 8 Steps 1–4 (pure logic) can also start once Task 7 merges.
- Run long sweeps with `caffeinate -dims` in the background. No other GPU jobs during benchmarks or sweeps.

## Review Focus

1. **Server fails to start** (bad GGUF path, port in use, bad override key): `Server` raises `RuntimeError` with the stderr tail within `startup_timeout` and leaves no orphan process. → Task 2.
2. **Sweep killed mid-run / torn last JSONL line:** rerun skips completed `(id, k, variant)` rows and ignores the malformed line. → Task 5.
3. **Output empty, truncated, or missing the answer pattern:** `grade` returns `False`, never raises. → Task 4.
4. **Generated MBPP code hangs or touches files:** runs under `python -I` in a temp dir with a 10 s timeout, and timeout means `False`. Not a sandbox; mark with `ponytail:`. → Task 4.
5. **Prompt exceeds context (HTTP 4xx/5xx):** `complete` raises `ServerError`. The sweep records an `error` row with `correct=False` and continues. A lazy sweep treats an error row as "not correct" and still escalates k. → Tasks 2, 5.

---

### Task 1: Toolchain, model download, spikes

**Files:**
- Create: `.gitignore`, `requirements.txt`, `scripts/setup_llama.sh`, `scripts/get_models.sh`, `tests/conftest.py`, `tests/test_smoke_llama.py`, `results/decisions.md`

**Interfaces:**
- Produces: `vendor/llama.cpp/build/bin/{llama-server,llama-cli,llama-lookup}`, `models/dory-q5.gguf`, `models/bf16-meta/`, pytest markers `integration` and `network`, and a `gguf` fixture (`Path` from `DORY_GGUF`, skip if unset).

- [ ] **Step 1:** `git init`. Write `.gitignore` (`vendor/ models/ .venv/ __pycache__/ *.pyc`) and `requirements.txt` (`numpy scikit-learn joblib datasets huggingface_hub torch transformers pytest`). Create `.venv` and install. Commit with the existing `reports/` and `research_notes/`.
- [ ] **Step 2:** Write `scripts/setup_llama.sh`. It clones ggml-org/llama.cpp to `vendor/llama.cpp` and runs `git fetch origin pull/30306/head:dory`. It then checks out the hard-coded `LLAMA_REF` and runs `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 10`. Run `vendor/llama.cpp/build/bin/llama-server --version`. Expected: version printed.
- [ ] **Step 3:** Write `scripts/get_models.sh`. It runs `hf download` for the Q5_K_M GGUF to `models/dory-q5.gguf`, and for the BF16 repo with `--include config.json tokenizer_config.json "chat_template*"` to `models/bf16-meta/`. Record in `results/decisions.md`: layer counts (8/20/8 vs 16/40/16), revision hashes, and the thinking toggle with how to disable it.
- [ ] **Step 4: Write the failing smoke tests** (integration) in `tests/test_smoke_llama.py`:
  - `test_loops_change_output`: `llama-cli -m $DORY_GGUF -no-cnv -n 48 --temp 0 --seed 0 -p "<fixed GSM8K question>" --override-kv dory.num_loops=int:K` produces different output for K=1 and K=4.
  - `test_embedding_endpoint`: a `llama-server --embeddings --pooling last` started at k=1 answers `POST /embedding {"content":"hello"}` with a vector of length > 100.
- [ ] **Step 5:** Run `DORY_GGUF=models/dory-q5.gguf pytest tests/test_smoke_llama.py -v`. Expected: PASS.
  - If loop outputs are identical: gate failure, stop and report to the user.
  - If the embedding test fails: record "probe infeasible, skip Task 9" and continue.
- [ ] **Step 6: Draft-override spike:** run `llama-server --help | grep -iE "override.*draft|draft.*override"`. Record in `results/decisions.md` whether a draft-side kv override exists. If none exists, Task 11 is skipped.
- [ ] **Step 7:** Commit: `chore: llama.cpp Dory fork build, model fetch, smoke tests and spikes`.

### Task 2: Server process wrapper and HTTP client

**Files:**
- Create: `dory/__init__.py`, `dory/server.py`, `dory/client.py`, `tests/test_server.py`, `tests/test_client.py`

**Interfaces:**
- Produces:
  - `server_args(gguf: Path, k: int, port: int, slots: int = 4, ctx_per_slot: int = 4096, kv_type: str = "f16", embeddings: bool = False, draft: Path | None = None, extra: list[str] = []) -> list[str]`.
    - Always includes `-ngl 99 -fa on --temp 0 --seed 0 -np {slots} -c {ctx_per_slot*slots} --port {port} --override-kv dory.num_loops=int:{k}`.
    - `kv_type="q8_0"` adds `-ctk q8_0 -ctv q8_0`.
    - `embeddings=True` adds `--embeddings --pooling last`.
    - `draft` is handled in Task 11.
  - `class Server` (context manager):
    - `__init__(self, args: list[str], port: int, startup_timeout: float = 120)`
    - attributes `.url: str` and `.pid: int`
    - `.rss_mb() -> float` via `ps -o rss= -p pid`
    - `.stop()`
    - Ready means `GET /health` returns 200.
  - `free_port() -> int`.
  - `@dataclass Completion(text: str, n_prompt: int, n_gen: int, prompt_tps: float, gen_tps: float, seconds: float, draft_n: int = 0, draft_accepted: int = 0)`.
  - `class ServerError(RuntimeError)`.
  - `complete(url: str, prompt: str, max_tokens: int, timeout: float = 900) -> Completion`:
    - POSTs `/v1/chat/completions` with one user message, plus the thinking-off setting from Task 1.
    - Fields come from `usage` and from `timings` (`prompt_per_second`, `predicted_per_second`, `draft_n`, `draft_n_accepted`).
  - `embed_last(url: str, text: str) -> list[float]`: POSTs `/embedding`.

- [ ] **Step 1: Write failing tests:**
  - `test_server_args_sets_loops`: `"dory.num_loops=int:4" in server_args(Path("m.gguf"), 4, 8080)`.
  - `test_server_args_q8_kv`: `kv_type="q8_0"` adds `-ctk q8_0 -ctv q8_0`.
  - `test_complete_parses_timings`: a stdlib `http.server` fake returns `{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2},"timings":{"prompt_per_second":100.0,"predicted_per_second":20.0,"draft_n":4,"draft_n_accepted":3}}`. Assert `text=="hi"`, `n_prompt==5`, `n_gen==2`, `gen_tps==20.0`, `draft_accepted==3`.
  - `test_complete_http_error_raises`: a 400 response raises `ServerError`.
  - `test_server_bad_model_raises`: `Server(server_args(Path("/nonexistent.gguf"),1,p),p,startup_timeout=30)` raises `RuntimeError`, and the pid is dead afterwards.
- [ ] **Step 2:** Run `pytest tests/test_server.py tests/test_client.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3:** Implement. Server uses `subprocess.Popen` with stderr to a temp file and polls `/health`. On timeout or early exit it kills the process and raises with the stderr tail.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then add and run integration test `test_real_server_completes`: k=1 server, `complete(url,"2+2=",8).n_gen > 0`.
- [ ] **Step 5:** Commit: `feat: llama-server wrapper and chat client`.

### Task 3: Speed and memory benchmark (Phase 0)

**Files:**
- Create: `dory/bench.py`, `tests/test_bench.py`, `results/bench.csv`, `results/speed.json`

**Interfaces:**
- Consumes: `server_args`, `Server`, `complete`, `free_port` (Task 2).
- Produces:
  - `summarize(rows: list[Completion], wall_seconds: float) -> dict` returning `{"gen_tps_per_req": mean gen_tps, "agg_tps": sum n_gen / wall_seconds, "prompt_tps": mean prompt_tps}`.
  - `results/speed.json`: `{"1": {"prompt_tps": float, "gen_tps": float}, "2": {...}, "4": {...}}` from slots=1, f16 KV runs.
  - `load_speed(path: Path = Path("results/speed.json")) -> dict[int, dict[str, float]]`.
  - `results/decisions.md` entry `SWEEP_SLOTS`.

- [ ] **Step 1: Write failing tests:**
  - `test_summarize`: Completions with n_gen 100/300, gen_tps 10/30, and wall 20 s give `agg_tps==20.0` and `gen_tps_per_req==20.0`.
  - `test_load_speed_int_keys`: written JSON reads back with int keys.
- [ ] **Step 2:** Run `pytest tests/test_bench.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `summarize`, `load_speed`, and the CLI `python -m dory.bench --out results/bench.csv`. It runs exactly these 11 configs on Q5: k ∈ {1,2,4} × slots ∈ {1,4,8} with f16 KV, plus k=4 with q8_0 KV at slots 1 and 4. Per config:
  - start the server and warm up once;
  - send 8 fixed ~1k-token prompts (GSM8K/MBPP text, `max_tokens=300`) at concurrency = slots;
  - write a CSV row (`summarize` output + `rss_mb`).
  Then write `speed.json` from the slots=1 f16 rows.
- [ ] **Step 4:** Run `pytest tests/test_bench.py -v` (PASS), then the CLI.
- [ ] **Step 5: Memory gate + decisions.** Start servers for k=1, 2 and 4 plus a k=1 embeddings server at the same time. Combined RSS must be < 16 GB; otherwise stop and report to the user (fallback is a per-context `num_loops` C++ patch, out of scope). Record in `results/decisions.md`:
  - single-stream tok/s per k vs the spec estimates;
  - `SWEEP_SLOTS = 8` if k=2 `agg_tps` at slots 8 ≥ 1.15× slots 4, else `4`;
  - q8_0 KV speed delta.
- [ ] **Step 6: Prompt-lookup speculative decoding.** Run `llama-server --help | grep -iE "lookup|ngram|spec"`.
  - If the server supports n-gram drafting, benchmark it.
  - Otherwise use `llama-lookup -m models/dory-q5.gguf --override-kv dory.num_loops=int:2 -n 300 --temp 0 -p <prompt>` on 10 MBPP prompts that contain code context, vs plain `llama-cli` with the same arguments.
  - Record `LOOKUP_FOR_CODE = true` if the speedup is ≥ 1.15x.
- [ ] **Step 7:** Commit: `feat: Phase 0 speed/memory benchmark`.

### Task 4: Eval sets and graders

**Files:**
- Create: `dory/evalsets.py`, `tests/test_evalsets.py`

**Interfaces:**
- Produces:
  - `@dataclass Item(id: str, source: str, prompt: str, gold: str, tests: list[str] | None = None)`, where `source ∈ {"gsm8k","mmlu_pro","mbpp"}`.
  - `MAX_TOKENS = {"gsm8k": 512, "mmlu_pro": 512, "mbpp": 768}`.
  - `load(source: str, split: str, n: int, seed: int = 0, exclude: set[str] = set()) -> list[Item]`. Deterministic: shuffle indices with `random.Random(seed)`, drop excluded ids, take the first n.
  - `load_split(name: str) -> list[Item]` for `name ∈ {"holdout","final","train"}`, implementing the Global Constraints split order and exclusions exactly.
  - `grade(item: Item, output: str) -> bool`.
- Datasets: `openai/gsm8k` (main), `TIGER-Lab/MMLU-Pro`, `google-research-datasets/mbpp` (full). ids are `f"{source}:{split}:{index}"`.
- Prompt suffixes (exact copy):
  - gsm8k: `"\n\nSolve step by step. End with 'The answer is N'."`
  - mmlu_pro: options `A. ...` through `J. ...`, then `"\n\nEnd with 'The answer is (X)'."`
  - mbpp: task text + first test assert, then `"\n\nReturn only the function in a ```python code block."`
- Grading:
  - gsm8k: last number in the output with commas stripped, compared as a float to the gold number after `####`.
  - mmlu_pro: last match of `answer is \(?([A-J])\)?`, else the last standalone `[A-J]` on the final line.
  - mbpp: last ```python block plus all tests in a temp dir, run with `python -I` and timeout 10 s; pass means exit 0.

- [ ] **Step 1: Write failing tests:**
  - `test_gsm8k_grade`: `"... The answer is 1,234."` vs gold `"1234"` is True; `"The answer is 12"` is False.
  - `test_mmlu_grade`: `"The answer is (C)"` is True vs C and False vs D.
  - `test_grade_garbage_is_false`: `""`, `"I don't know"`, and an unclosed code fence give False for every source.
  - `test_mbpp_pass_and_fail`: `def add(a,b): return a+b` with `assert add(1,2)==3` is True.
  - `test_mbpp_timeout_false`: an infinite-loop body returns False in < 15 s.
  - `test_splits_disjoint` (network): the id sets of `load_split` holdout/final/train are pairwise disjoint, with sizes 300/100/1500.
- [ ] **Step 2:** Run `pytest tests/test_evalsets.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Add the comment `# ponytail: subprocess+timeout is not a sandbox; use a container if prompts become untrusted`.
- [ ] **Step 4:** Run `pytest tests/test_evalsets.py -v -m "not network"`, then the network test. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: eval sets, splits and graders`.

### Task 5: Sweep runner (lazy + full) and Phase 0 quality gate

**Files:**
- Create: `dory/sweep.py`, `tests/test_sweep.py`, `results/sweep_holdout.jsonl` (output)

**Interfaces:**
- Consumes: `Item`, `load_split`, `grade`, `MAX_TOKENS` (Task 4); `Server`, `server_args`, `complete`, `ServerError` (Task 2); `SWEEP_SLOTS` (Task 3).
- Produces:
  - Row schema: `{"id","source","k","variant","output","correct","n_prompt","n_gen","error"}`.
  - `read_rows(path: Path) -> list[dict]`: skips malformed lines.
  - `pending(items: list[Item], k: int, variant: str, done: list[dict], lazy: bool) -> list[Item]`.
    - Returns items with no `(id, k, variant)` row.
    - When `lazy`, it also drops items that already have a `correct=True` row at a lower k for this variant.
  - `run_sweep(items: list[Item], ks: list[int], variant: str, gguf: Path, out: Path, slots: int, kv_type: str = "f16", lazy: bool = False) -> None`:
    - takes ks in ascending order, one server per k at a time;
    - runs `pending` items through a `ThreadPoolExecutor(slots)` and appends rows under a lock;
    - on `ServerError`, writes the row with `error=str(e)`, `correct=False`.
  - CLI: `python -m dory.sweep --split holdout|final|train --ks 1,2,4 --variant q5 --out F [--kv q8_0] [--lazy] [--sources gsm8k,mmlu_pro]`. `--sources` filters the split.

- [ ] **Step 1: Write failing tests:**
  - `test_pending_full`: 2 items with a done row `(id0,1,"q5")` give `[item1]` at k=1. Rows of another variant are ignored.
  - `test_pending_lazy`: item0 correct at k=1 and item1 wrong at k=1, then at k=2 with `lazy=True` gives `[item1]`, and with `lazy=False` gives both.
  - `test_read_rows_skips_torn_line`: 2 valid lines + `{"id": "x", "k"` give 2 rows.
  - `test_error_row_recorded`: monkeypatch `dory.sweep.complete` to raise `ServerError("ctx")` and `dory.sweep.Server` to a no-op context manager. 1 item at k [1] gives one row with `error=="ctx"` and `correct is False`.
- [ ] **Step 2:** Run `pytest tests/test_sweep.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Phase 0 quality run (becomes part of the holdout).** Run `python -m dory.sweep --split holdout --sources gsm8k,mmlu_pro --ks 1,2,4 --variant q5 --out results/sweep_holdout.jsonl`. Then run the same split at `--ks 4 --kv q8_0 --variant q5-kvq8 --out results/phase0_kvq8.jsonl`.
- [ ] **Step 6: Gate:**
  - (a) q5 accuracy at k=4 ≥ k=1 on those 200 items. Otherwise stop and report.
  - (b) Keep q8_0 KV (`CHOSEN_KV`) only if within 1.0 pt of f16 at k=4.
  - Record both in `results/decisions.md`.
- [ ] **Step 7:** Commit: `feat: lazy/full sweep runner; Phase 0 quality gate`.

### Task 6: Labels (Phase 1)

**Files:**
- Create: `dory/labels.py`, `tests/test_labels.py`, `results/sweep_train.jsonl`, `results/sweep_final.jsonl`, `results/labels_{train,holdout,final}.jsonl`

**Interfaces:**
- Consumes: `read_rows`, `run_sweep` (Task 5); `load_split` (Task 4).
- Produces:
  - `KS = [1, 2, 4]`.
  - `min_k(correct_by_k: dict[int, bool]) -> int | None`: smallest k with True. Returns `None` only if all of KS are present and False. A lazy item with missing higher ks after a True is still labeled.
  - `build_labels(sweep: Path, items: list[Item]) -> list[dict]`. Record: `{"id","source","prompt","label_k","correct_by_k":{k:bool},"tokens_by_k":{k:[n_prompt,n_gen]}}`, with error rows counted as incorrect.
  - `class_balance(labels: list[dict]) -> dict[int, float]` over non-None labels.
  - `non_monotone_rate(labels: list[dict]) -> float`: fraction of full-grid items correct at some k but wrong at a larger k.
  - CLI: `python -m dory.labels --split train|holdout|final --sweep F --out F`.

- [ ] **Step 1: Write failing tests:**
  - `test_min_k`:
    - `{1:False,2:True,4:True}` → 2
    - `{1:True}` (lazy) → 1
    - `{1:False,2:False,4:False}` → None
    - `{1:False,2:False}` (incomplete) → None, with no exception
    - `{1:False,2:True,4:False}` → 2
  - `test_class_balance`: [1,1,2,None] gives `{1:2/3, 2:1/3}`.
  - `test_non_monotone_rate`: two items, one `{1:True,2:False,4:True}` and one `{1:False,2:True,4:True}`, give 0.5.
- [ ] **Step 2:** Run `pytest tests/test_labels.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Run sweeps** with `CHOSEN_KV`, `--slots SWEEP_SLOTS`, in the background under `caffeinate -dims`:
  - holdout remainder: `--split holdout --ks 1,2,4` (resume skips the Phase 0 rows), to `results/sweep_holdout.jsonl`
  - `--split final --ks 1,2,4`, to `results/sweep_final.jsonl`
  - `--split train --ks 1,2,4 --lazy`, to `results/sweep_train.jsonl`

  Then build all three label files.
- [ ] **Step 6: Gate:** every class in `class_balance(train)` ≤ 0.80; otherwise stop and report. Record in `results/decisions.md`: class balance, None count, and `non_monotone_rate(holdout)`. If that rate is > 0.10, flag that "correct at k implies correct at higher k" is weak, and that the oracle in Task 7 is optimistic.
- [ ] **Step 7:** Commit: `feat: Phase 1 loop-count labels`.

### Task 7: Offline evaluation harness

**Files:**
- Create: `dory/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: full-grid label records (Task 6); `load_speed` (Task 3).
- Produces:
  - `modeled_seconds(n_prompt: int, n_gen: int, k: int, speed: dict) -> float` = `n_prompt/speed[k]["prompt_tps"] + n_gen/speed[k]["gen_tps"]`.
  - `policy_metrics(labels, choice: dict[str,int], speed: dict, extra_seconds: dict[str,float] | None = None) -> dict` returning `{"acc","mean_k","mean_seconds"}`. Accuracy uses `correct_by_k[choice[id]]`; seconds = `modeled_seconds(*tokens_by_k[k], k, speed)` + extra.
  - `fixed(labels, k)`, `oracle(labels)` (label_k, else 1), and `random_matched(labels, mean_k: float, seed=0)`, each `-> dict[str,int]`. `random_matched` mixes the two KS neighbours of mean_k so the expected mean equals mean_k.
  - `paired_bootstrap(a: list[bool], b: list[bool], n=10000, seed=0) -> tuple[float,float,float]` returning `(mean(a−b), lo95, hi95)`.
  - `report(labels, policies: dict[str, tuple[dict[str,int], dict[str,float] | None]], speed) -> str`: a markdown table plus the bootstrap of each policy vs `fixed2`.

- [ ] **Step 1: Write failing tests** on a 3-item fixture with a hand-made speed dict:
  - `test_modeled_seconds`: prompt 1000 at 500 tps plus gen 300 at 30 tps gives 12.0.
  - `test_oracle_dominates`: oracle acc ≥ max fixed acc, and oracle mean_k ≤ fixed4 mean_k.
  - `test_policy_metrics_values`: exact numbers for `fixed(labels,2)` with `extra_seconds={id:1.0}`.
  - `test_bootstrap_identical`: `paired_bootstrap(x, x) == (0.0,0.0,0.0)`.
  - `test_random_matched_mean`: 10,000 fake ids at mean_k=2.5 have a mean within 0.05 of 2.5, and every choice is in KS.
- [ ] **Step 2:** Run `pytest tests/test_evaluate.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement with numpy only.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: offline evaluation with modeled latency`.

### Task 8: Text baseline controller (Phase 2)

**Files:**
- Create: `dory/nvidia_classifier.py`, `dory/features.py`, `dory/policy.py`, `tests/test_policy.py`

**Interfaces:**
- Consumes: labels (Task 6), Task 7 functions, `load_speed`.
- Produces:
  - `dory/nvidia_classifier.py`: the custom multi-head model and loading code copied from the `nvidia/prompt-task-and-complexity-classifier` card. Runs on `mps` (fallback `cpu`), truncates to 512 tokens.
  - `text_features(prompts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(X, is_code, seconds)`.
    - X has shape (n, 9): creativity, reasoning, contextual_knowledge, domain_knowledge, constraint_ct, number_of_few_shots, prompt_complexity_score, log1p(char length), is_code.
    - `is_code` = `task_type_1 == "Code Generation"`.
    - `seconds` is the measured per-prompt classifier time.
  - `fit(X, y, cv: bool = False) -> Pipeline`: `StandardScaler` + `LogisticRegression(max_iter=2000, class_weight="balanced")`. With `cv=True`, uses `LogisticRegressionCV(Cs=10, cv=5)` instead.
  - `choose_k(proba, is_code, tau: float, ks=KS, min_code_k=2) -> np.ndarray`: the smallest k with cumulative P(label ≤ k) ≥ tau, then `max(k, min_code_k)` where `is_code`.
  - `tau_for_mean_k(proba, is_code, target: float) -> float` (bisection; uses predictions only, no labels).
  - `save_policy(path, kind: str, model, tau: float, feature_set: str)` and `load_policy(path) -> dict`, where `feature_set ∈ {"text","probe","probe+text"}`.

- [ ] **Step 1: Write failing tests:**
  - `test_choose_k_cumulative`: proba `[[0.5,0.3,0.2]]` gives 1 at tau 0.5 and 2 at tau 0.8.
  - `test_code_never_k1`: the same proba with `is_code=[True]` at tau 0.5 gives 2.
  - `test_tau_for_mean_k`: Dirichlet proba (seed 0, n=500) gives a mean k within 0.1 of 2.0 at the returned tau.
  - `test_fit_predicts_classes`: separable synthetic data gives train accuracy > 0.9 for both `cv=False` and `cv=True`.
- [ ] **Step 2:** Run `pytest tests/test_policy.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Train + evaluate.** CLI `python -m dory.policy --kind text`:
  - cache features to `results/features_text_{split}.npy`;
  - fit on train rows where label_k is not None;
  - set `tau = tau_for_mean_k(holdout proba, target=2.0)`;
  - append `report` on the holdout to `results/decisions.md`, covering text policy (extra = classifier seconds), fixed1/2/4, random_matched(2.0), and oracle.
- [ ] **Step 6: Gate:** pass if the text policy beats fixed2 at matched mean k by ≥ +1.0 pt accuracy, or matches it (bootstrap CI contains 0) with ≥ 15% lower mean_seconds. Save `results/policy_text.joblib` either way, and record pass or fail.
- [ ] **Step 7:** Commit: `feat: text-feature loop-count controller`.

### Task 9: Hidden-state probe controller (Phase 3)

Skip this task if Task 1 recorded "probe infeasible".

**Files:**
- Create: `dory/probe.py`, `tests/test_probe.py`

**Interfaces:**
- Consumes: `Server`, `server_args(embeddings=True)`, `embed_last` (Task 2); `fit(cv=True)`, `choose_k`, `tau_for_mean_k`, `save_policy`, `text_features` (Task 8); `report`, `modeled_seconds` (Task 7).
- Produces:
  - `probe_features(prompts: list[str], url: str) -> np.ndarray`: L2-normalized last-token final-layer embedding from a k=1 embeddings server.
  - `probe_seconds(labels) -> dict[str,float]` = `n_prompt / speed[1]["prompt_tps"]` per id (modeled probe tax).
  - CLI `python -m dory.probe`: caches features to `results/features_probe_{split}.npy`, then fits `probe` and `probe+text` with `cv=True`. Evaluates on the holdout with extra = probe tax (+ classifier seconds for the combination).
- Add the comment `# ponytail: final-layer last-token embedding; mid-layer via eval-callback hook if probe underperforms`.

- [ ] **Step 1: Write failing tests:**
  - `test_probe_features_normalized`: monkeypatch `dory.probe.embed_last` to return `[3.0, 4.0]`; row norms equal 1.0.
  - `test_probe_seconds`: n_prompt 1000 with `prompt_tps` 500 gives 2.0.
- [ ] **Step 2:** Run `pytest tests/test_probe.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run the CLI and append the holdout report to `results/decisions.md`.
- [ ] **Step 6: Gate and policy selection:**
  - Promote probe (or probe+text) only if it beats text at matched modeled mean_seconds by ≥ +1.0 pt.
  - Otherwise keep text if it passed Task 8. If neither passed, use `None` (fixed k=2 with the code floor).
  - Copy the winner to `results/policy.joblib` and record the choice.
- [ ] **Step 7:** Commit: `feat: hidden-state probe controller and policy selection`.

### Task 10: Router + final test (Phase 4)

**Files:**
- Create: `dory/router.py`, `tests/test_router.py`

**Interfaces:**
- Consumes: `load_policy`, `choose_k`, `text_features` (Task 8); `probe_features` (Task 9); `Server`, `server_args`, `complete`, `free_port` (Task 2); `policy_metrics`, `report` (Task 7); `CHOSEN_KV`, `SWEEP_SLOTS` and `LOOKUP_FOR_CODE` (Tasks 3 and 5).
- Produces:
  - `class Router` (context manager):
    - `__init__(self, policy_path: Path | None, gguf: Path, kv_type: str, ks=KS, slots: int = 4, lookup_for_code: bool = False, spec_k4: bool = False)`. Starts one server per k, plus a k=1 embeddings server if the policy uses the probe.
    - `decide(prompt: str) -> tuple[int, float]` returns `(k, controller_seconds)`.
    - `generate(prompt: str, max_tokens: int = 512) -> tuple[Completion, int]`.
    - `policy_path=None` means fixed k=2 with the code floor.
  - CLI: `python -m dory.router [--policy results/policy.joblib] "prompt"` prints k, controller ms, tok/s and the text.

- [ ] **Step 1: Write failing tests:**
  - `test_decide_code_floor`: monkeypatch `dory.router.Server` to a no-op and `text_features` to return `is_code=[True]`, with a policy whose proba favours k=1. `decide` returns k ≥ 2.
  - `test_no_policy_defaults_k2`: non-code prompt gives k=2.
- [ ] **Step 2:** Run `pytest tests/test_router.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Requests go to `servers[k].url`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Gate (holdout):**
  - Mean controller overhead on 50 holdout prompts through the live router is < 50 ms for the text policy (the probe is reported against its tax).
  - Live accuracy on those 50 is within 5.0 pts of offline `policy_metrics` for the same ids.
- [ ] **Step 6: Final test, run exactly once.** Use `policy_metrics` on `labels_final.jsonl` for the chosen policy vs fixed1/2/4 and oracle, with the same tau, no refit. Record as the headline result in `results/decisions.md`.
- [ ] **Step 7:** Commit: `feat: per-request loop-count router; final test result`.

### Task 11: Self-speculative decoding, k=1 draft for k=4 (gated Phase 5)

Skip this task if Task 1 Step 6 found no draft-side kv override.

**Files:**
- Modify: `dory/server.py` (`server_args` draft handling), `dory/router.py` (`spec_k4`)
- Create: `tests/test_spec.py`

**Interfaces:**
- Consumes: `server_args`, `complete` (`draft_n`, `draft_accepted`), `Router`, `load_speed`.
- Produces: `server_args(..., draft: Path | None, draft_k: int = 1)`. When `draft` is set it adds `-md {draft} --draft-max 4` and the draft-side override `dory.num_loops=int:{draft_k}`, using the flag name recorded in Task 1.

- [ ] **Step 1: Write failing test** `test_server_args_draft`: args for k=4 with `draft=Path("m.gguf")` contain `"-md"` and a string ending in `dory.num_loops=int:1` after the draft override flag.
- [ ] **Step 2:** Run `pytest tests/test_spec.py -v`. Expected: FAIL. Implement. Run again. Expected: PASS.
- [ ] **Step 3: Measure** on 100 holdout prompts, slots=1, with a k=4 server plus a k=1 draft:
  - acceptance = `sum(draft_accepted)/sum(draft_n)`;
  - gen tok/s vs `speed[4]["gen_tps"]`;
  - outputs must be identical to plain k=4 greedy outputs on 20 prompts.
- [ ] **Step 4: Gate:** acceptance ≥ 0.80, speedup ≥ 1.2x, and identical outputs. If all pass, default `spec_k4=True` in the router CLI. Record the numbers.
- [ ] **Step 5:** Commit: `feat: self-speculative decoding for k=4`.
