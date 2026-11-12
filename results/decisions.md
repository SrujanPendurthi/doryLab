# Decisions and gate results

## Task 1

- llama.cpp: PR #30306 ("dory support for llama.cpp", OPEN) is the Dory PR. LLAMA_REF = 1c3cdd23787750814aa21a070dc4d6dc451af871 (PR head; build reports `0.8.0-dev commit 1c3cdd2`). Pinned the PR head, not master (master support for `dory` arch not checked). Shallow clone; built targets: llama-server, llama-cli, llama-completion, llama-lookup.
- Deviation: this llama.cpp has no `-no-cnv` in llama-cli (only `-st`); `-no-cnv` lives in `llama-completion`, so the smoke test uses `llama-completion` (added to the build targets).
- Model repos: GGUF `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF` rev df53755cb8741843af8e019e088819fbef17e29b, file `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (3.67 GB), stored at models/gguf/, symlinked as models/dory-q5.gguf. BF16 `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` rev f6c832e6963ddb74f62c75813a3e8c74b1af67b1 (config.json, tokenizer_config.json, chat_template.jinja only, in models/bf16-meta/).
- Layer counts (config.json): n_input_layers=16 / n_recurrent_layers=40 / n_output_layers=16 (72 layer_types entries incl. mlp; NOT 8/20/8), default n_recurrent_loops=2, recurrent_kv_cache_mode=per_loop, sliding_window=1024.
- Thinking toggle: chat template var `enable_thinking` (default True). True -> generation prompt `<|im_start|>assistant\n<think>\n`; False -> `<|im_start|>assistant\n<think></think>`. Disable via llama-server `--chat-template-kwargs '{"enable_thinking":false}'` (also `--reasoning off` exists) or per-request `chat_template_kwargs`.
- Gate: K=1 vs K=4 outputs differ (e.g. "2+2=4. 4-2=2..." vs "2+2=4. 4*4=16..."): PASS.
- Embedding: llama-server `--override-kv dory.num_loops=int:1 --embeddings --pooling last -ngl 99`, POST /embedding {"content":"hello"} returns vector len > 100: PASS. Probe feasible (Task 9 not skipped).
- Draft-override spike: `llama-server --help` has no draft-side `--override-kv` (only `--override-tensor-draft`, `--cache-type-k/v-draft`). Task 11 is SKIPPED.

## Task 3

- Bench (results/bench.csv, 11 configs; 8 prompts built from GSM8K/MBPP test text, actual mean n_prompt = 1199.75 tokens, max_tokens 300, thinking off). CHAT_TEMPLATE_KWARGS set to {"enable_thinking": false}.
- Single-stream (slots=1, f16) from results/speed.json: k=1 gen 23.3 tok/s, prompt 281 tok/s; k=2 gen 13.6, prompt 150; k=4 gen 8.3, prompt 90.
- Modeled latency for 1.2k prompt + 300 gen: k=1 17.2 s, k=2 30.1 s, k=4 49.3 s vs spec estimates 11 / 17 / 27 s (about 1.6-1.8x slower, as expected from 16/40/16 layers rather than 8/20/8).
- SWEEP_SLOTS = 4. k=2 agg_tps slots 8 = 18.28 vs slots 4 = 17.71 (1.03x < 1.15x). (For reference k=1: 36.0 vs 26.4 = 1.36x; k=4: 13.4 vs 11.0 = 1.21x; only the k=2 rule was specified.)
- q8_0 KV (k=4): slots=1 gen 7.98 vs 8.34 tok/s (-4.4%), agg 5.86 vs 6.09 (-3.7%); slots=4 gen 5.41 vs 5.03 (+7.5%), agg 10.96 vs 11.03 (-0.7%). Net about neutral; RSS 8.0 vs 9.6 GB (slots 1), 8.9 vs 10.3 GB (slots 4).
- Memory gate (scripts/memgate.py, controller ruling: system-wide delta, not summed RSS): 4 servers (k=1,2,4 + k=1 embeddings, slots=1) all healthy and each served one ~1.2k-token request. Used (active+wired+compressor) 11.60 -> 17.53 GB = +5.93 GB (<= 16 GB); swap used 18752.75 -> 18752.75 MB = +0 MB (<= 1 GB); memory_pressure free 82% -> 60%. PASS. Summed RSS 17.7 GB (4.2/4.5/5.0/4.0) is an over-count of the shared mmap'd GGUF. footprint phys_footprint 0.5-1.4 GB per process. Note: baseline swap was already 18.8 GB used before the run (pre-existing, other project).
- Lookup (scripts/lookup_bench.py): llama-server supports n-gram drafting (`--spec-type ngram-simple`), so llama-lookup was not needed. k=2, slots=1, greedy, 10 MBPP prompts with the reference solution as code context ("rewrite with type hints and docstring"), 1224 generated tokens total: plain 16.33 tok/s vs ngram-simple 12.08 tok/s = 0.74x (draft_n 90, accepted 14); outputs identical 10/10. LOOKUP_FOR_CODE = false (needs >= 1.15x).
- Note: pre-existing failing test tests/test_policy.py::test_text_features_shape_real (transformers config load of a model; fails on the base commit too, unrelated).

## Task 5

Phase 0 quality gate (holdout gsm8k+mmlu_pro, 200 items, q5, f16 KV, greedy, thinking off):

| dataset | k=1 | k=2 | k=4 |
|---|---|---|---|
| gsm8k (100) | 87.0 | 86.0 | 85.0 |
| mmlu_pro (100) | 48.0 | 43.0 | 50.0 |
| combined (200) | 135 | 129 | 135 |

- Gate (a) k=4 >= k=1: PASS (135 = 135, tie). Loops add no aggregate accuracy at fixed k; per-item disagreement is the router's only signal.
- Oracle (correct at any k): gsm8k 93, mmlu_pro 58 (151/200 = 75.5% vs best fixed 67.5%). Items wrong at k=1 but fixed by k=2 or k=4: gsm8k 6, mmlu_pro 10.
- mmlu_pro truncation at 512 tokens: 29/38/36 items at k=1/2/4; gsm8k 7/7/9.
- Gate (b): CHOSEN_KV = f16 (q8_0 quality check skipped, Ruling V: memory not tight, q8 speed-neutral).

## Task 6

Labels (min k correct; None = wrong at all of 1,2,4). Sweeps: 0 error rows in all files after 3 passes.

| split | n | k=1 | k=2 | k=4 | None |
|---|---|---|---|---|---|
| train (lazy) | 1500 | 0.878 | 0.090 | 0.031 | 449 |
| holdout | 300 | 0.896 | 0.075 | 0.030 | 99 |
| final | 100 | 0.881 | 0.090 | 0.030 | 33 |

(class balance is over non-None labels)

- Gate (every class <= 0.80): FAIL (k=1 = 0.878 on train). Ruling Z: continue instead of stopping — fit already uses class_weight="balanced"; remaining tasks cost minutes; the Task 8 gate vs fixed2 is the real usefulness test. A negative result there is the answer.
- non_monotone_rate(holdout) = 0.093 (<= 0.10, not flagged, but close: the oracle in Task 7 is somewhat optimistic). final: 0.06.
- Holdout fixed-k accuracy (all 300): k1 60.0, k2 58.7, k4 61.0, oracle 67.0. Per source k1/k2/k4: gsm8k 87/86/85, mmlu_pro 48/43/50, mbpp 45/47/48.
- Note: fixed1 beats fixed2 at ~1/2 the modeled cost, so fixed2 is a weak baseline; Task 8 also reports fixed1 for the honest comparison.

## Task 8

Text policy holdout report (tau via tau_for_mean_k target 2.0):

policy | acc | mean_k | mean_seconds | Δacc vs fixed2 [95% CI]
--- | --- | --- | --- | ---
text | 0.593 | 2.00 | 18.86 | +0.007 [-0.010, +0.027]
fixed1 | 0.600 | 1.00 | 10.38 | +0.013 [-0.023, +0.050]
fixed2 | 0.587 | 2.00 | 18.52 | +0.000 [+0.000, +0.000]
fixed4 | 0.610 | 4.00 | 30.90 | +0.023 [-0.010, +0.057]
random_matched | 0.587 | 2.00 | 18.52 | +0.000 [+0.000, +0.000]
oracle | 0.670 | 1.11 | 11.20 | +0.083 [+0.053, +0.117]

tau=0.4326 gate(FAIL): dacc=+0.007 [-0.010, +0.027], seconds 18.86 vs fixed2 18.52

## Task 9

Probe holdout report (kv=f16)

policy | acc | mean_k | mean_seconds | Δacc vs fixed2 [95% CI]
--- | --- | --- | --- | ---
fixed1 | 0.600 | 1.00 | 10.38 | +0.013 [-0.023, +0.050]
fixed2 | 0.587 | 2.00 | 18.52 | +0.000 [+0.000, +0.000]
fixed4 | 0.610 | 4.00 | 30.90 | +0.023 [-0.010, +0.057]
oracle | 0.670 | 1.11 | 11.20 | +0.083 [+0.053, +0.117]
text | 0.593 | 2.00 | 18.90 | +0.007 [-0.010, +0.027]
probe | 0.590 | 2.00 | 18.12 | +0.003 [-0.023, +0.030]
probe+text | 0.590 | 2.00 | 18.27 | +0.003 [-0.023, +0.030]

- Gate: probe 0.590 and probe+text 0.590 vs text 0.593 at matched mean_k 2.0, so neither is >= +1.0 pt. Probe NOT promoted.
- Text failed Task 8, so per plan the chosen policy is None (fixed k=2 with the code floor). No results/policy.joblib is written.
- Note: fixed1 with the code floor (mbpp at k=2) scores (87+48+47)/300 = 60.7% on holdout. That beats fixed2 (58.7%) at lower cost. It is reported on final as a comparison only, not adopted, since it was chosen by looking at holdout.

## Task 10

### Headline: final test (100 items: gsm8k 35, mmlu_pro 35, mbpp 30), run once, no refit, same tau

policy | acc | mean_k | mean_seconds | Δacc vs fixed2 [95% CI]
--- | --- | --- | --- | ---
chosen=None (fixed2+floor) | 0.620 | 2.00 | 20.28 | +0.000 [+0.000, +0.000]
fixed1 | 0.590 | 1.00 | 11.69 | -0.030 [-0.090, +0.030]
fixed4 | 0.630 | 4.00 | 34.67 | +0.010 [-0.040, +0.060]
oracle | 0.670 | 1.12 | 12.61 | +0.050 [+0.010, +0.100]
fixed1+code floor (comparison) | 0.620 | 1.30 | 12.51 | +0.000 [-0.050, +0.050]
text (failed gate, reference) | 0.630 | 1.94 | 20.43 | +0.010 [-0.020, +0.040]

- Chosen policy (None = fixed k=2 + code floor): acc 0.620, mean 20.3 modeled s.
- Neither learned controller (text, probe) beat fixed k=2 on holdout, and on final text is +1.0 pt (CI -2 to +4), which is noise.
- Fixed k=1 for non-code and k=2 for code ties fixed2 on final (0.620) at 12.5 s vs 20.3 s, about 38% less latency. On holdout it was 60.7 vs 58.7. That comparison came after seeing holdout, so it is a recommendation, not a validated result.
- The oracle (0.670 at 12.6 s) shows per-prompt headroom of about 5 pts that neither text features nor last-token hidden states captured.

### Live router gate (holdout, 50 prompts: 17 gsm8k, 17 mmlu_pro, 16 mbpp; scripts/live_gate.py, results/live_gate.json)

- Live accuracy 0.640 vs offline 0.640 for the same ids, 0 per-item disagreements: PASS.
- Controller overhead: text policy 59.6 ms mean (> 50 ms: FAIL), live chosen None policy 43.5 ms. The None policy still runs the NVIDIA classifier just to detect code for max_tokens and the floor. Ruling AE: not fixing, since the text policy is not deployed and 44-60 ms is about 0.3% of a ~20 s request.
- Router decisions are serialized with a lock (the MPS classifier is not thread-safe).

## Task 11
