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
- Note (resolved): tests/test_policy.py::test_text_features_shape_real failed at the time. It passes since the Task 8 classifier-load fix.

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

- **Caveat (final review): mmlu_pro is truncation-bound.** About a third of mmlu_pro outputs hit the 512-token cap at every k: train 246/600, 236/368 and 228/325 at k=1/2/4; holdout 29/38/36. Truncated outputs almost never grade correct, so the mmlu labels largely measure whether the answer fits in 512 tokens, not loop benefit. That limits what the null result says. Per-source accuracy / modeled seconds (breakdown only, no refit):

| split | subset | fixed1 | fixed2 | fixed4 | fixed1+floor | oracle |
|---|---|---|---|---|---|---|
| holdout | gsm8k+mbpp (200) | 0.660 / 8.4s | 0.665 / 14.8s | 0.665 / 24.2s | 0.670 / 10.4s | 0.715 / 8.8s |
| holdout | mmlu_pro (100) | 0.480 / 14.4s | 0.430 / 26.0s | 0.500 / 44.4s | 0.480 / 14.4s | 0.580 / 15.9s |
| final | gsm8k+mbpp (65) | 0.662 / 8.9s | 0.662 / 15.6s | 0.708 / 27.8s | 0.708 / 10.2s | 0.738 / 10.0s |
| final | mmlu_pro (35) | 0.457 / 16.8s | 0.543 / 29.0s | 0.486 / 47.5s | 0.457 / 16.8s | 0.543 / 17.5s |

- Next step before any further controller work: re-sweep mmlu_pro with max_tokens 1024 (or a short-answer prompt). Ruling AF: not done in this run, because it needs about 6+ GPU hours, and the final split's one-shot evaluation is already spent.

### Live router gate (holdout, 50 prompts: 17 gsm8k, 17 mmlu_pro, 16 mbpp; scripts/live_gate.py, results/live_gate.json)

- Live accuracy 0.640 vs offline 0.640 for the same ids, 0 per-item disagreements: PASS.
- Controller overhead: text policy 59.6 ms mean (> 50 ms: FAIL), live chosen None policy 43.5 ms. The None policy still runs the NVIDIA classifier just to detect code for max_tokens and the floor. Ruling AE: not fixing, since the text policy is not deployed and 44-60 ms is about 0.3% of a ~20 s request.
- Router decisions are serialized with a lock (the MPS classifier is not thread-safe).

## Task 11

Originally SKIPPED (llama-server copies the target's kv_overrides into the draft). Unskipped as a follow-up: target = stock GGUF (metadata `dory.num_loops=4`, no override), draft = APFS clone with `dory.num_loops=1` (`scripts/make_draft.sh`), `--spec-type draft-simple --spec-draft-n-max 4`.

- 100-prompt bench (`results/spec_k4.json`): acceptance 0.84, 20/20 outputs identical. Its 0.29x speed figure is an artifact: spec and plain arms ran hours apart under varying machine load.
- Draft-length sweep (`results/draft_sweep.jsonl`, 24 holdout prompts, 256 tokens, sequential arms): n2 6.92, n4 7.54, n6 7.53, n8 6.50, n8 p-min 0.7 6.80 tok/s; k=2 draft 5.46 (n4) / 2.21 (n8); CPU draft 0.12. Per-prompt best-n_max oracle 8.02 vs fixed n4 7.54 (+6%): a confidence-driven draft length is not worth a llama.cpp patch. A clean sequential plain rerun (9.70) beat every arm, which showed sequential arms are unreliable here.
- Interleaved A/B (`scripts/spec_ab.py`, `results/spec_ab.jsonl`; all servers up, arms rotated per prompt): plain 8.39, n4 9.88, n6 10.23 tok/s. n4 1.21x (95% CI 1.16-1.26, 22/24 prompts faster), n6 1.27x (1.19-1.34). 24/24 identical. Acceptance n4 0.84, n6 0.76.
- Gate (acceptance >= 0.80, speedup >= 1.2x, identical): PASS with n4. Router runs the k=4 server with the draft by default (`--no-spec` disables). Moot for the chosen None policy (fixed k=2 starts no k=4 server).
- CPU/GPU split (`results/offload_ab.jsonl`, interleaved, 12 prompts): `-ngl 54` of 72 layers 0.94x plain, same graded accuracy (4/8). Full GPU stays.

## Follow-ups (2026-11-12)

- q8_0 KV check (`results/sweep_holdout_q8.jsonl`, k=4, 200 gsm8k+mmlu holdout): f16 134 vs q8 131 (-1.5 pts; gsm8k 85 vs 82, mmlu 49 = 49), McNemar p=0.51. Gate (within 1.0 pt) FAIL: CHOSEN_KV stays f16. q8 saves ~1.5 GB per server if memory matters.
- Memory gate, router config (3 k-servers at slots=4 + embedding server): +8.28 GB, swap growth 0 MB: PASS.
- Grader made truncation-aware and re-applied (`scripts/regrade.py`): True->False on 4 holdout, 2 final, 5 train rows. Labels rebuilt: 11 of 1900 rows changed. Text policy, probe and final test rerun below (Ruling AJ: rerun allowed after a grader fix; both versions recorded). Conclusions unchanged: text policy fails its gate, probe not promoted, chosen policy None.
- Final test after regrade: chosen fixed2+floor 0.620 (20.28 s), fixed1 0.590, fixed4 0.610 (was 0.630), oracle 0.670, fixed1+code floor 0.620 (12.51 s), text 0.630.

## Task 8 (rerun after regrade)

policy | acc | mean_k | mean_seconds | Δacc vs fixed2 [95% CI]
--- | --- | --- | --- | ---
text | 0.583 | 2.00 | 18.77 | -0.003 [-0.020, +0.013]
fixed1 | 0.590 | 1.00 | 10.38 | +0.003 [-0.033, +0.043]
fixed2 | 0.587 | 2.00 | 18.52 | +0.000 [+0.000, +0.000]
fixed4 | 0.607 | 4.00 | 30.90 | +0.020 [-0.013, +0.053]
random_matched | 0.587 | 2.00 | 18.52 | +0.000 [+0.000, +0.000]
oracle | 0.667 | 1.12 | 11.26 | +0.080 [+0.050, +0.113]

tau=0.4312 gate(FAIL): dacc=-0.003 [-0.020, +0.013], seconds 18.77 vs fixed2 18.52

## Probe holdout report (kv=f16, rerun after regrade)

policy | acc | mean_k | mean_seconds | Δacc vs fixed2 [95% CI]
--- | --- | --- | --- | ---
fixed1 | 0.590 | 1.00 | 10.38 | +0.003 [-0.033, +0.043]
fixed2 | 0.587 | 2.00 | 18.52 | +0.000 [+0.000, +0.000]
fixed4 | 0.607 | 4.00 | 30.90 | +0.020 [-0.013, +0.053]
oracle | 0.667 | 1.12 | 11.26 | +0.080 [+0.050, +0.113]
text | 0.583 | 2.00 | 18.77 | -0.003 [-0.020, +0.013]
probe | 0.590 | 2.00 | 18.13 | +0.003 [-0.023, +0.033]
probe+text | 0.590 | 2.00 | 18.18 | +0.003 [-0.023, +0.033]

## Neural Engine (Core ML) classifier check

`scripts/ane_bench.py`, `results/ane_bench.json`: NVIDIA classifier converted to Core ML (fp16 mlprogram, static buckets L=128/512; 59/60 holdout prompts fit 128), batch 1, tokenization included.

- Core ML CPU_AND_NE: median 17.9 ms (p90 18.2) vs PyTorch MPS 44.3 ms and CPU 44.8 ms: 2.5x faster. Parity vs PyTorch fp32 on 60 prompts: argmax identical on all 8 heads, max logit diff 0.024, router feature max diff 0.0038, is_code 100% agreement. MLComputePlan: 454/516 ops on the Neural Engine (L=128), 478/516 (L=512).
- CPU_AND_GPU, CPU_ONLY and ALL produce NaN logits (fp16 overflow off the ANE path): unusable as converted.
- Cost: ~2.1 GB RSS (vs ~0.9 GB PyTorch) and ~16 s first load. Conversion needs two DeBERTa monkeypatches (in the script only). coremltools segfaults freeing Core ML objects; the bench isolates each arm in a subprocess ending with `os._exit(0)`.
- Not wired into the router: the deployed None policy needs the classifier only for is_code; ANE would cut that 44 ms to ~18 ms (about 0.1% of a ~20 s request) at +1.2 GB RAM.
