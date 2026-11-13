# Weight-Space Programming for Dory — Design

## Intent

Make Dory1.0-4B smarter and/or faster by having Opus write changes directly into its weights,
in the spirit of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab; GPT-6 Astra edited
0.016% of Qwen3.5-0.8B's weights, fixing reasoning loops; math up to 6x). The skeptic paper
(Ahmed, arXiv 2611.08070) shows Opus reading raw weights mixes real local arithmetic with confident
dataset-prior guesses, so edits must come from measured evidence and be verified by forward passes.

## Method (the user's four rules)

1. Build our own harness with great care.
2. Frontier LLMs bring priors and biases to edits; correctness is our responsibility, without drowning
   in autopilot checks. A few deliberate checks per edit, not every check an LLM can invent.
3. Start with a base objective. Measure what the base model already does. Establish baselines.
   Set up activation probes and a neuron/logit lens, and establish circuits (roughly).
4. Treat the results of 1-3 as a research corpus; Opus proposes edits from the corpus.

## Objective

Primary metric: accuracy at k=1 (one recurrent loop) under the existing token caps
(gsm8k 512, mmlu_pro 512, mbpp 768), pooled over GSM8K + MMLU-Pro + MBPP. Secondary: mean generated
tokens. Rationale: ~1/3 of MMLU-Pro answers truncate at 512; shorter focused reasoning is both
smarter (fewer cut-off answers) and faster (tokens are the latency). k=1 is ~3x cheaper than k=4 and
edits to the shared recurrent block act on every loop.

Success: k=1 accuracy up with mean tokens flat or down, beating a size-matched random edit, on
held-out data, with no per-benchmark regression beyond noise and no ARC-Challenge regression.
Stretch: k=1 reaches k=4 accuracy.

## Data discipline

| Set | Contents | Use | Visibility |
|---|---|---|---|
| Research | existing `train` split | probes, lens, before/after; corpus | free |
| Screen | 60 of `holdout` (20 per source) | fast pass/fail per edit | scores only |
| Selection | all 300 `holdout` | confirm screen survivors | scores only |
| Final | 300 fresh (100 per source, unused ids) | once, at the end | sealed |
| Damage check | 200 ARC-Challenge test | once per selection survivor | scores only |

## Decisions (the controller's calls, made during design)

- "Opus" = the controller session (Claude Opus 5.5) reading corpus files; no separate API pipeline.
- No transformers/PyTorch Dory exists (config has no `auto_map`; transformers 5.14 has no `dory`).
  We reimplement the forward pass in PyTorch from llama.cpp's `src/models/dory.cpp` and require parity
  with llama-server before any probe result is trusted.
- Weights: BF16 safetensors from `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` (~8 GB) into the HF cache.
- Evaluation format: Q8_0 GGUF made by the vendored converter (`vendor/llama.cpp/conversion/dory.py`).
  Base and every edit are evaluated in the same format; old Q5_K_M numbers are not reused as baselines.
- Edits are explicit patch files (tensor name, positions or low-rank factors, operation), applied to
  the BF16 base; every edit is reversible and diffable.
- Controls: each edit is compared to a random edit of the same size in the same tensors.
- Opus never edits from raw weights alone; its mechanism claims are hypotheses that forward passes test.
  Stated confidence carries no weight.
- Circuit claims must survive a content-swap control (behavior follows meaning, not position),
  mirroring the skeptic paper's label permutation.

## Non-goals

- Fine-tuning or gradient-based training as the edit method (allowed only as a reference point, not done here).
- Changing the loop router; this work changes the model.
- Proving the edited model correct in general.

## Anti-goals

- Gains that come from overfitting the selection set (hence the sealed final set).
- Edits that win on the three benchmarks but damage general ability (hence ARC).
- Accepting an edit because Opus's explanation sounds right.
