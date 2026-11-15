# Circuits and interpretation (controller, from 00-04)

What the numbers support, what they don't, and the edit hypotheses that follow. Claims here are hypotheses
until an edit tests them by forward passes (spec rule).

## Failure mode
- Holdout k=1 accuracy 0.577; k=2 0.603; k=4 0.627. The k=4 gain is almost all MMLU-Pro (0.44 → 0.53); gsm8k
  0.84 → 0.88; mbpp 0.45 → 0.47.
- 47/300 research rows truncate. 44 are overlong (no repeated 8-gram, no answer phrase), 1 is a loop that is really
  option enumeration, 2 other. 37 of 47 are MMLU-Pro. Unlike Qwen3.5-0.8B in Sparks, Dory does not get stuck in
  repetition loops; it reasons too long (checks every option at length) and runs out of tokens. So a Sparks-style
  "loop breaker" has nothing to fix; the lever is verbosity and when the model commits to an answer.

## Where the answer forms (02-lens)
- MMLU-Pro answer letter and gsm8k answer first reach top-1 under the logit lens only in the output blocks
  (half-block 56+, i.e. right after the last recurrent loop) for all mmlu items and 60-70% of gsm8k items, at every
  loop count. At loop boundaries inside the recurrence, 0/30 items decode the answer (the residual between loops is
  not in the output basis; the lens is uninformative there, not proof that the answer is absent).
- mbpp's first answer token (`def`) is top-1 by half-block 38 in loop 1: trivial token.

## What predicts failure (03-probes)
- will_truncate is linearly decodable from the prompt-end residual before any token is generated: pooled AUC
  0.90 at half-block 40 (mid recurrent, loop 1) vs 0.80 for source identity alone; within MMLU-Pro 0.90 vs null
  0.44. Within gsm8k weaker (0.69). So "this question will run long" is represented early and mostly in the
  recurrent block. It may encode question difficulty or option complexity rather than a verbosity control.
- will_be_correct within-source is weak for gsm8k (≈ null), moderate for mmlu (0.75-0.77 at 40-56) and mbpp
  (0.82-0.83 at 64-71, output blocks).

## Ablations (04-ablations)
- Whole-half-block skips: layer.56 (first output attention) and layer.34 dominate absolute damage; the largest
  truncated-vs-finished differentials are small (layer.16 −0.18, layer.34 −0.13 nats/token mean).
- Heads (80 measured in the top-5 attention layers) and top-64 neuron groups (5 MLP layers) all have |differential|
  ≤ 0.02. No sparse head or neuron set separates overlong from finished trajectories.
- Conclusion: no localized "keep reasoning" circuit at head/neuron granularity. The behaviour is distributed; the
  evidence favors a direction-level (low-rank) edit over sparse entry edits.

## Edit hypotheses (ordered)
1. **Truncation-direction ablation (token path).** d = mean residual over generated tokens, truncated minus
   finished MMLU-Pro rows (within-source, so source identity is not the direction), at recurrent MLP halves.
   Edit: remove d from the writes of recurrent `down_proj` (ΔW = −α d dᵀW, rank 1 per tensor). Predicted: shorter
   MMLU outputs, fewer truncations, accuracy flat or up because truncated rows currently score 0.
2. **Commit-direction steering.** d = residual at the position that starts the final "The answer is" sentence
   minus the row's mean generated residual. Projecting it out should delay answering (sign check); adding it
   should make the model commit sooner. Needs a constant input to add; candidate: rank-1 on recurrent down_proj with
   the mean normalized input direction.
3. Sparse edits are deprioritized by the ablation result above.

Precheck (before any screen): hook the same projection into the bf16 torch model and measure teacher-forced logprob
of own continuations on research rows: overlong MMLU rows should get less likely, finished-correct rows should not.
