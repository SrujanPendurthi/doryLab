# Adaptive loop-count methods for looped (recurrent-depth, weight-shared) LMs

Scope: inference-time choice of how many loops a looped LM runs, per sequence or per token. Sources are primary papers and model cards where I could read them; items marked **[snippet only]** come from search summaries I could not verify in full text. Date context: Nov 2026.

## Key question 1: Looped / recurrent-depth LMs with adaptive exit (exit criteria, savings vs accuracy)

### Takeaway
Two published families cover most of the field. Huginn-0125 uses a training-free convergence rule (KL between successive steps < 5e-4) applied zero-shot. Ouro (ByteDance LoopLM) trains a per-step learned exit gate with an entropy regularizer. Neither paper reports a clean "FLOPs saved vs accuracy lost" number; both report accuracy-vs-fixed-depth curves instead. Mixture-of-Recursions is a different design (router picks depth per token at train and inference).

### Cited Findings

**Huginn-0125 (Geiping et al., 2025)**
- Exit rule: "the KL-divergence between two successive steps"; if it "falls below 5×10⁻⁴, we stop iterating, sample the output token." Per-token, zero-shot (model not trained for early exit). Section 6.1, Figure 10. — [arXiv 2502.05171 v2 HTML](https://arxiv.org/html/2502.05171v2)
- The paper gives no FLOP or wall-clock savings figure for early exit. Its only early-exit quality check is MTBench: 5.63 standard vs 5.56 with early exit. Authors say early exit "would be more efficient in practice" (motivation, not measured). Same source.
- Reported per-step accuracy (zero-shot, 3.5B, 0.8T tokens; r = recurrent steps):
  - ARC-E: 49.07 (r=4), 65.11 (r=8), 69.49 (r=16), 69.91 (r=32)
  - ARC-C: 27.99, 35.15, 37.71, 38.23
  - HellaSwag: 43.46, 58.54, 64.67, 65.21
  - MMLU: 23.39, 25.29, 31.25, 31.38
  - GSM8K CoT (strict/flexible) at r=32: 34.80/42.08; r=1 gives 0.00/0.00. Same source, Tables 1 and 4.
  - ARC-C saturates around 8–12 iterations with no few-shot examples, about 20 with one, about 32 with 25–50. Same source, Figure 9.
- Headline claim (abstract): performance improves "up to a computation load equivalent to 50 billion parameters." — [arXiv 2502.05171](https://arxiv.org/abs/2502.05171)
- Model card documents four stopping rules selected with `criterion=`: `entropy-diff`, `latent-diff`, `kl`, `argmax-stability`. Example `exit_threshold=5e-4`. No per-criterion defaults are given. Recommendation: "use `kl` for interesting exits and `argmax-stability` for conservative exits." Caveat: "The model is not trained for this, so this is a proof-of-concept." Released Apache-2.0. — [HF tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125)
- Model card: adaptive compute uses `generate_with_adaptive_compute`; requires `HuginnDynamicCache`, with `lookup_strategy="latest-m4"` for adaptive compute and `compress-s16` for KV sharing. Fewer than 4 steps "will result in very coarse answers." Benchmarks improve up to about `num_steps=64`. Same HF page.

**Ouro / LoopLM (ByteDance, "Scaling Latent Reasoning via Looped Language Models," arXiv 2510.25741)**
- Per-step gate: λ_t(x) = σ(Linear_φ(h^(t))), an instantaneous exit probability. Survival S_t = Π(1−λ_j); exit distribution p_t = λ_t·S_{t−1}; leftover mass goes to the final step. Training loss is expected task loss minus β·entropy(p_φ), with a uniform prior over exit steps ("decouple exit decisions driven by input difficulty from any global compute preference"). Section 3.2–3.3. — [arXiv 2510.25741 v2](https://arxiv.org/html/2510.25741v2)
- Adaptive exit results (MMLU, Section 5.4.1, Figure 5): trained gate reaches 66% at average exit round 2.5, untrained/standard gate about 64%, a gap of about 2–3 points at most operating points. A hidden-state threshold gets within 1–2 points of the trained gate at 2–3 average rounds. Static baseline: 40% at 1 round, 60% at 2 rounds, 67.35% at 4 rounds. The paper reports no % compute saving.
- Fixed-depth numbers (Tables 10–13), MMLU T=1→4: Ouro-1.4B 41.21 → 60.43 → 66.71 → 67.45; Ouro-2.6B 51.55 → 67.63 → 73.57 → 74.60. Ouro-2.6B "tends to peak slightly earlier, at T=3 or T=4." Same source.
- Thinking variant, fixed depth: Ouro-2.6B-Thinking AIME 2024 T=1→4: 3.00 → 52.00 → 70.33 → 64.70; OlympiadBench T=1→4: 18.96 → 68.59 → 75.56 → 76.44. Same source.
- Abstract claims Ouro 1.4B and 2.6B "match the results of up to 12B SOTA LLMs." — [arXiv 2510.25741](https://arxiv.org/abs/2510.25741)
- Model cards: `total_ut_steps` (default 4) and `early_exit_threshold` (search summary says default 1.0, meaning no early exit; **[snippet only]**). — [HF ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/README.md)
- Adaptive exit is **not supported in vLLM** per a secondary hosting page; treat as **[snippet only]**. HF card transformers version guidance conflicts across sources (transformers<4.56 vs ==5.9.0); check the official card before running.

**Mixture-of-Recursions (MoR, Bae et al., arXiv 2507.10524 v3)**
- Not a post-hoc exit. A router assigns each token a recursion depth; trained from scratch so train and inference match. Two variants: expert-choice (each recursion step selects top-k tokens; auxiliary loss for inference) and token-choice (single top-1 decision from the first block fixes depth for the whole pass). Best token-choice config: softmax with MLP router and balancing loss. — [arXiv 2507.10524 v3 HTML](https://arxiv.org/html/2507.10524v3)
- Numbers, equal training compute (16.5e18 FLOPs): vanilla 315M params NLL 2.7824, few-shot avg 42.3; MoR expert-choice 2 recursions, 167M params, NLL 2.7511, few-shot avg 43.1. At 20B tokens, 25% fewer training FLOPs, 19% less training time, 25% less peak memory. Throughput: MoR-4 "up to 2.06×" at max batch on 360M-scale models, with slight quality cost. Same source.
- Code: [github.com/raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions) (linked from the arXiv page; not opened).

**Relaxed Recursive Transformers (Bae et al., ICLR 2025, arXiv 2410.20672)**
- Converts a pretrained LLM into a shared block looped multiple times, with per-depth LoRA to relax tying. Rank is the knob between vanilla and fully tied. This is a parameter-sharing conversion, not an adaptive-exit method. I did not read its benchmark numbers. — [arXiv 2410.20672 v2](https://arxiv.org/pdf/2410.20672v2), [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/54d6a55225cebbdc16fbb0e45c5bdf2b-Paper-Conference.pdf)

**Newer, from search snippets only [snippet only]**
- LoopFormer (arXiv 2602.11451): reports that adaptive early-exit looped models show representational collapse; proposes shortcut-modulated elastic depth. — [arXiv 2602.11451](https://arxiv.org/pdf/2602.11451)
- Second-order "acceleration" exit (arXiv 2509.23314 v2): decoding-free, O(d) per token-step vs O(V) for KL exits; authors claim better latency-quality than step-norm and KL baselines (authors' own claim). — [arXiv 2509.23314 v2](https://arxiv.org/html/2509.23314v2)
- Looped State-Space LMs with adaptive exit-state selection (arXiv 2607.10110). — [alphaxiv](https://www.alphaxiv.org/abs/2607.10110.md)

### Inferences
- The only zero-training exit with an explicit threshold on a published model is Huginn's KL rule (5e-4). Its paper reports no savings table, so "how much compute you save" must be measured on your own model and task.
- Ouro's trained gate has the most published detail, but its own numbers show the advantage over a hidden-state threshold is small (1–2 points at 2–3 rounds).
- Accuracy cost of early exit is largely unquantified in the primary papers. Budget your own evaluation.

### Gaps
- No primary-source number for "% FLOPs saved at X% accuracy loss" for either Huginn or Ouro.
- Full text of MoR's exit-quality numbers beyond perplexity/few-shot was not extracted.

## Key question 2: Per-token vs per-sequence adaptivity, KV cache, batching

### Takeaway
Per-token exit forces a KV policy for skipped depths. Huginn attends to the deepest available KV. Ouro's own paper reports near-lossless last-step KV reuse, but an independent reproduction (CDB, 2026) could not match that and found large losses. Batching is solved by a 2026 scheduler (Continuous Depth Batching) that reports 1.5–1.9× throughput over fixed-depth batching.

### Cited Findings

**Huginn KV handling (Section 6.1–6.2, Remark 6.1)**
- Early-exited tokens: "we attend to the last, deepest available KV states in the cache," justified because successive recurrent KVs come from the same projections of successive states. — [arXiv 2502.05171 v2](https://arxiv.org/html/2502.05171v2)
- KV sharing: iteration i reads/writes cache entry i mod k. Paper says this "does not reduce performance" on MTBench (cache budget 4: 5.86 vs 5.63 standard). The text renders the budget as "44"; verify against Appendix Table 6.

**Ouro KV handling (Section 5.4.2, Table 14)**
- Naively each step needs its own cache, "leading to 4×" memory for the 4-step model. Reusing caches during prefill drops GSM8K by more than 10 points.
- Decoding, GSM8K / MATH-500: full 4× cache 78.92 / 82.40; first-step only 18.73 / 8.43; last-step only 78.85 / 80.40; averaged 78.73 / 78.52. Last-step reuse is the paper's recommended 4× reduction. — [arXiv 2510.25741 v2](https://arxiv.org/html/2510.25741v2)

**Continuous Depth Batching (CDB, arXiv 2608.09444, 2026)**
- Schedules per loop iteration rather than per token. Splits decoding into prelude, recurrent core, and coda queues. Exit decisions route tokens between queues. Modes: no-refill (batch shrinks as tokens exit) and refill (freed slots get new work). Uses a shared KV cache with one slot per layer, overwritten each loop step. A lookahead gate decides one step early so the CPU can prepare the next batch. — [arXiv 2608.09444 HTML](https://arxiv.org/html/2608.09444v1)
- Throughput: abstract reports "1.5–1.9× higher offline throughput and 45–90% lower normalized latency under dynamic serving load." Ouro refill reaches 94–99% of the FLOP bound; Huginn up to 1.9× at 73–92% of bound. Without early exits CDB stays within 2% of a plain continuous-batching baseline. Online (moderate load): refill cuts normalized latency 45–57% on ShareGPT and 84–88% on Alpaca. Same source, Sections 6.1–6.3.
- Shared-KV accuracy (its reproduction, GSM8K): Ouro full depth-indexed cache 77.86%, first-then-shared 71.34%, single-slot 0.23%. The authors state they "could not reproduce the near-lossless sharing reported in the Ouro paper." Same source, Appendix.
- Limitations: lookahead gate forces r_min=2 (3 for convergence criterion), limiting savings at low r_max. FLOP bound excludes prefill, scheduling, KV overhead. No chain-of-thought / reasoning-length evaluation. Only shared KV layout evaluated.
- Huginn adaptive exit "applies only during decoding. Prefill always runs at full depth." — [search summary of arXiv 2608.09444; verified in full text above](https://arxiv.org/html/2608.09444v1)

**MoR KV (Section 3 of its paper)**
- Recursion-wise caching: only routed tokens store K/V at that depth, attention restricted to cached tokens. Recursive KV sharing: all tokens pass block 1, KV cached once and reused in later recursions. — [arXiv 2507.10524 v3](https://arxiv.org/html/2507.10524v3)

### Inferences
- Per-token exit with a fixed-slot KV layout is the main engineering cost. Keep one KV slot per layer, or accept the deepest-available policy (Huginn style).
- Given the CDB result, do not assume published "near-lossless" KV sharing transfers to your model. Measure with your own tasks.
- Per-sequence exit is simpler (whole batch shrinks or finishes together) but wastes compute on easy tokens in the same batch. Per-token exit needs a scheduler like CDB to realize savings under batching.

### Gaps
- No public serving framework (vLLM, SGLang) with adaptive exit for looped models was confirmed. vLLM lacks Ouro adaptive exit per a secondary source.
- CDB code repository was not located.

## Key question 3: Zero-training heuristics vs trained controllers (prompt classifier, hidden-state probe, RL policy)

### Takeaway
No paper I could read runs a controlled accuracy-vs-compute Pareto across all three families on one model. Best-supported evidence: a training-free convergence rule works on Huginn (zero-shot), a trained per-step gate beats a hidden-state threshold by only 1–2 points at matched depth on Ouro, and a prompt-level (pre-loop) gate performs poorly. Trained RL controllers exist (2026) but I could not verify their results.

### Cited Findings
- Huginn convergence rule (KL < 5e-4) is zero-shot and per-token. — [HF Huginn card](https://huggingface.co/tomg-group-umd/huginn-0125)
- Ouro trained gate vs hidden-state threshold: trained 66% vs untrained ~64% at avg round 2.5; hidden-state threshold within 1–2 points of trained at 2–3 rounds. — [arXiv 2510.25741 v2](https://arxiv.org/html/2510.25741v2)
- CDB (2608.09444) compares: Ouro's learned gate, a distilled lookahead gate (matches teacher accuracy), a distilled pre-loop gate that predicts exit before the loop starts ("does not perform well"), and Huginn's convergence criterion. Gated models "Pareto-dominate fixed exits." Ouro performs well down to mean depth ≈2.5; Huginn down to ≈8. Figure 10 values are in the figure, not the text. — [arXiv 2608.09444](https://arxiv.org/html/2608.09444v1)
- 2026 preprint "Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts" (arXiv 2607.20519) directly asks whether trajectory signals compete with trained gates. I could not read the PDF (binary-only extraction), so no numbers. — [arXiv 2607.20519](https://arxiv.org/pdf/2607.20519)
- RL controllers (search snippets only **[snippet only]**): ReLAR (arXiv 2606.17524) uses RL-trained depth and action controllers for refinement steps on hidden states, evaluated on medical, math, multi-hop, open-ended generation. — [alphaxiv](https://www.alphaxiv.org/abs/2606.17524.md). LoopRPT (arXiv 2603.19714) adds a time penalty during RL pretraining to encourage earlier exits. — [arXiv 2603.19714](https://arxiv.org/pdf/2603.19714). RLTT (arXiv 2602.10520) distributes reward across the latent trajectory. — [arXiv 2602.10520](https://www.arxiv.org/pdf/2602.10520)
- A 2026 paper (arXiv 2603.21376, **[snippet only]**) calibrates exit probabilities by KL between intermediate and final logits as teacher-student targets.

### Inferences
- Cheapest viable starting point: convergence threshold (Huginn-style) with a sweep of the threshold. It needs no training and gives an accuracy-vs-depth curve in an afternoon.
- If a trained controller is worth it, the evidence favors a per-step gate on the loop's own hidden state (Ouro style, or distilled lookahead as in CDB) over a prompt-only classifier. The pre-loop prompt gate performed poorly in CDB.
- Predicted gain from trained gates is modest at matched depth (1–3 points on MMLU in Ouro). Whether it survives on reasoning tasks is unknown.

### Gaps
- No head-to-head Pareto of zero-training vs trained controllers on the same looped model with published numbers.
- 2607.20519 and the RL controller papers could not be verified in full text.

## Key question 4: Evaluation methodology (accuracy vs average loops, benchmarks)

### Takeaway
The field reports accuracy vs fixed recurrent depth (step-count curves) and occasionally accuracy vs mean exit depth. Common benchmarks: ARC-E/C, HellaSwag, MMLU, GSM8K (CoT, flexible match), MATH-500, AIME, OlympiadBench, MTBench, HumanEval. Reporting average exit depth alongside accuracy is the convention that lets you compare exit rules.

### Cited Findings
- Huginn: accuracy at r=4/8/16/32 on ARC-E, ARC-C, HellaSwag, MMLU (zero-shot), plus GSM8K CoT at r=1 vs 32. — [arXiv 2502.05171 v2](https://arxiv.org/html/2502.05171v2)
- Ouro: fixed T=1..4 (and T=8 extrapolation) on ARC-C, MMLU, OlympiadBench, AIME 2024/2025, SuperGPQA; accuracy vs average exit round for trained vs untrained gate (MMLU). — [arXiv 2510.25741 v2](https://arxiv.org/html/2510.25741v2)
- CDB: Pareto plots of GSM8K (gsm8k_cot, flexible match) accuracy vs mean loop depth; latency and throughput on Alpaca and ShareGPT with replayed exit decisions. — [arXiv 2608.09444](https://arxiv.org/html/2608.09444v1)
- MoR: validation NLL and few-shot average at equal FLOPs and equal data; throughput at batch sweep. — [arXiv 2507.10524 v3](https://arxiv.org/html/2507.10524v3)
- Huginn card: benchmarks improve up to num_steps≈64, then plateau. — [HF card](https://huggingface.co/tomg-group-umd/huginn-0125)

### Inferences
- A usable Pareto plot needs: accuracy vs mean exit depth for each threshold, against fixed-depth at the same mean depth (CDB's comparison). Also report tokens/sec under batching, since FLOP savings do not translate directly (CDB bound is 73–99%).
- Use reasoning benchmarks (GSM8K, MATH-500) as well as knowledge MCQ (MMLU, ARC). Huginn's MMLU is nearly flat until r=8 and its GSM8K is 0 at r=1, so the benchmark chosen changes the conclusion.

### Gaps
- No paper published a full accuracy-vs-average-loops table for a looped LM on reasoning tasks with the exit rule swept, in the text I could read.

## Key question 5: GitHub repos and code for adaptive exit in looped models

### Takeaway
Official code: Huginn (training repo and HF model with built-in adaptive compute). Ouro has model weights on HF; a community reimplementation exists (rkstgr/LoopLM). MoR has official code. CDB, ReLAR, and the 2026 gate-diagnosis paper: code not located.

### Cited Findings
- **seal-rg/recurrent-pretraining** (Huginn training code and data recipe). Linked from the Huginn paper and HF card. [github.com/seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining). — [arXiv 2502.05171](https://arxiv.org/abs/2502.05171)
- **tomg-group-umd/huginn-0125** (HF model, Apache-2.0). Built-in `generate_with_adaptive_compute` with criteria `kl`, `entropy-diff`, `latent-diff`, `argmax-stability`, `exit_threshold`. Needs `HuginnDynamicCache`. [HF card](https://huggingface.co/tomg-group-umd/huginn-0125). **This is the most directly usable working implementation of a training-free exit.**
- **ByteDance/Ouro-1.4B, Ouro-2.6B** (HF, weights plus model code for the learned gate). [HF Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/README.md). Watch the transformers version conflict noted above.
- **rkstgr/LoopLM** (community reimplementation of Ouro; learned exit gate, Stage I entropy-regularized loss, Stage II adaptive gate loss, early-exit strategies Q-exit and hidden-state diff, KV cache sharing). 58 commits, 40 stars, 7 forks; README shows no benchmark results; license not seen; some checklist items unchecked. [github.com/rkstgr/LoopLM](https://github.com/rkstgr/LoopLM). Treat as unverified research code.
- **raymin0223/mixture_of_recursions** (MoR official code). [GitHub](https://github.com/raymin0223/mixture_of_recursions). Not opened.
- **merlresearch/ANIRA**: per-token variable-depth recurrent Transformer framework, PyTorch/CUDA research code. Not an Ouro-style exit. [github.com/merlresearch/ANIRA](https://github.com/merlresearch/ANIRA)
- **koayon/awesome-adaptive-computation**: curated list. [git.tdem.in/koayon/awesome-adaptive-computation](https://git.tdem.in/koayon/awesome-adaptive-computation). Not opened; useful to find remaining code.
- LoopFormer project page: [loopformer.github.io](https://loopformer.github.io). Code not found in search.

### Inferences
- For an implementer picking an approach: start from Huginn's HF adaptive-compute path (zero-training, works today), measure the accuracy-vs-depth curve, then decide whether a trained gate (Ouro recipe, via rkstgr/LoopLM as a reference) is worth it. Add a CDB-style scheduler only if batched throughput matters.
- Check licenses on rkstgr/LoopLM and CDB before reuse.

### Gaps
- Could not confirm official Ouro training code, CDB code, or the RL controller repos.
- Huginn README-level threshold defaults were not stated; sweep yourself.

## Overall recommendation (inference from the above)
1. Baseline: Huginn-style convergence exit (KL or argmax-stability) on the target looped model, zero-training, sweep the threshold, plot accuracy vs mean depth on GSM8K and MMLU.
2. If the baseline is poor on reasoning, try a per-step learned gate on the hidden state (Ouro recipe), not a prompt classifier.
3. For KV, use last-step reuse (Ouro Table 14) as the first candidate, but verify against CDB's reproduction, which found large losses.
4. For batched serving, a CDB-style per-iteration scheduler is the only published approach with measured throughput gains (1.5–1.9×).
