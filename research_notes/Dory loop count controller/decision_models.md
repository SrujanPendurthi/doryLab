# Decision / Routing / Difficulty Models as an External Loop-Count Controller for Dory

Scope: open-weight models small enough to co-run with a 4B model on a 24GB Apple M4. Sources checked Nov 2026 via search snippets and HF model-card fetches. Dory's own architecture is not documented in this project (no notes found under doryAgent), so the Dory-side sections are conditional.

## What is "Laya"?

### Takeaway
Laya is an open-weight, Apache-2.0 "non-autoregressive decision model" from Convai Innovations. It is a ~0.4B-421M encoder classifier that answers typed structured questions (choice, ordinal score, yes/no probability) in one forward pass. It is not a generative model and not a router in the RouteLLM sense, but it is a plausible small controller head if loop counts are framed as a `choice` question.

### Cited Findings
- Model is `convaiinnovations/laya`, Apache-2.0, Transformers/Safetensors, listed at 0.4B params — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Positioned as "non-autoregressive System 1 decision model" that "never generates text" — [HF model card](https://huggingface.co/convaiinnovations/laya)
- English root checkpoint: ModernBERT-large backbone (395M) plus a 2-layer decision head trained from scratch, total 421M, 512-token context — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Multilingual checkpoint: mmBERT-base, 322M total, 1,024-token default (up to 8,192 with `max_len`), "100+ languages" — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Inputs: a state (text/email/ticket/JSON) plus typed questions; outputs: probabilities per option, e.g. `result["answers"][name]["noul"]` — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Latency on Tesla T4 (vendor-reported): 39.5 ms single question (English), 32.8 ms (multilingual); 103–332 questions/s batched — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Benchmarks (vendor-reported): typed-decisions accuracy 0.766 fine-tuned vs 0.362 base English; AG News 0.950; Banking77 (77 labels) only 0.425 vs 0.870 for "Jev" — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Limitations stated by authors: base checkpoints near chance zero-shot on typed decisions; accuracy degrades with >20 options (shared token budget); ordinal `score` is weakest; `noul` can follow option labels; probabilities over-confident before temperature fitting (ECE 0.466 → 0.081) — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Secondary coverage describes it as "open-weight decision model" from Convai Innovations, with hallucination claims disputed — [SourceForge/Gradually/aiweekly coverage found via search](https://www.gradually.ai/en/ai-models/laya/), [aiweekly](https://aiweekly.co/alerts/laya-open-sources-a-33ms-multilingual-typed-decision-engine)

### Inferences
- Laya as-is cannot see Dory's hidden states; it only reads text. It is a pre-hoc controller (see Q4).
- Loop count fits Laya's `choice` primitive: options "1".."K" with K ≤ ~8 stays under the high-cardinality failure mode the authors describe. Needs fine-tuning on (prompt → min-loops-correct) labels; base checkpoint is near chance.
- Memory is trivial (~0.9 GB fp16) next to a 4B model on 24GB.

### Gaps
- Ambiguity: no other "Laya" open-source decision/routing model surfaced. The name collides with unrelated products (e.g. the SourceForge "laya" project, which is a different thing). Confirm with the user that Convai's `laya` is the intended one.
- Vendor and secondary numbers only; no independent evaluation found.
- Base training corpus not documented on the card.

## Other open-source router / difficulty / complexity models

### Takeaway
The best-documented small options are NVIDIA's prompt-task-and-complexity-classifier (0.2B DeBERTa, NVIDIA Open Model License, scores six complexity dimensions) and Katanemo's Arch-Router-1.5B (Qwen2.5-based, generative route-label output). RouteLLM is open (Apache-2.0) but targets model selection, not loop budgets. Academic budget predictors exist (2602.01237) but do not appear to have public small checkpoints. None is trained for loop counts; all need relabeling.

### Cited Findings
- **NVIDIA prompt-task-and-complexity-classifier**: 0.2B params, DeBERTa-v3-base backbone with mean pooling and per-output heads; outputs 11 task types plus six complexity scores (creativity, reasoning, contextual knowledge, domain knowledge, constraints, few-shots) in 0–1; overall weighted score; NVIDIA Open Model License, "ready for commercial use"; 512-token default; English only; trained on 4,024 human-annotated English prompts; 10-fold accuracy ~0.94–0.997 per dimension (domain knowledge lowest at 0.937); no latency numbers; tested on A10G with NVIDIA GPU requirement (Volta+, CUDA 12) — [HF model card](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier)
- NVIDIA's own docs describe it for routing prompts to suitable models and data curation — [NVIDIA blog via search](https://developer.nvidia.com/blog/?p=94447), [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/prompt-task-and-complexity-classifier). A community ONNX conversion exists — [preflight ONNX](https://huggingface.co/preflight/prompt-task-and-complexity-classifier-ONNX). Note: the GPU-only requirement may need an ONNX/CPU path on Mac (inference: MPS or ONNX Runtime CoreML; untested).
- **Arch-Router-1.5B** (Katanemo): paper arXiv 2506.16655 ("Aligning LLM Routing with Human Preferences"), reported 93.17% routing accuracy (authors' claim); takes JSON route definitions plus conversation inside a fixed prompt and emits `{"route": "..."}`; base Qwen2.5-1.5B-Instruct lineage; Katanemo research license; HF card lists "2B params" in safetensors metadata despite the 1.5B name; no latency numbers — [paper](https://arxiv.org/abs/2506.16655), [HF card](https://huggingface.co/katanemo/Arch-Router-1.5B)
- **RouteLLM** (LMSYS): Apache-2.0; matrix-factorization router `routellm/mf_gpt4_augmented` (hidden size 128); MT-Bench result of 95% of GPT-4 quality at 26% GPT-4 calls (14% with augmented data), per LMSYS blog — [GitHub](https://github.com/lm-sys/RouteLLM), [blog](https://lmsys.org/blog/2024-07-01-routellm), [PyPI](https://pypi.org/project/routellm). Routes between model endpoints, so it is a strong-vs-weak-model switch, not a loop-count predictor.
- **Predictive Scheduling for Efficient Inference-Time Reasoning** (Brown, Muppidi, Shahout; arXiv 2602.01237, Feb 2026): trains small predictors before generation to estimate reasoning length/difficulty. Two variants: an MLP on intermediate transformer hidden states (middle layers 12–17 most informative) or a LoRA classifier on raw question text. A greedy allocator splits a fixed token budget; reports up to 7.9 pp GSM8K gain over uniform budgeting at equal cost; coarse classes beat fine regression — [arXiv 2602.01237](https://arxiv.org/abs/2602.01237). Code/checkpoint availability not verified.
- **SelfBudgeter** (arXiv 2505.11274): model pre-estimates reasoning cost, then budget-guided RL; up to 74.47% length compression on MATH — [arXiv 2505.11274](https://arxiv.org/pdf/2505.11274v1). Reported via search snippet only.

### Inferences
- The most directly reusable architecture is the NVIDIA classifier shape (small encoder, multi-head, label outputs): swap its heads for a loop-count head trained on your labels. 0.2B is comfortable on a Mac.
- The 2602.01237 "hidden-state MLP" approach is the closest published match to a difficulty probe for a looped model, and it can be done with Dory's own hidden states (Q4).
- Arch-Router is heavier (1.5B generative) and outputs text labels; poor fit for a numeric decision unless you reframe loop counts as route names.

### Gaps
- No small open checkpoint found specifically predicting recurrence or loop depth; all candidates above are for model selection or reasoning length.
- Latency on M4 not measured for any candidate.
- Predictive-Scheduling checkpoints/code not confirmed.

## Learned halting approaches usable as an external controller

### Takeaway
Ouro (ByteDance, Apache-2.0, 1.4B and 2.6B) is the one open looped LM with a built-in learned exit gate; its gate is in-model, not external. Classical ACT and PonderNet are training-time methods that require the model itself to be trained with the halting head. For an external controller on a fixed looped model like Dory, the cheapest trainable options are probes on hidden states (MLP or logistic) and early-exit classifiers, not ACT/PonderNet.

### Cited Findings
- **Ouro / LoopLM**: arXiv 2510.25741 ("Scaling Latent Reasoning via Looped Language Models", Oct 29, 2025); 1.4B and 2.6B; "match the results of up to 12B SOTA LLMs"; trained on 7.7T tokens; uses an "entropy-regularized objective for learned depth allocation" — [arXiv abstract](https://arxiv.org/abs/2510.25741), [HF card 1.4B](https://huggingface.co/ByteDance/Ouro-1.4B). Apache-2.0 on the 1.4B card. Default `total_ut_steps` = 4; `early_exit_threshold` default 1.0 (never exits early); context 4K train, extendable to 64K — [HF card 1.4B](https://huggingface.co/ByteDance/Ouro-1.4B). Safetensors metadata says "1B params" vs card's 1.4B (unexplained).
- Ouro 2.6B card recommends transformers ≤ 4.54.1 due to KV-cache issues; llama.cpp GGUF conversion reportedly does not work for the architecture — [HF card 2.6B](https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/README.md), [InsiderLLM review](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/). vLLM always runs the full step count — [HF card 1.4B](https://huggingface.co/ByteDance/Ouro-1.4B).
- Follow-up analysis: with exit-state selection, all recurrent steps are still executed, so the selected exit step measures prediction depth, not wall-clock savings — [arXiv 2607.10110 / search summary](https://www.alphaxiv.org/abs/2607.10110.md).
- arXiv 2607.20519 (Popescu, Sáez de Ocáriz Borde, Liò; "Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts", CC BY 4.0) argues that learned halting uses a single exit distribution both for stopping and for weighting per-depth losses, tested on Ouro-1.4B/2.6B — [PDF](https://arxiv.org/pdf/2607.20519). I could read only metadata from the PDF, so its claims are unverified here.
- AdaPonderLM (Mar 2026) learns token-wise early exit during pretraining; gates allocate more compute to high-NLL tokens; PonderLM-3 (arXiv 2603.02023) does token-dependent refinement depth — via search snippets: [arXiv 2603.02023](https://arxiv.org/pdf/2603.02023). LoopFormer (arXiv 2602.11451) reports representational collapse with naive early exit in looped models — [arXiv](https://arxiv.org/html/2602.11451v1).
- **ACT** (Graves, arXiv 1603.08983, 2016): learns halting via a time penalty; the penalty weight must be hand-tuned and behavior is sensitive to it — [arXiv](https://arxiv.org/abs/1603.08983), [ar5iv](https://ar5iv.arxiv.org/html/1603.08983).
- **PonderNet** (arXiv 2107.05407, Banino et al. 2021): geometric halting prior, no explicit step penalty; a stable ACT successor — [arXiv](https://arxiv.org/pdf/2107.05407). PALBERT (Balagansky et al., 2022) applies it to stacked transformers — via search summary.

### Inferences
- ACT/PonderNet need the base model trained with the halting head; they do not fit a frozen Dory. Only useful if you retrain Dory.
- For a frozen looped model, a probe on per-loop hidden states (or per-loop logit KL/entropy deltas) is cheap: one small MLP or logistic regression per loop count, trainable on a few thousand labeled prompts on a laptop. This is an inference, not a published result for Dory.
- Ouro's built-in gate is the strongest "learned halting" evidence, but the Ouro exit gate is known to be an in-model mechanism. Its real compute saving requires a custom inference loop (vLLM does not support it).

### Gaps
- Ouro's exit-gate math is not described on the model card; need the full paper (arXiv 2510.25741) for mechanism details, not fetched in full here.
- Whether 2607.20519's proposed "trajectory readouts" beat simple probes is unverified.
- No published cheap-probe benchmark for loop-count prediction specifically.

## Controller inputs: raw prompt (pre-hoc) vs. Dory hidden states (online)

### Takeaway
Pre-hoc text-only controllers (Laya, NVIDIA classifier, Arch-Router) are cheap and run before any loop, but they only see surface difficulty. Online controllers that read Dory's own states after k loops can see actual convergence and are usually more accurate, but they pay for at least k loops before deciding. Hybrid: pre-hoc budget plus a cheap online stop check after 1–2 loops.

### Cited Findings
- Predictive Scheduling (2602.01237) finds middle-layer hidden states carry richer size signals than raw-text classifiers, and a LoRA text classifier is also viable; the MLP-on-hidden-states variant helps most at small budgets — [arXiv 2602.01237](https://arxiv.org/abs/2602.01237).
- Ouro-style halting is inherently online (gate is computed per loop) — [HF card 1.4B](https://huggingface.co/ByteDance/Ouro-1.4B).
- ACT's own paper observes that compute concentrates on hard transitions, showing that halting signals correlate with per-step prediction difficulty — [arXiv 1603.08983](https://arxiv.org/abs/1603.08983v2).

### Inferences
- Pre-hoc (text-only): zero cost on Dory; can run on Laya or the NVIDIA classifier (~0.2–0.4B) concurrently with the 4B model. Weakness: a short prompt can need many loops (multi-hop, code), and text features miss the model's own uncertainty. Expect a ceiling set by how well text predicts loop need.
- Online (hidden states after loop 1 or 2): accurate for "already converged" tokens; cost is the early loops plus the probe. The probe can be a few-KB MLP. Recommended for per-token decisions because they can use the same states Dory already computes.
- Per-token vs. per-input: online per-token halting can save compute only if Dory's inference loop actually skips computation for exited tokens. Per-input budgets are easier to realize in batch serving.

### Gaps
- No published comparison of pre-hoc vs online controllers on a looped LM. The tradeoff above is inference from the cited papers.
- Dory's hidden-state dimensionality and loop structure are not documented in this project, so probe sizing is unknown.

## Getting training labels: min loops correct

### Takeaway
Labels come from running Dory at every loop count K ∈ {1..K_max} on a labeled set and recording the smallest K whose answer is correct (or whose output stops changing). This costs roughly the sum over K of Dory forward passes per prompt. No published recipe for Dory was found; the cost estimates below are my arithmetic, not sourced.

### Cited Findings
- Ouro's default is 4 steps (total_ut_steps) with early exit disabled at threshold 1.0 — [HF card 1.4B](https://huggingface.co/ByteDance/Ouro-1.4B). This sets a practical K_max range (4 to 8) for an Ouro-like model.
- Predictive Scheduling trains predictors on labels of per-query reasoning length and difficulty, then allocates a total budget — [arXiv 2602.01237](https://arxiv.org/abs/2602.01237). Its labels were built the same way (run at multiple budgets), per the abstract summary.

### Inferences
- Label cost: for N prompts and K_max loop counts, the labeling pass costs about N × K_max × (average loop-cost). Example, with my arithmetic and no measured throughput: N = 5,000, K_max = 6, is 30,000 Dory runs. On a 4B model that is hours to a day on an M4; measure actual tok/s before committing.
- Use "first K that matches the full-K answer" as the label rather than ground truth, so you can label unlabeled data. Also record the answer-stability curve, which gives soft labels.
- Use a tolerance: the minimum K where the answer matches the full-loop answer; treat ties by picking the smaller K to reward efficiency. Keep a held-out split because controllers overfit to prompt style.
- Laya-style probabilities need calibration on this label set before the threshold is used (per the Laya card).

### Gaps
- No measured Dory or M4 throughput in this project; cost numbers are estimates only.
- Label noise from exact-match answer checks is not studied for loop controllers.

## Recommended next steps (short)
1. Measure Dory's per-loop cost and check where its hidden states are exposed.
2. Build a labeled set: prompts → first matching K for K = 1..K_max, with stability curves.
3. Baseline 1: text-only pre-hoc classifier (NVIDIA classifier head, or Laya fine-tuned on `choice` over loop counts). Baseline 2: MLP probe on Dory's hidden states after loop 1.
4. Compare accuracy vs. average loops used at matched compute.

## Source list
- Laya: https://huggingface.co/convaiinnovations/laya
- Ouro 1.4B: https://huggingface.co/ByteDance/Ouro-1.4B; Ouro 2.6B: https://huggingface.co/ByteDance/Ouro-2.6B; paper: https://arxiv.org/abs/2510.25741
- NVIDIA classifier: https://huggingface.co/nvidia/prompt-task-and-complexity-classifier
- Arch-Router: https://arxiv.org/abs/2506.16655, https://huggingface.co/katanemo/Arch-Router-1.5B
- RouteLLM: https://github.com/lm-sys/RouteLLM
- Predictive Scheduling: https://arxiv.org/abs/2602.01237
- SelfBudgeter: https://arxiv.org/pdf/2505.11274v1
- ACT: https://arxiv.org/abs/1603.08983; PonderNet: https://arxiv.org/pdf/2107.05407
- Adaptive depth diagnostics: https://arxiv.org/pdf/2607.20519
- LoopFormer: https://arxiv.org/html/2602.11451v1; PonderLM-3: https://arxiv.org/pdf/2603.02023
