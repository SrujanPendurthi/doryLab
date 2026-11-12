# NVIDIA Dory 1.0 4B (looped LM): loop-count control

Research date: as of 2026-11-11. Sources are primary where marked (arXiv paper, HF model card); the rest are secondary and flagged.

## What Dory is

### Takeaway
Dory1.0-4B is NVIDIA's 4B-parameter latent-recurrent ("looped") normalized-GPT, released as open weights. One checkpoint can run with 1, 2, or 4 recurrent loops at inference, and the loop count is a runtime setting (not a separate checkpoint). Loop count is chosen per prompt or per worker during training; no learned exit exists.

### Cited Findings
- Name, authors, arXiv 2611.03998, "Dory1.0-4B: A Looped Normalized-GPT", submitted 2 Nov 2026, CC BY 4.0 on the paper; authors Maxim Fishman, Simeng Sun, Shantanu Acharya, Faisal Ladhak, Brian Chmiel, Santiago Akle Serano, Ron Banner, Boris Ginsburg, Krishna C. Puvvada. [arXiv abs](https://arxiv.org/abs/2611.03998); [arXiv HTML](https://arxiv.org/html/2611.03998v1)
- Model weights: HF collection `nvidia/nemotron-labs-dory` with four repos: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, `...-NVFP4`, `...-MXFP8`, `...-GGUF` (Q5_K_M). [HF collection](https://huggingface.co/collections/nvidia/nemotron-labs-dory)
- Model card license: OpenMDW License Agreement v1.1, "ready for commercial or non-commercial use." [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Pretraining: ~6T tokens, first ~4.5T in NVFP4, then BF16; context extended to 256K (about 300B tokens of extension). SFT then RLVR (23 environments, 200 steps). [arXiv](https://arxiv.org/html/2611.03998v1)
- Release date conflict: a secondary source says open weights released 16 Oct 2026; arXiv v1 is dated 2 Nov 2026; a news article dated 4 Nov 2026 covers the paper. Treat the exact release date as unverified. [aiweekly](https://aiweekly.co/alerts/nvidias-dory10-4b-uses-looped-layers-to-match-9b-rivals)
- No NVIDIA launch blog was found (per one secondary source). No GitHub repo for Dory itself is listed in the paper; it cites NeMo RL, NeMo Evaluator, NeMo Gym, Model-Optimizer, Harbor. [arXiv](https://arxiv.org/html/2611.03998v1)

### Inferences
- The HF repo is the primary artifact to pull code/config from; the paper's architecture details and the card's runtime config (below) are the authoritative sources for controlling loops.

### Gaps
- Official GitHub repo for model code: none found.
- Paper's full text was read via a summarizing fetch, not raw PDF; numeric tables were extracted from that summary. Verify before quoting.

## Architecture and looping

### Takeaway
Input block -> shared recurrent block (re-applied k times) -> output block. Normalized GPT (nGPT) with interleaved sliding-window and full-attention layers. The input representation is re-injected into the recurrent block each loop (per secondary source).

### Cited Findings
- Blocks: input block, shared recurrent block, output block. [arXiv](https://arxiv.org/html/2611.03998v1); [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Layer counts CONFLICT:
  - arXiv (paper summary): 16-layer input, 40-layer recurrent, 16-layer output (72 physical layers). [arXiv](https://arxiv.org/html/2611.03998v1)
  - BF16 model card (summary): 8-layer input, 20-layer recurrent, 8-layer output. [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16); also [aiweekly-style secondary coverage](https://aiweekly.co/alerts/nvidias-dory10-4b-uses-looped-layers-to-match-9b-rivals) and [mindstudio](https://www.mindstudio.ai/blog/nvidia-dory-1-0-4b-looped-model) say 8/20/8.
  - Likely reconciliation (my inference, unverified): the paper may count attention and MLP sublayers separately (2 x 8 = 16, 2 x 20 = 40). Check the raw paper/config.
- Attention: interleaved sliding-window and full-attention layers. [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Total parameters: "4-billion" per the abstract (one secondary source says 3.97B). [arXiv abs](https://arxiv.org/abs/2611.03998)

### Loop-count distribution (training)
- Pretraining (continuation): loop count drawn uniformly from {1,2,3,4} (p=0.25 each), deterministic per optimizer step and microbatch, shared across data-parallel replicas; a rank-level override forces ~9.9% of replicas to 1 loop. Long-context extension uses same distribution with ~8.3% 1-loop override. [arXiv](https://arxiv.org/html/2611.03998v1)
- SFT: uniform over {1,2,3,4}, with 0.1 one-loop override fraction. [arXiv](https://arxiv.org/html/2611.03998v1)
- RLVR: each rollout worker runs a fixed k in {1,2,3,4}; prompts assigned loops in a 4:1:1:1 quota; all 16 rollouts in a GRPO group share one k. [arXiv](https://arxiv.org/html/2611.03998v1)
- Difficulty-aware curriculum: routing score s = a*q + (1-a)*u with a = 0.3 (q = profiled pass-rate quantile, u uniform); harder prompts fill deeper-loop slots first. [arXiv](https://arxiv.org/html/2611.03998v1)
- Trained range: 1-4 loops. Base-model evals also test 6 and 8 (extrapolation). [arXiv](https://arxiv.org/html/2611.03998v1)
- Not Poisson; not a learned per-token depth. Loop count is fixed per prompt (curriculum) or per worker (RL).

### Adaptive halting / early exit
- None. The paper describes no learned exit gate, no KL/entropy exit, no per-token adaptive depth. Loop count is one value per request at serve time. [arXiv](https://arxiv.org/html/2611.03998v1)
- The "elastic depth" phrasing means choosing k at inference, not adaptive halting.

### Monotonic in loops?
- Mostly yes on coding/agentic/long-context; not uniformly.
- Base model (fixed loops, NVIDIA table): MATH-500 60.60 (1) -> 65.00 (2) -> 69.40 (4) -> 68.40 (6) -> 69.40 (8). MMLU-Pro 51.52 -> 55.06 -> 56.09 -> 55.72 -> 55.86. Some benchmarks flatten or dip at 6-8 (e.g., HumanEval 62.34 at 4, 60.52 at 6). [arXiv](https://arxiv.org/html/2611.03998v1)
- Post-trained: flattening after 2 loops on GPQA Diamond (72.41 at 2, 72.16 at 4) and AA-Omniscience. [arXiv](https://arxiv.org/html/2611.03998v1); [model card summary](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)

## Inference API: how loop count is set

### Takeaway
Loop count is a runtime config override, not a generate() argument. The HF transformers path is not documented on the card for loop control. vLLM uses `--hf-overrides '{"n_recurrent_loops": ...}'`; llama.cpp uses `--override-kv dory.num_loops=int:...`. The paper does not give an argument name. Per-token variation is not supported in the documented interface (loop count is per model/config at serve time, and the training design is per prompt/worker).

### Cited Findings
- vLLM: `--hf-overrides '{"n_recurrent_loops": 2, "recurrent_kv_cache_mode": "per_loop"}'`. Support is under review in vllm-project/vllm#60149; install from that PR until released. [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- llama.cpp: `--override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop"`. Support in ggml-org/llama.cpp#30306; requires GGUF (e.g., Q5_K_M). [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Transformers: card shows only `AutoModelForCausalLM.from_pretrained("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", device_map="auto")`. No loop-count argument shown; trust_remote_code not mentioned. [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Card lists supported runtimes: Transformers, vLLM, SGLang, llama.cpp, Docker Model Runner; hardware NVIDIA Hopper and Blackwell, Linux. [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Default recommended: 2 loops; 1 for minimum compute; 4 for more accuracy. [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Per-token variation: nothing found. The paper states "the recurrent block can be applied a variable number of times" without a per-token mechanism. [arXiv abs](https://arxiv.org/abs/2611.03998)

### KV cache across loops
- Two modes:
  - `per_loop` (default): each loop keeps its own KV; recurrent KV memory grows linearly with k.
  - `last_loop`: each loop overwrites the previous loop's entries; attention KV stays at 1-loop size. Queries attend to final-loop entries for earlier tokens and current-loop entries for same-step tokens.
- Paper: KV footprint equals a non-looped model at any k under last-loop caching. [arXiv](https://arxiv.org/html/2611.03998v1); [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- Table 6 (k=4, per-loop vs last-loop): Terminal-Bench 22.19 vs 20.22; AA-LCR 72.94 vs 70.81; GPQA 72.16 vs 72.10; SciCode 35.11 vs 35.37. [arXiv](https://arxiv.org/html/2611.03998v1)
- Self-speculative decoding exists in the paper: drafts with k=1, verifies with k=4, gamma=4, shares KV/activations. [arXiv](https://arxiv.org/html/2611.03998v1)
- Serving uses prefix caching and chunked prefill (paper). [arXiv](https://arxiv.org/html/2611.03998v1)

## Published results

### Takeaway
NVIDIA reports accuracy rising from 1 to 2 loops, with smaller gains at 4; agentic/coding tasks gain most. All numbers are NVIDIA-reported; none independently verified.

### Cited Findings (post-trained, BF16, loops 1 / 2 / 4)
- SWE-bench Verified: 50.3 / 59.5 / 61.8 (Qwen3.5-4B 54.7; Qwen3.5-9B 61.0; Gemma 4 26B-A4B 57.40; Qwen3.6-35B-A3B 73.4 self-reported). [arXiv](https://arxiv.org/html/2611.03998v1)
- Terminal-Bench v2.1: 12.5 / 18.1 / 22.2. [arXiv](https://arxiv.org/html/2611.03998v1)
- SciCode: 27.3 / 34.94 / 35.11. [arXiv](https://arxiv.org/html/2611.03998v1)
- GPQA Diamond: 69.19 / 72.41 / 72.16 (Qwen3.5-9B 80.93). [arXiv](https://arxiv.org/html/2611.03998v1)
- GDPval-AA v2 Elo: 815 / 946 / 967. [arXiv](https://arxiv.org/html/2611.03998v1)
- Average of 10-ish listed benchmarks (equal weights): 35.17 / 39.67 / 40.60 (Qwen3.5-9B 32.21; Gemma 4 26B-A4B 36.73; Qwen3.6-35B-A3B 42.30). [arXiv](https://arxiv.org/html/2611.03998v1)
- Abstract claim: with 2+ loops, Dory outperforms Qwen3.5-9B, Gemma 4 12B, and Gemma 4 26B on average across 12 benchmarks (no per-benchmark numbers in the summary). [arXiv abs](https://arxiv.org/abs/2611.03998)
- Efficiency: at 4 loops, 1.3-2.5x fewer output tokens than at 1 loop on agentic tasks. Table 6 shows output-token ratio ~1.00-1.03x for k=4 with last-loop cache vs k=1 on the listed tasks (different metric; reconcile before citing). [arXiv](https://arxiv.org/html/2611.03998v1)

### Quantized variants (paper Table 4), loops 1/2/4
- BF16 vs NVFP4-QAD vs Q5_K_M-QAD close at k=4 (e.g., GPQA 72.16 BF16 vs 72.35 NVFP4-QAD vs 72.66 Q5_K_M-QAD). MXFP8 PTQ tracks BF16. [arXiv](https://arxiv.org/html/2611.03998v1)

### Latency / compute per loop
- Compute per token scales roughly with k for the recurrent block (my inference: recurrent block is 40 of 72 physical layers per paper, or 20 of 36 per card; still, per-token FLOPs grow with k).
- Inference speed tests: NVIDIA DGX Spark (paper). No tokens/s numbers found in the summaries. [arXiv](https://arxiv.org/html/2611.03998v1)
- NVIDIA advises end-to-end measurement because fewer output tokens can offset per-token cost (secondary). [Secondary: hyper-level coverage via mindstudio](https://www.mindstudio.ai/blog/nvidia-dory-1-0-4b-looped-model)

### Gaps
- No per-loop tokens/s or latency table found.
- No independent reproduction found. One review says benchmark claims are not independently verified.

## Supported runtimes and quantization

### Takeaway
Weights: BF16, NVFP4, MXFP8, GGUF Q5_K_M. Runtimes: transformers (loop control undocumented on card), vLLM and llama.cpp need PRs/branches for loop control as of the sources read. MLX: nothing found.

### Cited Findings
- HF repos listed above. [HF collection](https://huggingface.co/collections/nvidia/nemotron-labs-dory)
- vLLM support pending PR #60149; llama.cpp support in PR #30306. [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- One secondary source says standard inference stacks need a branch build; another says vLLM and llama.cpp builds from open PRs run today. [aiweekly-alt: explainx](https://explainx.ai/blog/nvidia-nemotron-labs-dory-4b-looped-model-nvfp4-adjustable-loops-swe-bench-2026); [orcarouter](https://www.orcarouter.ai/blog/nemotron-labs-dory-1-0-4b-quiet-release)
- MLX: not mentioned in the sources read.

### Gaps
- Whether transformers needs trust_remote_code or a custom class to set loops: not confirmed. The card's snippet has no loop setting; a `config` override (e.g., `n_recurrent_loops`) is plausible by analogy with vLLM but unverified.
- Whether llama.cpp PR #30306 is merged as of 2026-11-11: unverified.

## Close variants and related work

### Takeaway
Dory is the NVIDIA 4B looped model. Related looped/recurrent-depth work: Huginn (Geiping et al.), Ouro (LoopLM), Nanbeige 4.2 3B looped, and Huginn-style retrofits.

### Cited Findings
- Ouro LoopLM (Oct 2025) — basis Dory cites; Ouro 1.4B-R4 competitive with 4B models. [secondary](https://www.datastudios.org/post/nvidia-dory-4b-looped-reasoning-256k-context-elastic-inference-benchmarks) (unverified against the Ouro paper)
- Huginn recurrent-depth: [arXiv 2502.05171](https://arxiv.org/abs/2502.05171); Huginn-0125 model at [HF](https://huggingface.co/tomg-group-umd/huginn-0125). Secondary: 3.5B, 800B tokens. [huggingface](https://huggingface.co/papers/2502.05171)
- Retrofitted recurrence: [arXiv 2511.07384](https://arxiv.org/html/2511.07384v1).
- Nanbeige 4.2 3B looped transformer: [runtimewire](https://runtimewire.com/article/nanbeige-releases-4-2-3b-looped-transformer) (secondary, unverified).

### Gaps
- Did not check Ouro paper directly.

## Source quality notes
- Primary: arXiv 2611.03998 (fetched via summarizer, not raw PDF), BF16 model card (fetched via summarizer).
- Secondary and flagged: aiweekly, mindstudio, datastudios, explainx, orcarouter, tilnote, madebyagents — dates, layer counts, and runtime status conflict with primaries.
- The first guessed HF URL without `-BF16` returned 401; the correct repo is `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`.

## Bottom line for an external loop controller
- Pick k in {1,2,3,4} per request by setting `n_recurrent_loops` (vLLM, via `--hf-overrides`) or `dory.num_loops` (llama.cpp, via `--override-kv`). Per-request switching in a single server process is not documented; you'd likely run separate servers or per-request override support (unverified).
- KV mode: `per_loop` (default) or `last_loop` (memory-saving).
- No built-in halting; the controller must supply k. Training used per-prompt difficulty routing, so a difficulty/complexity classifier mapping to k mirrors the training distribution.
- Per-token loop variation: not supported by design as documented.
