# KV Cache Quantization for Local Inference on Apple Silicon (as of Nov 2026)

Scope: what a llama.cpp (llama-server, Metal) user on an M4 MacBook with 24 GB can use today, plus the status of research KV methods, with a check on the "Pearl Research Labs KV-cache quantization method" premise.

**Headline caveat:** I could not find any KV-cache quantization method attributed to Pearl Research Labs. The only Pearl Research Labs material I found is about the Pearl Network blockchain and a Together AI inference partnership. See KQ2.

## KQ1: llama.cpp KV cache types, flash-attention requirements, Metal support, measured numbers

### Takeaway
On Metal, q8_0/q8_0 with flash attention is the safe default. Metal support for quantized K/V was only made real in Aug 2026 (PR #27390, which dequantizes to an F16 scratch buffer before FA). That path has a measured decode regression on some Apple chips, and mixed K/V types (for example q8_0 K with q4_0 V) fail on Metal in the reports I found. TurboQuant-style types (turbo3/turbo4) exist only in forks.

### Cited Findings
- Quantized V cache requires flash attention. The error text is "quantized V cache was requested, but this requires Flash Attention". Reported on Metal with gpt-oss-20b, Qwen3-Coder-30B-A3B and Gemma 4 31B. The issue was closed as stale/not planned on May 19, 2026 with no fix PR linked. — [llama.cpp #21450](https://github.com/ggml-org/llama.cpp/issues/21450)
- Mixed-type K/V fails on Metal in the same report: q8_0/q4_0 fails, while q4_0/q4_0 and f16/f16 work (q8_0/q8_0 worked on an M3 Ultra for Gemma 4 31B at 192 tok/s). The report also shows a q4_0/q4_0 Gemma 4 31B load at about 13 tok/s on M3 Ultra, versus about 141 tok/s for q5_0/q5_0 — a single report, not a controlled benchmark. — [llama.cpp #21450](https://github.com/ggml-org/llama.cpp/issues/21450)
- Metal dequant path: PR #27390, "metal : dequantize quantized KV to F16 before flash attention", merged into master as commit 70aff25 on Aug 20, 2026 (opened Aug 19). A new dequant kernel runs for K and V into a contiguous F16 buffer, then the existing F16 FA kernels run. Covers Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. — [PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390); release notes for tag b10532 list it: [newreleases b10532](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10532)
- Author's speedups on M2 Ultra, `-ctk q8_0 -ctv q8_0 -fa 1`, relative to the pre-PR quantized-KV path (not to F16): qwen3 0.6B Q8_0 pp2048 9665 → 13050 tok/s (1.35x); pp2048@d8192 2238 → 3829 (1.71x); qwen3moe 30B-A3B Q8_0 tg32@d8192 50.07 → 64.73 (1.29x). Perplexity on Qwen2.5-0.5B with q8_0 KV: 1.0008, same as F16. — [PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390)
- Decode regression: on Apple M4 Max (qwen3 0.6B Q8_0, tg128@d8192) the PR went from 216.45 ± 0.61 to 144.32 ± 0.28 tok/s (about −33%); at tg128@d4096, 255.42 → 211.90. ggerganov's follow-up on M5 Max also reported a token-generation drop with the PR. — [PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390)
- Follow-up PR #27438, "metal : dequant kv cache only for large batches", merged Aug 20, 2026, disables the dequant path for small batches to avoid the decode regression. Device-specific tuning was noted as possibly needed later. — [PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390) (the PR page names #27438; I did not open #27438 itself)
- Rotation (Hadamard) before KV quantization: PR #21038, "llama : rotate activations for better quantization", merged into master Apr 1, 2026 (opened Mar 26 by ggerganov). Adds no new types; works with existing q4_0/q5_0/q4_1 etc. Not supported for MLA. V rotation uses 64×64 blocks; 64 was chosen partly because Metal matmul kernels need ne00 ≥ 64. Env var to disable (LLAMA_ATTN_ROT_DISABLE) listed as TODO in the PR description. — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
- Rotation PR accuracy (CPU/CUDA only; no Metal numbers in the thread): wiki PPL, master → PR: Qwen3 8B BF16 q4_0 7.6451 → 7.5012 (F16 baseline 7.3203); Gemma3 4B Q8_0 q4_0 7.8535 → 7.7928 (baseline 7.6905); Qwen3 0.6B q4_0 62.0161 → 46.2503 (baseline 13.6711). KLD on Qwen3.5-9B q4_0/q4_0: mean KLD 0.007979 → 0.004782. — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
- Same PR, AIME25 ×8, gpt-oss-20b, low reasoning, no-rotation → rotation: Q8_0 31.7% → 37.1%; Q5_1 30.8% → 32.5%; Q5_0 25.4% → 32.5%; Q4_1 18.3% → 28.3%; Q4_0 2.0% → 21.7% (one comment lists the baseline as 0.0%). This is the clearest evidence in the sources that un-rotated q4_0 KV can badly damage long-reasoning output. — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
- Same PR, CUDA speed, Qwen3 30B-A3B Q4_0, llama-bench: tg128 229.42 → 202.37 (0.88x); tg128@d32768 101.33 → 94.13 (0.93x); pp512 7603.68 → 7409.35 (0.97x). — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
- TurboQuant-style types (turbo3, turbo4, tq3_0) are not in mainline llama.cpp as far as I could verify; they live in forks (TheTom's turboquant_plus / llama-cpp-turboquant). The discussion ran Mar 25 to Jun 2026 with no maintainer decision to merge. — [discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969); fork: [turboquant_plus](https://github.com/TheTom/turboquant_plus)
- Turbo on Metal, Qwen3.6-35B-A3B Q8_0, M5 Max 128 GB, wikitext-2 c=4096: PPL f16 5.7438, q8_0 5.7433, turbo3 5.8092, turbo4 5.7810. KL vs f16: q8_0 0.0016, turbo4 0.0131, turbo3 0.0199. Top-1 agreement: q8_0 98.64%, turbo4 95.28%, turbo3 93.93%. Decode at 64K depth: f16 59.8, q8_0 57.9, turbo3 49.9, turbo4 55.2 tok/s. The fork's f16 K / turbo4 V prefill fell 34–78× at 8K–128K (attributed to a Metal FA fallback on this fork). q4_0 was not tested in this article. — [DEV Community part 2](https://dev.to/defilan/turboquant-on-a-macbook-pro-part-2-perplexity-kl-divergence-and-asymmetric-kv-on-m5-max-1gb1)
- Turbo on Metal, earlier discussion numbers: M2 Pro short decode 0.73x of q8_0; blended pp8192+tg128 0.90x; after a context-scaling fix, turbo3/q8_0 0.987x at 2K and 0.995x at 32K. — [discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- Turbo FA: "turbo3 silently fails without FA on some hardware" (TheTom, Mar 26); FA needed for turbo3 V because the non-FA path materializes an O(n²) matrix (Madreag, Mar 26). — [discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)

### Inferences
- For a 24 GB M4 on mainline llama.cpp today: q8_0/q8_0 with `-fa 1` is the lowest-risk option (has the Metal dequant path and near-lossless quality). q4_0/q4_0 is also reported to work on Metal, but needs the rotation PR (#21038) to be reasonable at 4-bit. Avoid mixed K/V types on Metal until someone verifies #21450 against a build containing #27390.
- The post-Aug-2026 build matters: a llama.cpp from before PR #27390 likely cannot run quantized V with FA on Metal (per #21450). The exact behavior of mixed q8_0 K / q4_0 V after #27390 is unverified.
- I did not find a Metal benchmark of mainline q4_0 vs q8_0 KV decode speed on M4 at matched build. Treat the M4 Max −33% decode regression (pre-#27438) as the only measured Metal decode cost, and check whether #27438 removes it for decode.

### Gaps
- No post-#27438 Metal decode numbers for q8_0 or q4_0 KV found.
- Did not verify whether mixed q8_0 K / q4_0 V works on Metal after #27390.
- Did not open llama.cpp source for the quantized-KV validation logic; the error text comes from the issue report.

## KQ2: Research KV methods (2025–2026), llama.cpp/MLX/PyTorch-vLLM status, and Pearl Research Labs

### Takeaway
Mainline llama.cpp has merged a rotation method (#21038) and a Metal dequant path (#27390) but no KIVI/KVQuant/TurboQuant-type format. MLX's stock `QuantizedKVCache` is plain affine group quantization. TurboQuant-type formats exist in forks (llama.cpp, MLX-VLM). I found no Pearl Research Labs KV-cache method.

### Cited Findings
- **Pearl Research Labs:** search finds Pearl Research Labs as the creators of the "Pearl Network" blockchain, using "Proof of Useful Work" (inference compute also mines a coin). A partnership with Together AI (announced May 15, 2026) offers a discounted hosted endpoint for Gemma-4-31B-it-pearl. Sources disagree on whether that model is still available on Together's serverless API. — [Together AI blog](https://together.ai/blog/together-ai-partners-with-pearl-research-labs); [Pearl × Together AI](https://lordofpearls.xyz/blog/pearl-together-ai-gemma-4-31b-partnership) (third-party blog, unverified)
- **Pearl KV search:** searches for "Pearl Research Labs KV cache quantization", "Pearl Labs KV cache compression paper 2026" and similar returned no Pearl-authored KV paper, method or repository. Results were unrelated 2026 KV papers only (see below). — no URL, negative result
- **KIVI** (ICML 2024, arXiv 2402.02750): tuning-free asymmetric 2-bit KV, key per-channel, value per-token; claims 2.6× lower peak memory (including weights), up to 4× larger batch, 2.35–3.47× throughput. The abstract gives no baseline configs. — [arXiv 2402.02750](https://arxiv.org/abs/2402.02750)
- **KVQuant** (arXiv 2401.18079): 3-bit KV with <0.1 PPL degradation on Wikitext-2/C4 for LLaMA, Llama-2, Llama-3, Mistral, per a search summary (not fetched; treat as secondhand). Its implementation is PyTorch/CUDA research code, and I did not confirm any llama.cpp or MLX port. — [arXiv 2401.18079](https://arxiv.org/abs/2401.18079)
- **Quantize What Counts** (arXiv 2502.15075, ACL 2026): for a fixed budget, more bits for keys than values reduces error; key-favored allocation (4-bit K, 2-bit V) retains up to 98.3% accuracy versus uniform 4-bit. Models and benchmarks are in the full HTML, which I did not read. — [arXiv 2502.15075](https://arxiv.org/abs/2502.15075)
- **MLX (stock mlx-lm, `mlx_lm/models/cache.py`):** `QuantizedKVCache.__init__(self, group_size: int = 64, bits: int = 8)`; keys and values are quantized with `mx.quantize(..., group_size, bits)` in `update_and_fetch`. The only conversion path is `KVCache.to_quantized(group_size=64, bits=4)`; `make_prompt_cache` never builds a `QuantizedKVCache`. I did not find a `kv_bits` switch in this file, so I could not confirm how `generate` or the server passes `kv_bits`. — [mlx-lm cache.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py)
- **MLX TurboQuant:** mlx-vlm has a TurboQuant KV path (`--kv-bits`; fractional values such as 3.5 switch to TurboQuant; the last layer is skipped). These are secondary guide sources. — [instagit mlx-vlm guide](https://instagit.com/Blaizzy/mlx-vlm/turboquant-kv-cache-quantization-mlx-vlm/); [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) (listed in a search index, not opened)
- **Rotation in llama.cpp:** PR #21038 is a Hadamard rotation of Q/K/V before cache storage, merged Apr 1, 2026; the PR's own TurboQuant reference is the motivation. QuaRot and SpinQuant as papers were not fetched. — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
- **Other 2026 KV papers seen (none attributed to Pearl):** SemantiCache (arXiv 2603.14303), KVSculpt (arXiv 2603.27819), KV Cache Transform Coding (ICLR 2026), Fast KV Compaction via Attention Matching (arXiv 2602.16284). These are compaction/eviction-style methods, not drop-in quantizers. — [arXiv 2603.14303](https://arxiv.org/html/2603.14303v1); [arXiv 2603.27819](https://www.emergentmind.com/papers/2603.27819); [arXiv 2602.16284](https://www.emergentmind.com/papers/2602.16284)
- A Medium-style article claims a "900,000x" KV compression over TurboQuant. It does not name Pearl and gives no reproducible setup; I treat it as unreliable. — [promptzone](https://www.promptzone.com/henrik_nair/kv-cache-compression-hits-900000x-breakthrough-3j39.md)

### Inferences
- Implementation status by method (as far as verified):
  - llama.cpp mainline: Hadamard rotation (#21038, merged); Metal dequant of q4_0/q4_1/q5_0/q5_1/q8_0 (#27390, merged). TurboQuant types: fork only.
  - MLX: affine group quantization (stock mlx-lm); TurboQuant via mlx-vlm and third-party forks.
  - KIVI, KVQuant, QuaRot/SpinQuant, Pearl: no llama.cpp or MLX implementation confirmed; KVQuant/KIVI appear as PyTorch/CUDA research code.
- The premise "Pearl Research Labs' KV-cache quantization method" appears unsupported by public sources I could find. Before evaluating it, the user should provide the paper, repo or link. If the name is a different group, the right entity needs to be identified.

### Gaps
- No Pearl KV method found in arXiv, GitHub or general web searches. Did not search Pearl's own site or Hugging Face org (huggingface.co/pearl-ai appeared in a search result; not opened).
- KIVI/KVQuant implementation status in llama.cpp or MLX not checked against their repos.

## KQ3: Accuracy cost of q8_0 vs q4_0 KV on reasoning benchmarks for 2–8B models; greedy long CoT

### Takeaway
q8_0 KV is near-lossless across the sources found. q4_0 KV is usually small on perplexity for 4–8B models with rotation, but un-rotated q4_0 can collapse on long-reasoning benchmarks. There is no direct llama.cpp measurement of q4_0 KV on GSM8K/MMLU-Pro/coding for 2–8B models that I could find.

### Cited Findings
- q8_0 vs F16 quality, Qwen3.6-35B-A3B on Metal (not 2–8B): PPL delta 5.7433 vs 5.7438; KL 0.0016; top-1 agreement 98.64%. — [DEV Community part 2](https://dev.to/defilan/turboquant-on-a-macbook-pro-part-2-perplexity-kl-divergence-and-asymmetric-kv-on-m5-max-1gb1)
- q8_0 vs F16 on Vulkan (7900 XTX, [discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)): f16 6.8976, q8_0 6.8717, q4_0 6.7860 (both lower than F16 — noise-level differences, not a real improvement); tbq3_0 7.0171.
- q4_0 KV PPL cost in PR #21038 (wiki.test.raw, no rotation → rotation): Qwen3 8B q4_0 7.6451 (+4.4% over F16 7.3203), rotated 7.5012 (+2.5%); Gemma3 4B Q8_0 weights q4_0 7.8535 (+2.1% over 7.6905), rotated 7.7928 (+1.3%); Qwen3.5 4B q4_0 8.3475 vs F16 8.3266 (+0.25%), rotated 8.3448. — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) (percentages are my arithmetic)
- AIME25 ×8, gpt-oss-20b, low reasoning, no rotation: Q8_0 31.7%; Q4_0 2.0%; with rotation Q8_0 37.1%, Q4_0 21.7%. — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
- General research on quantized reasoning models: 4-bit weight-only or KV quantization ≤1% drop on the tested models; harder tasks degrade more (AIME-120 up to 4× greater than GSM8K), and aggressive low-bit quantization can lengthen outputs, especially in small models. Error may accumulate along long CoT. — [arXiv 2504.04823](https://arxiv.org/html/2504.04823v2) (search summary, not fetched)
- One secondhand report on DeepSeek V4 Flash: q8_0 KV left average PPL almost unchanged but changed which tokens were picked at 512 context; the author advised against drawing conclusions for long tasks. — [dev.to breachprotocol](https://dev.to/breachprotocol/quantizing-v4-flashs-kv-cache-in-llamacpp-changes-which-tokens-it-picks-38np) (single-author, unverified)

### Inferences
- For greedy long CoT, the risk is not visible in perplexity: a small PPL delta can still change argmax tokens that compound over thousands of tokens. The gpt-oss-20b AIME25 numbers (un-rotated q4_0 at 2.0%) show the worst case in the sources; with rotation the gap narrows to 37.1 vs 21.7 — still large, which suggests q4_0 KV should not be assumed safe for reasoning tasks without a task eval.
- For a 2–8B model on an M4, q8_0 KV is the conservative choice; q4_0 needs a task-specific eval (e.g., your own greedy-decoded reasoning set) before use.

### Gaps
- No llama.cpp measurement of q4_0 KV on GSM8K, MMLU-Pro or coding for 2–8B models found.
- No greedy long-CoT eval comparing q8_0 vs q4_0 KV with matched seeds found.
- The 2504.04823 and DeepSeek V4 numbers were not checked against their full papers.

## KQ4: Memory and speed effects on unified-memory Macs; Metal dequant overhead

### Takeaway
Quantized KV saves memory and reduces bytes read per attended token, but on Metal the current quantized-KV path dequantizes to F16 and can reduce decode throughput. The dequant cost is the main speed risk. Speed gains reported so far are prefill, and only with the newest build.

### Cited Findings
- Dequant design: PR #27390 dequantizes K and V into an F16 scratch buffer per call before the F16 FA kernel. — [PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390)
- Decode regression, M4 Max: 216.45 → 144.32 tok/s at tg128@d8192 (qwen3 0.6B Q8_0). Follow-up #27438 disables dequant for small batches. — [PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390)
- Turbo fork on M5 Max (Qwen3.6-35B-A3B Q8_0): decode at depth 0/8K/32K/128K: q8_0 K/turbo4 V 82.9/75.4/66.0/41.0 tok/s; f16 K/turbo4 V 72.8/16.9/8.6/2.8. Prefill pp512 at 64K: f16 602.0, q8_0 479.2, turbo3 469.8. — [DEV Community part 2](https://dev.to/defilan/turboquant-on-a-macbook-pro-part-2-perplexity-kl-divergence-and-asymmetric-kv-on-m5-max-1gb1)
- Quantized K with F16 V was the slow combination in that fork; the author attributes it to a Metal FA kernel fallback. — [DEV Community part 2](https://dev.to/defilan/turboquant-on-a-macbook-pro-part-2-perplexity-kl-divergence-and-asymmetric-kv-on-m5-max-1gb1)
- Gemma 4 31B Q4_K_M on M3 Ultra, 32K: q8_0/q8_0 ~192 tok/s; q5_0/q5_0 ~141 tok/s; q4_0/q4_0 ~13 tok/s (loads). — [llama.cpp #21450](https://github.com/ggml-org/llama.cpp/issues/21450) (single reporter; no method given)
- Rotation PR CUDA decode 0.88x (tg128) with rotation on. — [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
- Community report, Dgx Spark, corrected Apr 1 (turbo discussion): q4_0 generation at 110K context fell from 38.0 to 24.0 tok/s (−36.8%); an earlier 92.5% prompt-collapse claim was retracted. — [discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)

### Inferences
- Memory: q8_0 stores 8.5 bits per value and q4_0 4.5 bits per value (ggml block layouts: 34 and 18 bytes per 32 values). This is my computation from the block sizes, not from a fetched source; compared with F16 (16 bits), q8_0 halves the KV bytes and q4_0 cuts them to roughly 28%. At 24 GB, KV savings matter mainly for long contexts; for 2–8B models at moderate context the KV is small compared with weights.
- Decode: decode is usually bandwidth-bound, so smaller KV can help at long depth in principle. The Metal numbers show the dequant overhead can outweigh that on the current path, so quantized KV may be slower than F16 for decode at moderate depth. Test at your target depth.
- Prefill gains from the newer Metal path (2.3× to 1.7× in the author's runs) are the clearest speed win so far.

### Gaps
- No M4 (base or Pro) benchmark of mainline quantized vs F16 KV at matched build found.
- No measurement after PR #27438 for decode on M4.
- Memory figures are my own calculation from block layouts; not verified against llama.cpp source.

## Practical recommendation for the M4 / 24 GB case (inference from the above)
1. Use a llama.cpp build at or after commit 70aff25 (PR #27390) plus #27438 for quantized KV with `-fa 1`. Older builds may refuse quantized V on Metal.
2. Start with `--cache-type-k q8_0 --cache-type-v q8_0` and compare decode tok/s against f16 at your real context depth before keeping it.
3. Avoid mixed K/V types on Metal until verified (#21450). Avoid un-rotated q4_0 KV for long-reasoning tasks; if you use q4_0, use a build that includes rotation (#21038) and run a greedy eval on your own tasks.
4. Turbo/TurboQuant types need a fork and are not recommended for a production setup based on these sources.
5. Do not adopt "Pearl Research Labs KV quantization" as a candidate: no method, paper, or code was found. Ask the user for the source.

## Sources (consolidated)
- llama.cpp PR #27390: https://github.com/ggml-org/llama.cpp/pull/27390
- llama.cpp PR #21038: https://github.com/ggml-org/llama.cpp/pull/21038
- llama.cpp issue #21450: https://github.com/ggml-org/llama.cpp/issues/21450
- llama.cpp discussion #20969: https://github.com/ggml-org/llama.cpp/discussions/20969
- llama.cpp release b10532: https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10532
- DEV Community, TurboQuant on MacBook Pro part 2: https://dev.to/defilan/turboquant-on-a-macbook-pro-part-2-perplexity-kl-divergence-and-asymmetric-kv-on-m5-max-1gb1
- mlx-lm cache.py: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py
- KIVI arXiv 2402.02750: https://arxiv.org/abs/2402.02750
- Quantize What Counts arXiv 2502.15075: https://arxiv.org/abs/2502.15075
- Quantization Hurts Reasoning? arXiv 2504.04823: https://arxiv.org/html/2504.04823v2
- Together AI / Pearl Research Labs: https://together.ai/blog/together-ai-partners-with-pearl-research-labs
