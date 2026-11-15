# Pearl Research Labs: KV Cache Quantization (PearlQuant)

Scope: what Pearl Research Labs has released on KV cache quantization, as of 2026-11-13. Sources are primary (the lab's own pages) unless labeled otherwise. The lab's KV-cache work is a single blog post with no paper, no code, and no independent evaluation found.

## Who is Pearl Research Labs

### Takeaway
Pearl Research Labs (pearlresearch.ai) is an Israeli company that runs the Pearl (PRL) proof-of-useful-work blockchain and also sells inference infrastructure. The PearlQuant post is one of several blog posts on its research page. Several other "Pearl" companies exist (Pearl AI / Pearl Labs Creator, Pearl.com, hellopearl), and none of them is this lab.

### Cited Findings
- The lab's site describes Pearl Research as building "inference infrastructure and algorithms," with an OpenAI-compatible API at inference.pearlresearch.ai/v1 and Managed Inference and On Premise options — [pearlresearch.ai](https://pearlresearch.ai)
- Research page lists: PearlQuant (KV cache), "Optimizing Grouped GEMM for MoE on Blackwell," "Hawkeye: Reproducing GPU-Level Non-Determinism" (arXiv 2603.20421), "A Fused Blockwise Hadamard Quantization Kernel in CuTe DSL" (X post), "400x Faster Hashing on GPUs" (X post), the Pearl Protocol whitepaper, and the Pearl INT whitepaper — [pearlresearch.ai/research](https://pearlresearch.ai/research)
- Research posts show no publication dates. The only date-like strings are a "© 2026" footer and arXiv IDs — [pearlresearch.ai/research](https://pearlresearch.ai/research)
- Protocol papers: "Proofs of Useful Work From Arbitrary Matrix Multiplication" — [arXiv 2504.09971](https://arxiv.org/abs/2504.09971); "The Economics of Proof-of-Useful-Work" — [arXiv 2606.06700](https://arxiv.org/abs/2606.06700)
- Co-founder and CEO is Omri Weinstein, a theoretical computer scientist (Hebrew University). Sources differ on his PhD details, so they are not used here — [iq.wiki](https://iq.wiki/wiki/omri-weinstein.md); [ourcryptotalk.com](https://ourcryptotalk.com/founders-corner/who-is-omri-weinstein-founder-of-pearl)
- Mainnet launch reported April 27, 2026, and the Together AI partnership on May 15, 2026 — [Together AI blog](https://www.together.ai/blog/together-ai-partners-with-pearl-research-labs) (secondary coverage: [TraeAI glossary](https://www.traeai.com/en/glossary/pearl-research-labs))
- Hugging Face org `pearl-ai` (Pearl Research Labs): four models (Gemma-4-31B-it-pearl, Qwen3-30B-A3B-Instruct-2507-pearl, Llama-3.3-70B-Instruct-pearl, Llama-3.1-8B-Instruct-pearl). No papers, no datasets, and nothing on KV cache or quantization — [huggingface.co/pearl-ai](https://huggingface.co/pearl-ai)

### Inferences
- The lab's inference-optimization output is blog posts and kernel write-ups. No peer-reviewed or arXiv paper on KV cache quantization was found.

### Gaps
- No author list or publication date for the PearlQuant post.

## How the method works (PearlQuant)

### Takeaway
PearlQuant is a lattice vector quantizer inspired by NestQuant. It quantizes groups of 8 values on the E8 lattice after a random rotation. The lab reports 2.1–3.2 bits per value. The post omits most of the design details: per-channel vs per-token, key/value asymmetry, outlier handling, calibration, and any full-precision window are not described.

### Cited Findings
- Primary source is the PearlQuant post: "KV cache quantization for production-scale LLM serving" — [pearlresearch.ai/research/blog/pearlquant](https://pearlresearch.ai/research/blog/pearlquant)
- Core approach: values are quantized in groups of eight on the E8 lattice (240 facets per cell), not one value at a time. The post says it "rounds groups of values together rather than each value separately, to reduce quantization error" — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- A random rotation is applied before quantization so values behave like a Gaussian cloud — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Inspiration cited: NestQuant (arXiv 2502.09720), which uses E8 nested-lattice vector quantization for weights, activations, and KV cache — [NestQuant arXiv](https://arxiv.org/abs/2502.09720v1)
- Mixed precision: the FP8 baseline stores NoPE in FP8 and RoPE in BF16 (9.1 bits per value average). PearlQuant configs average 3.2, 2.3, or 2.1 bits per value — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Theoretical wasted-bits comparison (Figure 5): PearlQuant at 2.75/3.75 bits: 0.39/0.42; TurboQuant MSE at 3/4 bits: 0.56/0.60; NVFP4 at 4.5 bits: 1.15; FP8 at 8 bits: 2.65 — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Not specified in the post: per-channel vs per-token, keys vs values treatment, Hadamard use inside PearlQuant, outlier handling, calibration, residual/full-precision window — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- The separate "Fused Blockwise Hadamard Quantization Kernel in CuTe DSL" post concerns activation quantization on the inference hot path, not the KV cache — [pearlresearch.ai/research](https://pearlresearch.ai/research)

### Inferences
- The method is a vector-quantization scheme (E8 lattice plus random rotation), closer to NestQuant and TurboQuant-style theory than to KIVI/KVQuant-style per-channel scalar schemes. This is an inference from the lattice description, not a stated classification.

### Gaps
- Exact bit allocation for keys vs values, per-channel vs per-token layout, and whether a residual window exists are unknown.
- Algorithm details beyond the blog summary, such as the lattice decoder and scale handling, are not described.

## Claims and benchmarks

### Takeaway
All reported numbers are from the lab on one model (GLM-5.3) and one 8×B200 node. Quality is measured as parity with FP8. Speed claims are at high concurrency under a Claude Code trace workload. No independent reproduction was found.

### Cited Findings
- Model: GLM-5.3 (78 layers, sparse attention); example context of 100k tokens per session; KV cache 5.1 GB at FP8 vs 1.8 GB at PearlQuant-3.2 — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Headline claim: "uses 65% less KV cache storage per value than the FP8 baseline, with accuracy and text quality on par with FP8" — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant); the same figure appears on the research index — [pearlresearch.ai/research](https://pearlresearch.ai/research)
- Quality table (FP8 / PQ-3.2 / PQ-2.3 / PQ-2.1):
  - KL divergence on PG-19: ~0.04 (noise floor) / 0.057 / 0.122 / 0.207
  - RULER: 97.2 / 97.1 / 96.5 / 94.8
  - LongBench v2: 69.2 / 68.2 / 69.0 / 60.4
  - MMLU-Pro: 86.3 / 86.1 / 85.5 / 85.7
  - LiveCodeBench: 95.7 / 94.9 / 93.6 / 95.2
  - Source: [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- "Wait" token rate change vs FP8: PQ-3.2 −1% to +3%; PQ-2.3 +10% to +28%; PQ-2.1 +48% to +106% — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- The post states "PearlQuant-3.2 shows no statistically significant difference from FP8 on any of the four benchmarks" — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Self-judge on 800 MMLU-Pro questions: judge blamed the quantized answer 53.2% (PQ-3.2), 52.7% (PQ-2.3), 69.7% (PQ-2.1); a second FP8 run scored 52.5% — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Serving: one NVIDIA HGX 8×B200 node; vLLM with LMCache, DP=8, expert parallelism, speculative decoding; kernel in CuTe DSL for Blackwell — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Workload: replayed Claude Code session traces from the AgentX dataset (SemiAnalysis), AIPerf 13.0 — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- p90 TTFT at 128 sessions: FP8 144 s, NVFP4 81 s, PearlQuant 7.3 s. Peak throughput +24% vs FP8; at light load FP8 is about 8% ahead — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- Benchmarks used: PG-19 (arXiv 1911.05507), RULER (arXiv 2404.06654), LongBench v2 (arXiv 2412.15204), MMLU-Pro (arXiv 2406.01574), LiveCodeBench (arXiv 2403.07974) — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)

### Inferences
- The quality parity claim holds for PQ-3.2 on this set. PQ-2.1 shows measurable degradation on KL, LongBench v2, and the "wait" metric, which the post does not frame as a failure. The post's own numbers do not support "no quality loss" at 2.1 bits.
- The TTFT gap (144 s vs 7.3 s) is measured under queueing at high concurrency, so it mostly reflects memory-capacity-driven batching, not per-token decode speed. The post gives no single-stream decode-speed number.

### Gaps
- No decode tokens/s at batch 1, no prefill numbers, and no context-length sweep beyond the 100k example.
- Baselines: KIVI, KVQuant, QuaRot, llama.cpp q8_0/q4_0 KV, and FP8 KV-cache comparisons in other frameworks are not in the post. TurboQuant appears only as a theoretical comparison, since its kernels don't support GLM-5.3's DSA attention (source: [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)). Apples-to-apples comparison against KIVI, KVQuant, or QuaRot is absent.
- No results on dense models (Llama, Qwen) and no Apple silicon or Metal results.

## Code, framework, license

### Takeaway
No public PearlQuant code, repository, or license was found. The lab's GitHub org has no KV-cache or quantization repo. Framework support is vLLM/LMCache on NVIDIA Blackwell only, per the post.

### Cited Findings
- The PearlQuant post names no repo or license. Links are to the GitHub org, Hugging Face, and NVIDIA CUTLASS (CuTe DSL) — [PearlQuant post](https://pearlresearch.ai/research/blog/pearlquant)
- GitHub org `pearl-research-labs` lists four public repos: `pearl` (Go, blockchain monorepo), `antseed-github-verification`, `pips` (Pearl Improvement Proposals, CC0-1.0), and `blockbook` (Trezor fork, AGPL-3.0). None is a KV-cache or quantization repo — [github.com/pearl-research-labs](https://github.com/pearl-research-labs)
- Search for a PearlQuant or Pearl-branded KV-cache implementation in llama.cpp, MLX, or Metal returned nothing relevant — [WebSearch: Pearl KV llama.cpp Metal](https://www.google.com/search?q=Pearl+KV+cache+quantization+llama.cpp+Apple+Metal) (search results only; no Pearl hits)
- Kernel write-ups on the lab's research page (grouped GEMM, fused Hadamard) are described as blog posts, not released libraries — [pearlresearch.ai/research](https://pearlresearch.ai/research)

### Inferences
- PearlQuant is not open source as of this check. Apple silicon and Metal support is unknown and, given the Blackwell CuTe DSL kernel, unlikely.

### Gaps
- License: unknown, since no code is released.
- Whether the kernel will be released is not stated.

## Independent reproduction, critique, adoption

### Takeaway
No independent reproduction, critique, or adoption of PearlQuant was found in searches of X, Reddit (r/LocalLLaMA), Hacker News, arXiv, and general web. The only third-party coverage of the lab is crypto-focused.

### Cited Findings
- Searches for "PearlQuant" alone and with KV cache, GLM-5.3, LocalLLaMA, and reddit returned no hits for PearlQuant. The closest name match is Google's PolarQuant (arXiv 2502.00527), a different method — [alphaxiv PolarQuant](https://alphaxiv.org/abs/2502.00527)
- Third-party coverage of the lab centers on the PRL token and proof-of-useful-work economics. Blockworks argues miners choose the multiplications, so the protocol proves the math was done, not that it served AI requests. Hashrate Index makes a similar point — [Blockworks](https://blockworks.com/newsletter/0xresearch/issue/post_daf0d74a-2bf4-41e4-86d3-2a1275475a24); [Hashrate Index](https://hashrateindex.com/blog/pearl-prl-ai-compute-cryptocurrency/)
- Together AI's blog says the Gemma-4-31B-it-pearl endpoint is priced more than 25% below normal because of PRL emissions tied to the workload — [Together AI blog](https://www.together.ai/blog/together-ai-partners-with-pearl-research-labs)
- Sources differ on the partnership's framing. One calls it Pearl's first major enterprise partnership; another calls it "niche and experimental" — [PulseAugur](https://pulseaugur.com/zh-cn/cluster/46560-pearl-labs-partners-with-together-ai-for-inference-optimization); [SparkPulse](https://sparkpulse-ui-next.pages.dev/product/together-ai)

### Inferences
- The lab's KV-cache claims have had no visible outside scrutiny as of 2026-11-13. Treat all numbers as the lab's self-reported results.

### Gaps
- No independent benchmark, no reproduction, and no adoption by any inference engine was found. The Together AI deal is about a pricing model for a hosted endpoint, not about KV-cache quantization.

## Overall caveats for the report writer

- The lab's only KV-cache work is one blog post, with no date, author, paper, or code. All numbers are self-reported on one model (GLM-5.3), one 8×B200 node, and a Claude Code-style agentic trace.
- Baselines are FP8 (main), NVFP4, and a theoretical TurboQuant. KIVI, KVQuant, QuaRot, and llama.cpp q8_0/q4_0 were not compared.
- Speed claims are concurrency and TTFT under queueing, not single-stream decode speed.
- Do not confuse this lab with other "Pearl" companies, or with Google's PolarQuant.
- Sources are primary for the method and numbers, and crypto-news secondary for the company background. The Hugging Face and GitHub pages were read through a fetch summarizer, not raw page views.
