# Running a looped 4B LLM + controller on Apple M4 (base, 24 GB)

Research date: 2026-11-11. Machine: Apple M4 base (10-core CPU, 10-core GPU, ~120 GB/s unified memory bandwidth), 24 GB, macOS 27.
Confidence key: [Measured] = number published by a source from a real run; [Claimed] = vendor/blog figure, not verified; [Est] = my first-principles estimate, labeled as such.

## Runtime support for looped / recurrent-depth models

### Takeaway
No mainstream runtime supports Dory, Huginn, or Ouro out of the box on Apple Silicon. MLX is the only realistic path for per-request loop control, and it needs a custom model file (one precedent exists for Ouro). llama.cpp has a Dory PR fork but no evidence of per-request loop control. Ollama has no support for any of them. Ouro's own card says "No llama.cpp, no Ollama, no GGUF."

### Cited Findings
- Ouro has an MLX bf16 conversion by ArcadaLabs (mlx-lm 0.29.1), but stock mlx-lm does not ship an `ouro` architecture. The user must copy a custom `ouro.py` into `mlx_lm/models/`. The custom file adds sandwich RMSNorm, recurrent looping over blocks, a KV cache sized `num_hidden_layers * total_ut_steps`, and an early-exit gate. — [ArcadaLabs/Ouro-2.6B-mlx-bf16](https://huggingface.co/ArcadaLabs/Ouro-2.6B-mlx-bf16)
- The same page gives no user-facing loop-count setting for MLX; it names `total_ut_steps` only in text. — [ArcadaLabs/Ouro-2.6B-mlx-bf16](https://huggingface.co/ArcadaLabs/Ouro-2.6B-mlx-bf16)
- Ouro loop count is set via the transformers config: `config.total_ut_steps` (default 4), with `early_exit_threshold` (default 1.0). vLLM ignores early exit and always runs 4 loops. Requires `transformers==4.54.1` or earlier; 4.56+ reportedly breaks KV cache handling. — [insiderllm Ouro guide](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/)
- The same guide says "No llama.cpp, no Ollama, no GGUF" for Ouro. — [insiderllm Ouro guide](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/)
- Ouro model cards list Ouro-1.4B, Ouro-2.6B, and Thinking variants; Apache-2.0; paper arXiv 2510.25741 ("Scaling Latent Reasoning via Looped Language Models"). Cards warn of transformers version compatibility. — [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B), [ByteDance/Ouro-2.6B README](https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/README.md)
- Dory: a modelfit.io post says mlx-lm has no Dory implementation and mlx-community has published no Dory conversion (as of its writing). It says no llama.cpp release loads a Dory GGUF until a PR merges. NVIDIA's documented Apple route is to clone a contributor fork, check out one exact revision, build llama.cpp with Metal, and run the server. — [modelfit.io Dory memory requirements](https://modelfit.io/blog/nvidia-nemotron-labs-dory-1-0-4b-mac-memory-requirements/)
- Dory vLLM support is a pending PR; llama.cpp support comes from a specific PR. — [explainx Dory](https://explainx.ai/blog/nvidia-nemotron-labs-dory-4b-looped-model-nvfp4-adjustable-loops-swe-bench-2026), [tilnote Dory llama.cpp](https://tilnote.io/pages/6af040a4d22a688f3e065952)
- Dory released 2026-10-16 (open weights, ~4B, 256K context). Quants reported: BF16, MXFP8, NVFP4, Q5_K_M GGUF. — [aiweekly Dory](https://aiweekly.co/alerts/nvidias-dory10-4b-uses-looped-layers-to-match-9b-rivals), [modelfit.io](https://modelfit.io/blog/nvidia-nemotron-labs-dory-1-0-4b-mac-memory-requirements/)
- Huginn (tomg-group-umd/huginn-0125, 3.5B, arXiv 2502.05171) documents a `num_steps` argument for recurrent iterations at inference, loaded via Transformers with `trust_remote_code=True`. No MLX port or GGUF was found in search. — [huginn-0125 model card PR](https://huggingface.co/tomg-group-umd/huginn-0125/blob/refs%2Fpr%2F6/README.md?code=true), [arXiv 2502.05171](https://arxiv.org/html/2502.05171v1)
- Huginn community fork with standard Llama-style module names may simplify porting. — [KristianS7/huginn-0125](https://huggingface.co/KristianS7/huginn-0125)
- mlx-lm has one model file per architecture under `mlx_lm/models/`, plus a `MODEL_REMAPPING` dict for aliases. That is the port target. — [DeepWiki mlx-lm model architectures](https://deepwiki.com/ml-explore/mlx-lm/5-model-architectures) (AI-generated wiki; verify against the repo)

### Inferences
- Per-request/per-token loop control is most practical in MLX: the forward pass is Python, so a `loops` argument on the model's `__call__` is straightforward. llama.cpp would need arch-level changes to expose loop count per call; the sources only describe loop count as a deployment setting for Dory. Ollama is out.
- A Dory port to MLX means writing `mlx_lm/models/dory.py`: 8 input layers, a shared recurrent block run N times with the input re-injected each pass, 8 output layers, hybrid sliding/full attention, and a `loops` arg. The Ouro `ouro.py` file is the closest template. Expect a few hundred lines and a parity test against the PyTorch/transformers reference (the fork), not against the GGUF.
- A Huginn port is simpler in principle (its `num_steps` is already a forward arg) but is a 3.5B model with a different architecture and no community MLX conversion, so it is a second-choice target.
- Speed reference point: the same Ouro-2.6B bf16 run was ~1.7x faster under MLX than PyTorch MPS (see Tokens/sec section), so MLX is the right runtime if you can port.

### Gaps
- No primary NVIDIA model card was retrieved (the Hugging Face page did not render in the fetch). Layer counts, max loops, and the parameter name for loop count are from secondary blogs and conflict (see below).
- No verified mlx-community Dory or Huginn repo was found.
- No llama.cpp PR was opened to confirm; the status of the Dory llama.cpp PR as of November 2026 is unverified.

## Quantization and expected decode tokens/sec on M4 base

### Takeaway
For a 4B dense-equivalent on M4 base at ~120 GB/s, expect roughly 30-40 tok/s at 4-bit and 18-25 tok/s at 8-bit for one loop, if weights are read once per token. Looped models re-read the shared block weights once per loop, so tok/s falls roughly as 1/(effective layers read per token). Benchmark the actual Dory port before relying on any number.

### Cited Findings
- Base M4 has ~120 GB/s memory bandwidth; decode is bandwidth-bound. A binned 8-core-GPU M4 is ~20% slower than the full 10-core. Reported Llama 3.1 8B Q4 decode on base M4 is 18-28 tok/s (Ollama Q4_K_M, and 23-28 tok/s on a 16 GB MacBook Air). — [specpicks M4 Llama 3.1 8B](https://specpicks.com/reviews/run-llama-3-1-8b-on-m4) [Claimed/Measured by third party, unverified]
- Llama-3.2 1B/3B quantization study (single tool report): 8-bit nearly lossless (~2% top-token flips, perplexity barely moves); 4-bit flips 15-21% of top tokens and perplexity rises by ~1 point or more, worst on 1B (+3.5 at 4-bit for 1B, +1.4 for 3B). — [mlx-quant-fidelity](https://pypi.org/project/mlx-quant-fidelity/0.3.0/) [Claimed; single tool]
- Counterpoint: one OCR/VLM blog says 4-bit was not the sweet spot there. Community guidance is to start at 4-bit and move to 8-bit if quality suffers. — [DEV Community OCR](https://dev.to/vimal_nakrani/i-tested-quantized-unlimited-ocr-on-mac-4-bit-was-not-the-sweet-spot-f2k), [mlx-quant-fidelity](https://pypi.org/project/mlx-quant-fidelity/0.3.0/)
- Mixed precision: keeping a few sensitive layers at 8-bit recovers most quality (reported for KV cache: perplexity 21 to 507 at uniform 4-bit KV on one model). — [mlx-optiq blog](https://mlx-optiq.pages.dev/blog/not-all-layers-are-equal) [Claimed]
- Commands: `mlx_lm.convert -q` for 4-bit; `--q-bits 8 --q-group-size 64` for 8-bit. — [mlx-lm PyPI](https://pypi.org/project/mlx_lm)
- Ouro-2.6B bf16 under MLX on an M5 Pro 64 GB: 1024 tokens in 85.3 s = ~12.0 tok/s, peak 5.76 GB. Same model with PyTorch fp16 on MPS: ~6.9 tok/s, 4.97 GB. Single informal prompt. — [ArcadaLabs Ouro MLX](https://huggingface.co/ArcadaLabs/Ouro-2.6B-mlx-bf16) [Measured, informal, M5 not M4]
- Ouro 4-bit MLX build exists: mlx-community/Ouro-2.6B-4bit. — [ArcadaLabs Ouro MLX](https://huggingface.co/ArcadaLabs/Ouro-2.6B-mlx-bf16)
- Dory Q5_K_M GGUF is ~3.675 GB; estimated VRAM at Q4 for Ouro-2.6B is ~1.6 GB, at FP16 ~5.2 GB (estimates, not measured). — [modelfit.io](https://modelfit.io/blog/nvidia-nemotron-labs-dory-1-0-4b-mac-memory-requirements/), [insiderllm](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/)

### Inferences (first-principles, labeled [Est])
Assumptions: M4 base effective bandwidth ~85-100 GB/s (about 75% of the 120 GB/s peak). Weight bytes per token: 4-bit MLX (group 64) ~4.5 bits/weight; 8-bit ~8.5 bits/weight. Dory-like layout: 36 effective layers at one loop, equal per-layer size, shared recurrent block read once per loop.

| Model | Quant | Weights in memory | Loops | Bytes read per token | Est. decode tok/s |
|---|---|---|---|---|---|
| Dense 4B-class (baseline) | 4-bit | ~2.3 GB | 1 | ~2.3 GB | ~35-40 |
| Dense 4B-class (baseline) | 8-bit | ~4.3 GB | 1 | ~4.3 GB | ~20-23 |
| Dory-like, 8+20+8 layers | 4-bit | ~2.3 GB | 1 | ~2.3 GB | ~35-40 |
| Dory-like, 8+20+8 layers | 4-bit | ~2.3 GB | 2 | ~3.6 GB | ~23-26 |
| Dory-like, 8+20+8 layers | 4-bit | ~2.3 GB | 4 | ~6.1 GB | ~14-16 |
| Dory-like, 8+20+8 layers | 8-bit | ~4.3 GB | 2 | ~6.8 GB | ~12-14 |
| Dory-like, 8+20+8 layers | 8-bit | ~4.3 GB | 4 | ~11.5 GB | ~7-8 |

Cross-check: Ouro-2.6B bf16 (5.2 GB) measured ~12 tok/s on M5 Pro (~300 GB/s bandwidth) with 4 loops. If weights were read once per token it would be ~50+ tok/s; 4 reads per token (~21 GB/token) gives ~14 tok/s, consistent with the measurement. So the "re-read per loop" model is roughly right and the table is likely optimistic for M4 base (add 10-30% overhead for Python and dispatch). Treat the table as an upper bound until measured.

Recommendation: Dory 4-bit at loops=2 (the card's default per secondary sources) is the sweet spot here, roughly 20-25 tok/s est. 8-bit with loops 4 is ~7-8 tok/s est. and not worth it on this bandwidth.

### Gaps
- No published base-M4 MLX decode numbers for a 4B model in the sources; the 8B Q4 reports are Ollama/llama.cpp, not MLX.
- No independent quality numbers for Dory 4-bit vs 8-bit. Loop count and quantization interact (lower precision may hurt a recurrent block more, since errors compound per loop). Measure this.

## Co-running a controller, memory budget, and tips

### Takeaway
Keep the controller inside the same MLX process. A linear probe on the looped model's hidden states is the cheapest option (one matmul, no second model), so use that first. A 0.5-1B classifier is the fallback. Memory is not a constraint at these sizes on 24 GB; the wired-memory limit and the KV cache are.

### Cited Findings
- `iogpu.wired_limit_mb` raises the GPU wired-memory limit (sysctl). Apple does not document it. `mlx.core.set_wired_limit` and `device_info()` (`max_recommended_working_set_size`) are the MLX-side API. The mlx-lm README example uses 20480 MB on a 24 GB Mac. Community notes say it resets on reboot and leave 6-8 GB for macOS. — [MLX set_wired_limit docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html), [Towards AI](https://pub.towardsai.net/5-ollama-and-mlx-settings-to-optimise-for-a-local-llm-on-a-mac-in-2026-f3d4bbfdfb07), [bestllmfor](https://bestllmfor.com/guide/optimiser-mac-silicon-llm/)
- Warning: one 24 GB user reported that raising the limit made crashes worse. — [kartit.net](https://www.kartit.net/blog/qwen38-27b-24gb-macbook) [single anecdote]
- Sources disagree on default limit (about 75% of RAM vs about 70%). — [bestllmfor Qwen](https://bestllmfor.com/guide/qwen3-32b-local-mac-m1-m2/), [Hannecke Medium](https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc)
- MLX memory-safety and crash guides: [Hannecke, MLX memory management](https://medium.com/@michael.hannecke/how-my-local-coding-agent-crashed-my-mac-and-what-i-learned-about-mlx-memory-management-e0cbad01553c), [DEV MLX memory checklist](https://dev.to/sleepyquant/mlx-memory-safety-checklist-6-layer-defense-for-m1m2-apple-silicon-2cbj)

### Inferences (memory budget, [Est])
- Dory 4-bit weights ~2.3-3.7 GB; 8-bit ~4.3 GB. Controller 0.5B 4-bit ~0.3 GB; 1B 4-bit ~0.6 GB; probe ~ negligible.
- KV cache is the real driver. Last-loop KV caching (one cache per recurrent attention layer) keeps KV at one-loop size; Ouro-style per-loop caching multiplies KV by loops. At 256K context, KV size must be measured; keep context under ~8-16K on 24 GB unless measured otherwise.
- Target wired limit: ~18 GB (18432 MB) with 6 GB for macOS, leaving headroom for the controller and KV. Set it with `sudo sysctl iogpu.wired_limit_mb=18432` and re-apply after reboot.
- Latency overhead: a linear probe is a few microseconds on GPU per token. A 0.5B classifier forward on a short prefix is single-digit ms, not per token. Run it once per request or per loop boundary, not per token, unless measured otherwise.

### Inferences (per-token loop control)
- A per-token `loops` argument is compatible with last-loop KV caching: each token writes one KV per recurrent layer regardless of loop count, so loop count can change token to token without cache reshaping. Per-loop caching (Ouro-style) also allows it but uses fixed slots per loop, so it wastes memory. Verify with a parity test that mixed loop counts do not corrupt the cache.
- Controller signals: the probe reads the hidden state after loop k (or the final-loop state). Early-exit gates (Ouro) are a built-in variant of the same idea; use it as a baseline.

### Tips (MLX)
- mx.compile: compile the per-loop block as a function; for variable loop count, compile per loop-count value (1, 2, 4) rather than per token to avoid recompilation.
- Avoid per-loop Python overhead: keep the loop in the model's `__call__` with a Python `for` over a fixed-size module list; no per-loop `mx.eval`. Call `mx.eval` once per token.
- Prompt caching: use mlx-lm's prompt cache for the prefix; the cache must be the class built for the Dory model (custom attention), so reuse it only with the same loop schedule for the prefix, or disable it for prefix tokens processed at a different loop count.
- Controller in the same process shares GPU and wired memory with the main model; run it on the same stream to avoid sync stalls.

### Gaps
- No published Dory or Huginn MLX inference numbers; no published M4 controller-overhead measurements.
- Whether Dory's last-loop KV design works with per-token loop changes is inferred from the description, not from code. Verify with the reference implementation.

## Source conflicts to resolve before building

- Dory recurrent layers: MindStudio says 20 shared recurrent layers (8 in, 8 out); another MindStudio post says 40 recurrent layers. Check the config.
- Dory max loops: one source says 1-4 loops per token; another says "up to eight times per token." MindStudio benchmarks 1, 2, 4 loops; default 2 per two secondary sources.
- Dory paper claim "1.3-2.5x fewer tokens with 4 loops" appears only in aggregator text.
- Ouro-2.6B MATH-500 claim (90.85 vs 62.30 for Qwen3-8B) is third-party, unverified.
- Huginn: the `num_steps` argument appears in a PR-branch README (refs/pr/6), not the main card; confirm on main.

## Next steps (ordered)
1. Pull the Dory config from Hugging Face (nvidia/...) and the reference modeling code from the contributor fork. Record layer counts, loop parameter name, and KV design.
2. Port to MLX: `mlx_lm/models/dory.py` modeled on ArcadaLabs `ouro.py`. Parity-check logits against the reference on the same prompt at loops 1, 2, 4.
3. Convert 4-bit and 8-bit; measure tok/s and peak memory at loops 1/2/4 with `mlx_lm.generate` on this M4 base, and compare to the table above.
4. Train or fit the controller: start with a linear probe on hidden states; test a 0.5B classifier only if the probe underperforms.
5. Set `iogpu.wired_limit_mb` to 18432 and measure with a 4K and 16K context.

## Sources
- [ArcadaLabs/Ouro-2.6B-mlx-bf16](https://huggingface.co/ArcadaLabs/Ouro-2.6B-mlx-bf16)
- [insiderllm Ouro-2.6B-Thinking guide](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/)
- [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B), [ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/README.md)
- [modelfit.io Dory on Mac](https://modelfit.io/blog/nvidia-nemotron-labs-dory-1-0-4b-mac-memory-requirements/)
- [MindStudio Dory architecture](https://www.mindstudio.ai/blog/nvidia-dory-1-0-4b-looped-model)
- [aiweekly Dory](https://aiweekly.co/alerts/nvidias-dory10-4b-uses-looped-layers-to-match-9b-rivals)
- [explainx Dory](https://explainx.ai/blog/nvidia-nemotron-labs-dory-4b-looped-model-nvfp4-adjustable-loops-swe-bench-2026)
- [tilnote Dory llama.cpp](https://tilnote.io/pages/6af040a4d22a688f3e065952)
- [orcarouter Dory](https://www.orcarouter.ai/blog/nemotron-labs-dory-1-0-4b-quiet-release)
- [huginn-0125 model card (PR ref)](https://huggingface.co/tomg-group-umd/huginn-0125/blob/refs%2Fpr%2F6/README.md?code=true), [huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125), [KristianS7/huginn-0125](https://huggingface.co/KristianS7/huginn-0125)
- [arXiv 2502.05171 (Huginn)](https://arxiv.org/html/2502.05171v1)
- [specpicks M4 Llama 3.1 8B](https://specpicks.com/reviews/run-llama-3-1-8b-on-m4)
- [mlx-quant-fidelity](https://pypi.org/project/mlx-quant-fidelity/0.3.0/)
- [mlx-lm PyPI](https://pypi.org/project/mlx_lm)
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)
- [DeepWiki mlx-lm architectures](https://deepwiki.com/ml-explore/mlx-lm/5-model-architectures)
- [MLX set_wired_limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
- [Towards AI MLX/Ollama settings](https://pub.towardsai.net/5-ollama-and-mlx-settings-to-optimise-for-a-local-llm-on-a-mac-in-2026-f3d4bbfdfb07)
- [bestllmfor optimise Mac](https://bestllmfor.com/guide/optimiser-mac-silicon-llm/)
- [kartit.net 24 GB anecdote](https://www.kartit.net/blog/qwen38-27b-24gb-macbook)
- [mlx-optiq layers blog](https://mlx-optiq.pages.dev/blog/not-all-layers-are-equal)
- [DEV OCR 4-bit](https://dev.to/vimal_nakrani/i-tested-quantized-unlimited-ocr-on-mac-4-bit-was-not-the-sweet-spot-f2k)

Not used (no direct evidence): llama.cpp GitHub issue search for Ouro (no PR found in search; check github.com/ggml-org/llama.cpp directly), Apple's ML Research MLX M4/M5 post (base M4 numbers not retrieved).
