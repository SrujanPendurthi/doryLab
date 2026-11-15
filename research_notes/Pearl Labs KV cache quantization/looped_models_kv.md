# Looped / Recurrent-Depth Models: KV Cache Memory and Compression

Status as of 2026-11-13. Sources were read via WebFetch/WebSearch; where a primary source could not be read, this is stated. Estimates are labeled "(estimate)".

## Q1. How do Ouro, Huginn, Dory, and CDB handle KV across loops?

### Takeaway
Looped models share weights across loops but not KV states, so a naive cache grows linearly with loop count. Three options appear in the literature: a separate cache per loop (Dory's default), a single shared slot per token (Ouro's "single-slot"), and keeping only the last loop's cache. Sharing has not reproduced cleanly on Ouro. Dory's own default is per-loop.

### Cited Findings
- Ouro (LoopLM) is a weight-tied recurrent-depth model, 1.4B and 2.6B, trained with an entropy-regularized depth objective. Its paper reports KV sharing as near-lossless (per the CDB appendix quoting it) — [Ouro arXiv](https://arxiv.org/abs/2510.25741)
- Huginn-3.5B runs a recurrent block for a sampled number of iterations; the abstract says the design supports "KV-cache sharing" and speculative decoding. The KV section was not visible in the fetched abstract, so the exact mechanism is unverified — [Huginn arXiv](https://arxiv.org/pdf/2502.05171)
- CDB (Continuous Depth Batching) Section 4.2 "Depth-aware KV Cache": tokens that exit early write no KV for deeper slots. Two fixes are offered: a *last-exited cache* (copy final KV into skipped slots; keeps r_max slots, adds copy overhead) and a *shared cache* (one KV slot per token per recurrent layer, overwritten each loop; "reducing recurrent-core KV memory by a factor of r_max"). The main experiments use the shared cache (Appendix A.2, Table 1). r_max = 4 for Ouro and 16 for Huginn (Section 6.1). Section 7 says the shared cache "may require specialized training," but "looped models can often tolerate this without additional training, though support for KV sharing is not guaranteed" — [CDB arXiv HTML v2](https://arxiv.org/html/2608.09444v2)
- **CDB Appendix E.1 (KV Cache Sharing), the reproduction failure you asked about.** The authors write: "We cannot reproduce the results from the Ouro paper, which reports near-lossless KV sharing." They suggest the cause "might be because they share only during decoding and keep a depth-indexed cache for the prompt." Ouro reportedly loses about 6 points under first-then-shared, and collapses under single-slot sharing. In CDB's Table 4 footnote, Ouro's single-slot result is 0.23%, against 78.85% reported in the Ouro paper. The footnote adds that Vendrell et al. (2026) find Ouro "requires specialized fine-tuning to perform well with single slot KV cache sharing." Ouro full depth-indexed accuracy is 78.09%, first-then-shared 72.40% (same source). I did not fetch Vendrell et al.; that citation is secondhand — [CDB arXiv HTML v2, App. E.1](https://arxiv.org/html/2608.09444v2)
- CDB reports speedups (up to 99% of the theoretical adaptive-depth maximum; 1.5–1.9x offline throughput on Ouro 1.4B and Huginn 3.5B). Its abstract mentions "looped KV-caching" but gives no memory numbers — [CDB arXiv abs](https://arxiv.org/abs/2608.09444)
- KV-replay paper (NeurIPS 2026, "Architectures and KV Replay for Depth-Recurrent Language Models"). It keeps only the final recurrence's KV and reuses it for all earlier recurrences. It uses Parcae-family models (1.3B, 770M, 370M), not Ouro/Huginn/Dory. Measured: at batch 64, context 2048, 1.3B, "KV-shareability reduces measured global HBM usage by 66.2% from 63.82 GB to 21.60 GB." Without a fix, shared KV costs accuracy (Core 37.27 → 33.94 at 1.3B; 370M Core 20.74 → 15.16). A "BeNice!" second-stage training recovers it (Core 37.40 → 37.43; validation PPL 10.79 → 10.78) — [NeurIPS 2026 abstract](https://neurips.cc/virtual/2026/179998); [full text](https://arxiv.org/html/2610.31633)
- Dory-1.0-4B, per the llama.cpp PR: "KV cache 'separate per loops', with the last loop's cache shared." Mode `last_loop` "retains historical tokens' final-loop KV"; the author notes this "changes predictions compared with separate caches, and prefill chunk size matters." Supports k=1–4 loops; layers = 16 + 40×k + 16. Metal is untested; K>4 is unsupported — [llama.cpp PR #30306](https://github.com/ggml-org/llama.cpp/pull/30306)
- Dory in vLLM: per-loop KV is the default; a later commit adds a "last loop only" mode. Merged into vllm main on 2026-10-25 (PR opened 2026-10-06) — [vLLM PR #60149](https://github.com/vllm-project/vllm/pull/60149)
- Dory is a 4B open-weight looped model released 2026-10-16. Secondary reports say SWE-bench Verified rises from 50.3 (1 loop) to 61.8 (4 loops), and the model card recommends 2 loops. All of these are vendor-reported — [aiweekly summary](https://aiweekly.co/alerts/nvidias-dory10-4b-uses-looped-layers-to-match-9b-rivals); [MindStudio summary](https://www.mindstudio.ai/blog/nvidia-dory-1-0-4b-looped-model)

### Inferences
- Dory was built for per-loop KV (default in both llama.cpp and vLLM). Single-slot or shared KV is therefore not something the model was trained for, and the Ouro result suggests it would need fine-tuning to work (inference, not measured on Dory).
- The `last_loop` mode in llama.cpp is a memory lever with an accuracy cost, and it is not the same thing as quantization. Those two should be measured separately.
- The "~4x normal KV per token" for per-loop at k=4 (a commit message quoted in the llama.cpp/vLLM discussions) matches the structure: four loop-indexed caches for each layer that appears in the recurrent block (inference from the PR structure, not a measurement).

### Gaps
- The Dory model card (huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B) returned HTTP 401 to WebFetch. I could not read its KV, cache-mode, or quantization text. Search summaries describe the card but are not primary.
- The Dory arXiv paper was not found. Author names (Fishman, Sun, Ginsburg) come from secondary coverage only.
- Dory layer counts conflict: 16/40/16 (llama.cpp PR, Korean source) vs 8/20/8 (aiweekly summary). The PR is more specific, so I use 16/40/16 with that caveat.
- Dory licensing is reported inconsistently (OpenMDW-1.1 vs NVIDIA Open Model License). Check the card before any commercial use.
- No measured KV bytes-per-token for Dory (layers, KV heads, head dim) was found.

## Q2. Does the Dory paper or model card say anything about KV size, cache modes, or quantization?

### Takeaway
The only primary-source KV statements I could read are the llama.cpp and vLLM PR texts, which describe per-loop KV as the default and a last-loop option. The model card, which is the main place this would be documented, was not readable.

### Cited Findings
- Per-loop KV default and last-loop mode, llama.cpp PR #30306 — [PR](https://github.com/ggml-org/llama.cpp/pull/30306)
- Per-loop KV default and "last loop only" mode, vLLM PR #60149 — [PR](https://github.com/vllm-project/vllm/pull/60149)
- Quantized formats tested in the llama.cpp PR: bf16, q4_k_m, q5_k_m. The PR reports 96/96 numerical checks (CPU and CUDA) and 36/36 real-checkpoint serving smoke tests. Metal is untested — [PR #30306](https://github.com/ggml-org/llama.cpp/pull/30306)
- Dory's HF card lists BF16, NVFP4, MXFP8, and GGUF weights (secondary) and a 256K context window (secondary) — [MindStudio summary](https://www.mindstudio.ai/blog/nvidia-dory-1-0-4b-looped-model)

### Inferences
- No source I could read gives a KV-quantization recommendation from NVIDIA for Dory. The lack of one is itself a signal that the team has not published KV-quant results for this model.

### Gaps
- Model card text on KV, cache modes, and quantization: not readable (HTTP 401). Needs a manual check or an authenticated fetch.
- Dory paper: not located.

## Q3. Do recurrent loops amplify KV quantization error?

### Takeaway
There is no direct measurement of KV-quant error amplification in Dory, Ouro, or Huginn. Adjacent literature points both ways. Quantization error does compound along recurrent trajectories in general (LoopQ, for weights and activations), and FlashLoop argues the residual KV between adjacent loops is easier to quantize at low bit, not harder.

### Cited Findings
- KVarN: KV quantization error accumulates across layers and decoding steps, and it compounds with longer generation. It proposes a Hadamard-rotated, variance-normalized quantizer that reduces accumulation at 2-bit on reasoning tasks. This is about depth and time, not loops — [KVarN arXiv](https://arxiv.org/pdf/2606.03458)
- LoopQ: post-training quantization for looped models. Errors "propagate along the recurrent trajectory and are amplified by subsequent loop-dependent transformations." It targets weights and activations, not the KV cache — [LoopQ arXiv](https://arxiv.org/abs/2605.16343)
- FlashLoop: KV residuals between adjacent loops "become progressively more amenable to low-bit quantization." Its abstract reports up to 6x KV memory reduction and up to 1.64x end-to-end speedup with "lossless accuracy" on "several Looped Transformers models," without naming them — [FlashLoop arXiv](https://arxiv.org/abs/2609.29812)
- CDB reports that shared-cache errors come from a token-independent basis shift in the first recurrence (KV-replay paper, Section 3.2). This is about sharing, not quantization — [KV-replay full text](https://arxiv.org/html/2610.31633)

### Inferences
- Per-loop KV quantization is a different error path from sharing. Per-loop quantization gives each loop its own noise; later loops attend to quantized keys from earlier loops. Whether that compounds across loops is untested. FlashLoop's residual observation suggests the inter-loop redundancy could make quantization safer, not worse.
- Your measured -1.5 points at q8_0 is larger than typical q8 KV loss on dense models, which may mean the per-loop error compounds, or may be unrelated (for example, the cache mode or prefill chunking). Test it: run k=1 vs k=4 at q8_0 and f16 on the same holdout. If the q8 loss grows with k, the amplification question is answered for your setup.

### Gaps
- No paper measures KV-quant error vs loop count on Dory, Ouro, or Huginn.
- FlashLoop's full text (bit-widths, which models) was not read.

## Q4. Can 4-bit rotated KV let k=4 run with more slots or 1024+ token outputs on 24 GB?

### Takeaway
Probably yes on paper for 1024-token outputs at 4 slots, but the estimate is sensitive to values I could not verify (Dory's per-loop KV dimensions, and how much of your 11 GB is KV). The bigger risk is the untested Metal path and the q8_0 accuracy loss, which 4-bit will likely make worse, not better.

### Cited Findings
- KV-quantization error accumulation is worst at low bits and long generations (KVarN, above). That argues against 4-bit without a rotation step, and it is the case you would be in with 1024-token outputs.
- llama.cpp PR: Metal is untested — [PR #30306](https://github.com/ggml-org/llama.cpp/pull/30306)

### Inferences (estimates)
Assumptions, all estimates:
- KV bytes per element: q8_0 ≈ 1.06 B (8.5 bits incl. scales); q4_0-class ≈ 0.56 B (4.5 bits). 4-bit rotated is in this range. Savings ≈ 47% of KV.
- Q5_K_M weights for 4B ≈ 2.9 GB. Looped weights are shared, so this does not scale with k.
- Your 11 GB at k=4, 4 slots includes weights (≈2.9 GB), compute buffers (unknown, maybe 0.5–1 GB), and KV (≈7–8 GB). The KV prompt+output length behind this is not stated, so the rows below are relative.
- KV scales linearly with slots × loops × tokens (prompt + output).

Relative scenarios (KV, using the ~7.5 GB baseline at 4 slots, k=4, q8_0, same length):
| Config | KV (est.) | Total with weights+buffers (est.) |
|---|---|---|
| q8_0, 4 slots, baseline length | ~7.5 GB | ~11 GB (measured) |
| q8_0, 4 slots, 2x length (1024 out, if baseline was 512 out) | ~15 GB | ~18–19 GB |
| 4-bit, 4 slots, 2x length | ~8 GB | ~11–12 GB |
| 4-bit, 8 slots, baseline length | ~7.5 GB | ~11 GB |

Read: 4-bit roughly doubles slot count or output length at the same memory. Metal's default GPU wired-memory cap on a 24 GB Mac is lower than 24 GB, so the q8_0 2x row (~18–19 GB) is likely to fail. Check `iogpu.wired_limit_mb` on the machine (not researched; from general knowledge).

Options from the literature that cut KV more than quantization:
- last_loop mode (llama.cpp): reduces per-loop KV toward 1x (estimate), with an accuracy cost and changed predictions (PR #30306).
- Shared cache (CDB, KV replay): reduces by r_max (4x at k=4) but requires training for Ouro-like models (CDB App. E.1; KV-replay Table 4).
- FlashLoop residual-quantization: up to 6x KV reduction reported, but not for Dory and not confirmed for these models.

### Gaps
- Dory KV dimensions (attention layers per loop, KV heads, head dim): not found. Needed for a real calculation.
- Whether llama.cpp's KV quant types are supported on the Dory path at all: not verified. Quantized V-cache usually requires flash attention in llama.cpp (from general knowledge, unsourced here).
- No measured accuracy for 4-bit rotated KV on any looped model. KVarN is the closest, and it is on non-looped reasoning models.
- Pearl Research Labs KV work: out of scope for this note (covered separately).

## Recommended tests (for the project)
1. Measure accuracy at k=1 and k=4 for f16, q8_0, and q4 KV on the holdout. The question is whether the -1.5 pt loss grows with k.
2. Test q4 KV with a rotation (if the llama.cpp build supports it) on k=4 at 1024 output tokens.
3. Run the Dory path on Metal directly (the PR says it is untested there).
4. Compare `last_loop` vs per-loop at k=4 for both accuracy and memory.

## Sources
- Ouro: https://arxiv.org/abs/2510.25741
- Huginn: https://arxiv.org/pdf/2502.05171
- CDB: https://arxiv.org/abs/2608.09444 and https://arxiv.org/html/2608.09444v2
- KV replay: https://neurips.cc/virtual/2026/179998 and https://arxiv.org/html/2610.31633
- llama.cpp PR #30306: https://github.com/ggml-org/llama.cpp/pull/30306
- vLLM PR #60149: https://github.com/vllm-project/vllm/pull/60149
- Dory HF card (401, unread): https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B
- KVarN: https://arxiv.org/pdf/2606.03458
- LoopQ: https://arxiv.org/abs/2605.16343
- FlashLoop: https://arxiv.org/abs/2609.29812
- Secondary Dory coverage: https://aiweekly.co/alerts/nvidias-dory10-4b-uses-looped-layers-to-match-9b-rivals ; https://www.mindstudio.ai/blog/nvidia-dory-1-0-4b-looped-model
