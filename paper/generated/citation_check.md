# Citation check (fetched 2026-11-12)

All URLs resolved. "Supports" = the fetched page supports the claim as worded.

| Key | URL | Resolves | Claim the paper uses it for | Supports |
|---|---|---|---|---|
| dory2026 | https://arxiv.org/abs/2611.03998 | yes | Dory1.0-4B is a looped nGPT whose recurrent block runs a variable number of times (1-4 loops), last-loop-only KV cache | yes |
| dory2026 (self-spec) | https://arxiv.org/html/2611.03998 | yes | Self-speculative decoding: same checkpoint drafts at k=1, verifies at k=4, gamma=4; 1.37-1.71x decode throughput (Fig. 6); 1.11-1.53x request duration (Table 7) | verified from full text, Sec. 6.2 (supersedes the abstract-only UNVERIFIED mark; Ruling P7) |
| dorycard | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | yes | Card recommends temperature 1.0 / top-p 0.95 with reasoning on (we use greedy, thinking off); default 2 loops; Q5_K_M GGUF export exists; benchmarks are BF16 only | yes |
| ouro2025 | https://arxiv.org/abs/2510.25741 | yes | Ouro: pretrained looped LMs (1.4B/2.6B) with entropy-regularized depth allocation | yes |
| huginn2025 | https://arxiv.org/abs/2502.05171 | yes | Recurrent-depth model that scales test-time compute by unrolling a recurrent block | yes |
| cdb2026 | https://arxiv.org/abs/2608.09444 | yes | Continuous Depth Batching serves depth-adaptive looped LMs (Ouro, Huginn); Appendix E.1 reports it could not reproduce Ouro's near-lossless KV sharing | yes. Verified from the full text, App. E.1: "We cannot reproduce the results from the Ouro paper, which reports near-lossless KV sharing"; Ouro r=4: 78.09 / 72.40 / 0.23 for full, first-then-shared and single-slot caches |
| ngpt2024 | https://arxiv.org/abs/2410.01131 | yes | nGPT normalized-transformer backbone that Dory builds on | yes |
| llamacpp30306 | https://github.com/ggml-org/llama.cpp/pull/30306 | yes | Dory support in llama.cpp (GGUF conversion, 1-4 loop control, last_loop KV mode) is an unmerged PR; we use a pinned commit | yes (open, marked ready for review 2026-11-12; Metal listed as untested) |
| vllm60149 | https://github.com/vllm-project/vllm/pull/60149 | yes | vLLM has Dory support for loop count via the PR | yes, but state differs from notes: page shows MERGED 2026-10-25, while research notes (and the model card) say "under review". Task 4: say "added in vLLM PR #60149", not "unmerged". Parameter name n_recurrent_loops is on the model card, not the PR page |
| nvclassifier | https://huggingface.co/nvidia/prompt-task-and-complexity-classifier | yes | ~0.2B DeBERTa multi-head classifier giving 11 task categories (incl. code generation) and complexity scores; basis for the routing features / ANE classifier | yes |
| laya | https://huggingface.co/convaiinnovations/laya | yes | Related typed-decision model with ~33 ms forward pass; not generative | yes (self-reported latency on Tesla T4; calibration needed). Author is an org handle; the accompanying paper is a Preprints.org item, not cited |
| gsm8k2021 | https://arxiv.org/abs/2110.14168 | yes | GSM8K benchmark | yes |
| mmlupro2024 | https://arxiv.org/abs/2406.01574 | yes | MMLU-Pro benchmark (10 options, reasoning-heavy) | yes |
| mbpp2021 | https://arxiv.org/abs/2108.07732 | yes | MBPP benchmark (974 tasks) | yes |
| leviathan2023 | https://arxiv.org/abs/2211.17192 | yes | Speculative decoding: draft and parallel verify, output distribution unchanged | yes |
| layerskip2024 | https://arxiv.org/abs/2404.16710 | yes | Self-speculative decoding drafting from early layers, verified by the rest | yes |
| draftverify2024 | https://arxiv.org/abs/2309.08168 | yes | Self-speculative decoding by skipping layers in the same model, lossless, no extra memory | yes |

## Flags for Task 4
1. SUPERSEDED: Dory self-speculative decoding is verified from the full text, Sec. 6.2 (the paper credits it to Dory).
2. SUPERSEDED: the CDB shared-KV non-reproduction is verified from Appendix E.1 and may be cited.
3. vLLM PR #60149 is merged per the live page; do not call it unmerged.
4. Not included (not fetched, not needed): Laya's preprint, secondary blog posts, weight-space papers.
