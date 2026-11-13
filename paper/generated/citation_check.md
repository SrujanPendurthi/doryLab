# Citation check (fetched 2026-11-12)

All URLs resolved. "Supports" = the fetched page supports the claim as worded.

| Key | URL | Resolves | Claim the paper uses it for | Supports |
|---|---|---|---|---|
| dory2026 | https://arxiv.org/abs/2611.03998 | yes | Dory1.0-4B is a looped nGPT whose recurrent block runs a variable number of times (1-4 loops), last-loop-only KV cache | yes |
| dory2026 (self-spec) | same | yes | Paper itself describes self-speculative decoding (k=1 draft, k=4 verify) | UNVERIFIED: abs page does not mention it; the research note cites the HTML body. Task 4: cite only as "per the paper body" or drop |
| dorycard | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | yes | Card recommends temperature 1.0 / top-p 0.95 with reasoning on (we use greedy, thinking off); default 2 loops; Q5_K_M GGUF export exists; benchmarks are BF16 only | yes |
| ouro2025 | https://arxiv.org/abs/2510.25741 | yes | Ouro: pretrained looped LMs (1.4B/2.6B) with entropy-regularized depth allocation | yes |
| huginn2025 | https://arxiv.org/abs/2502.05171 | yes | Recurrent-depth model that scales test-time compute by unrolling a recurrent block | yes |
| cdb2026 | https://arxiv.org/abs/2608.09444 | yes | Continuous Depth Batching serves depth-adaptive looped LMs (Ouro, Huginn) | yes. The "could not reproduce near-lossless shared KV" claim is NOT on the abs page; do not cite it from the abstract |
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
1. Drop or soften "Dory paper describes self-speculative decoding" unless verified against the paper body.
2. Do not attribute the CDB shared-KV non-reproduction finding from the abstract.
3. vLLM PR #60149 is merged per the live page; do not call it unmerged.
4. Not included (not fetched, not needed): Laya's preprint, secondary blog posts, weight-space papers.
