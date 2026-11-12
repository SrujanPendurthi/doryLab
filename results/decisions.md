# Decisions and gate results

## Task 1

- llama.cpp: PR #30306 ("dory support for llama.cpp", OPEN) is the Dory PR. LLAMA_REF = 1c3cdd23787750814aa21a070dc4d6dc451af871 (PR head; build reports `0.8.0-dev commit 1c3cdd2`). Pinned the PR head, not master (master support for `dory` arch not checked). Shallow clone; built targets: llama-server, llama-cli, llama-completion, llama-lookup.
- Deviation: this llama.cpp has no `-no-cnv` in llama-cli (only `-st`); `-no-cnv` lives in `llama-completion`, so the smoke test uses `llama-completion` (added to the build targets).
- Model repos: GGUF `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF` rev df53755cb8741843af8e019e088819fbef17e29b, file `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (3.67 GB), stored at models/gguf/, symlinked as models/dory-q5.gguf. BF16 `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` rev f6c832e6963ddb74f62c75813a3e8c74b1af67b1 (config.json, tokenizer_config.json, chat_template.jinja only, in models/bf16-meta/).
- Layer counts (config.json): n_input_layers=16 / n_recurrent_layers=40 / n_output_layers=16 (72 layer_types entries incl. mlp; NOT 8/20/8), default n_recurrent_loops=2, recurrent_kv_cache_mode=per_loop, sliding_window=1024.
- Thinking toggle: chat template var `enable_thinking` (default True). True -> generation prompt `<|im_start|>assistant\n<think>\n`; False -> `<|im_start|>assistant\n<think></think>`. Disable via llama-server `--chat-template-kwargs '{"enable_thinking":false}'` (also `--reasoning off` exists) or per-request `chat_template_kwargs`.
- Gate: K=1 vs K=4 outputs differ (e.g. "2+2=4. 4-2=2..." vs "2+2=4. 4*4=16..."): PASS.
- Embedding: llama-server `--override-kv dory.num_loops=int:1 --embeddings --pooling last -ngl 99`, POST /embedding {"content":"hello"} returns vector len > 100: PASS. Probe feasible (Task 9 not skipped).
- Draft-override spike: `llama-server --help` has no draft-side `--override-kv` (only `--override-tensor-draft`, `--cache-type-k/v-draft`). Task 11 is SKIPPED.

## Task 3

## Task 5

## Task 6

## Task 8

## Task 9

## Task 10

## Task 11
