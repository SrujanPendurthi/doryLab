# Dory1.0-4B-BF16 tensor inventory (rev f6c832e6963ddb74f62c75813a3e8c74b1af67b1)

Source: `models/bf16/model.safetensors` (507 tensors, all BF16, 7.95 GB), `config.json`, and the GGUF
`models/gguf/dory-base-bf16.gguf` (also 507 tensors). Converter mapping: `vendor/llama.cpp/conversion/dory.py`
plus gguf-py's tensor map; scale vectors carry no `.weight` suffix in HF and get `.weight` appended in GGUF.

## Config facts
hidden 2560, intermediate 7168, vocab 131072, 16 q heads, 4 kv heads, head_dim 256 (q/g width 4096, k/v width 1024),
untied embeddings, 72 physical layers = 36 attention + 36 MLP, alternating (even index = attention, odd = MLP).
`n_recurrent_loops` = 2 in config.json (and GGUF `dory.num_loops` = 2); `server_args` overrides it with the k argument.
Attention type per attention layer: `full_attention` when layer index % 8 == 6, else `sliding_attention` (window 1024).
RoPE theta: full 1e7, sliding 1e4. Attention/MLP biases: none. GGUF metadata also stores layer_pattern, rope profile, alpha_init 0.05, gate_init 50.596.

## Blocks (physical layer indices)
| Block | Layers | Count | Run |
|---|---|---|---|
| input | 0-15 | 16 (8 attn + 8 MLP) | once |
| recurrent | 16-55 | 40 (20 attn + 20 MLP) | looped k times, weights shared across loops |
| output | 56-71 | 16 (8 attn + 8 MLP) | once |

Editing a recurrent-layer tensor changes every loop; there are no per-loop weights.

## Global tensors
| HF name | Shape | Dtype | GGUF name |
|---|---|---|---|
| `backbone.embeddings.weight` | [131072, 2560] | BF16 | `token_embd.weight` |
| `lm_head.weight` | [131072, 2560] | BF16 | `output.weight` |
| `lm_head.logit_scale` | [131072] | BF16 | `output_scale.weight` |

## Attention layer L (L even, 36 layers; 0,2,...,70)
| HF name (`backbone.layers.L.`) | Shape | Dtype | GGUF name (`blk.L.`) |
|---|---|---|---|
| `attn_alpha` | [2560] | BF16 | `residual_alpha.weight` |
| `mixer.q_proj.weight` | [4096, 2560] | BF16 | `attn_q.weight` |
| `mixer.k_proj.weight` | [1024, 2560] | BF16 | `attn_k.weight` |
| `mixer.v_proj.weight` | [1024, 2560] | BF16 | `attn_v.weight` |
| `mixer.g_proj.weight` | [4096, 2560] | BF16 | `attn_gate.weight` |
| `mixer.o_proj.weight` | [2560, 4096] | BF16 | `attn_output.weight` |
| `mixer.sqk` | [4096] | BF16 | `attn_qk_scale.weight` |
| `mixer.s_gate` | [4096] | BF16 | `attn_gate_scale.weight` |

## MLP layer L (L odd, 36 layers; 1,3,...,71)
| HF name (`backbone.layers.L.`) | Shape | Dtype | GGUF name (`blk.L.`) |
|---|---|---|---|
| `mlp_alpha` | [2560] | BF16 | `residual_alpha.weight` |
| `mixer.gate_proj.weight` | [7168, 2560] | BF16 | `ffn_gate.weight` |
| `mixer.up_proj.weight` | [7168, 2560] | BF16 | `ffn_up.weight` |
| `mixer.down_proj.weight` | [2560, 7168] | BF16 | `ffn_down.weight` |
| `mixer.suv_gate` | [7168] | BF16 | `ffn_gate_scale.weight` |
| `mixer.suv_up` | [7168] | BF16 | `ffn_up_scale.weight` |

GGUF dims are reversed relative to HF (ggml order: `[in, out]`), e.g. `attn_q.weight` is [2560, 4096] in GGUF.
Tensor count check: 3 global + 36 x 8 + 36 x 6 = 507.

## Grouping
- input block: layers 0-15 (attn 0,2,..,14; MLP 1,3,..,15)
- recurrent block, shared across loops: layers 16-55 (attn 16,18,..,54; MLP 17,19,..,55)
- output block: layers 56-71 (attn 56,58,..,70; MLP 57,59,..,71)
Every layer in a block has exactly the per-layer tensor set above; shapes are identical across layers, only values differ.
The GGUF `blk.N` index equals the HF layer index.

## Quantization notes
1-D scale tensors (`residual_alpha`, `attn_qk_scale`, `attn_gate_scale`, `ffn_gate_scale`, `ffn_up_scale`, `output_scale`)
are kept at F32 (BF16 in the bf16 GGUF) by `tensor_force_quant`, not quantized to Q8_0.
