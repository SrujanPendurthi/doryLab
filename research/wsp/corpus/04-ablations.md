# Component ablations (teacher-forced logprob of own continuation, loops=1, mode zero)

Evidence: `results/wsp/corpus/ablations.json`. 20 truncated (loop/overlong) vs 20 finished-correct rows; effect = mean change in logprob; differential = truncated minus finished. Note: head/neuron "zero" is renormalized by nGPT's L2 norm and is not comparable to layer skip (Task 6 review). Neurons are ranked by mlp_act before ffn_sub_norm while the ablation zeroes after it, so the magnitude ranking is approximate.

## Top 20 by |differential|

| component | d logprob truncated | d logprob finished | differential |
|---|---|---|---|
| layer.16 | -0.655 | -0.472 | -0.183 |
| layer.34 | -1.413 | -1.284 | -0.129 |
| layer.0 | -0.496 | -0.601 | 0.105 |
| layer.1 | -0.246 | -0.330 | 0.084 |
| layer.36 | -0.348 | -0.408 | 0.060 |
| layer.2 | -0.135 | -0.191 | 0.056 |
| layer.56 | -2.713 | -2.665 | -0.047 |
| layer.6 | -0.084 | -0.126 | 0.042 |
| layer.62 | -0.157 | -0.196 | 0.039 |
| layer.71 | -0.137 | -0.176 | 0.039 |
| layer.13 | -0.089 | -0.126 | 0.037 |
| layer.30 | -0.153 | -0.189 | 0.036 |
| layer.50 | -0.160 | -0.125 | -0.035 |
| layer.54 | -0.124 | -0.157 | 0.032 |
| layer.4 | -0.108 | -0.140 | 0.032 |
| layer.22 | -0.127 | -0.158 | 0.032 |
| layer.32 | -0.216 | -0.246 | 0.030 |
| layer.19 | -0.180 | -0.210 | 0.029 |
| layer.60 | -0.083 | -0.054 | -0.029 |
| layer.45 | -0.109 | -0.137 | 0.029 |

