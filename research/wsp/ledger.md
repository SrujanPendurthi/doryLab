
## e000-noop

- time: 2026-11-14T11:23:23+00:00
- hypothesis: dry run
- evidence: []
- predicted: None
- realized: {"n_requested": 0, "n_changed": 0, "tensors": [], "gguf_tensors_differing": 0}
- screen: {"best": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.71666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 274.15}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}, "edit": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.46666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 273.4}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}}
- edit_vs_best: {"dacc_pts": 0.0, "ci_pts": [0.0, 0.0], "tokens_ratio": 1.001045614952294, "per_source_dacc_pts": {"gsm8k": 0.0, "mbpp": 0.0, "mmlu_pro": 0.0}}
- ctrl_vs_best: {"dacc_pts": 0.0, "ci_pts": [0.0, 0.0], "tokens_ratio": 1.0023526336426611, "per_source_dacc_pts": {"gsm8k": 0.0, "mbpp": 0.0, "mmlu_pro": 0.0}}
- arc_dacc_pts: 0.0
- decision: REJECT (reject: no gain (needs dacc>=+2 with CI>0, or dacc>=0 with tokens<=0.90x))
- wall-clock: 4632.1s
- cumulative: 1.287h
<!-- hours: 1.286683 -->

## e001

- time: 2026-11-14T13:03:04+00:00
- hypothesis: Removing the MMLU truncated-minus-finished residual direction from recurrent MLP writes (layers 17-55, alpha 1) shortens overlong reasoning
- evidence: ["research/wsp/corpus/05-circuits.md", "results/wsp/corpus/precheck_trunc_17-55.json"]
- predicted: MMLU truncation rate falls and MMLU accuracy rises; gsm8k/mbpp within noise
- realized: {"n_requested": 367001600, "n_changed": 260864507, "tensors": ["backbone.layers.17.mixer.down_proj.weight", "backbone.layers.19.mixer.down_proj.weight", "backbone.layers.21.mixer.down_proj.weight", "backbone.layers.23.mixer.down_proj.weight", "backbone.layers.25.mixer.down_proj.weight", "backbone.layers.27.mixer.down_proj.weight", "backbone.layers.29.mixer.down_proj.weight", "backbone.layers.31.mixer.down_proj.weight", "backbone.layers.33.mixer.down_proj.weight", "backbone.layers.35.mixer.down_proj.weight", "backbone.layers.37.mixer.down_proj.weight", "backbone.layers.39.mixer.down_proj.weight", "backbone.layers.41.mixer.down_proj.weight", "backbone.layers.43.mixer.down_proj.weight", "backbone.layers.45.mixer.down_proj.weight", "backbone.layers.47.mixer.down_proj.weight", "backbone.layers.49.mixer.down_proj.weight", "backbone.layers.51.mixer.down_proj.weight", "backbone.layers.53.mixer.down_proj.weight", "backbone.layers.55.mixer.down_proj.weight"], "gguf_tensors_differing": 20}
- screen: {"best": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.71666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 274.15}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}, "edit": {"acc": 0.6333333333333333, "n": 60, "mean_tokens": 244.1, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.95, "n": 20, "mean_tokens": 270.3}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 105.35}, "mmlu_pro": {"acc": 0.45, "n": 20, "mean_tokens": 356.65}}}}
- decision: REJECT (reject: screen)
- wall-clock: 351.7s
- cumulative: 1.384h
<!-- hours: 0.097682 -->

## e002

- time: 2026-11-14T14:10:02+00:00
- hypothesis: Removing the MMLU truncated-minus-finished residual direction from recurrent MLP writes (layers 17-55, alpha 0.5) shortens overlong reasoning
- evidence: ["research/wsp/corpus/05-circuits.md", "results/wsp/corpus/precheck_trunc_17-55.json"]
- predicted: MMLU truncation rate falls and MMLU accuracy rises; gsm8k/mbpp within noise
- realized: {"n_requested": 367001600, "n_changed": 210298804, "tensors": ["backbone.layers.17.mixer.down_proj.weight", "backbone.layers.19.mixer.down_proj.weight", "backbone.layers.21.mixer.down_proj.weight", "backbone.layers.23.mixer.down_proj.weight", "backbone.layers.25.mixer.down_proj.weight", "backbone.layers.27.mixer.down_proj.weight", "backbone.layers.29.mixer.down_proj.weight", "backbone.layers.31.mixer.down_proj.weight", "backbone.layers.33.mixer.down_proj.weight", "backbone.layers.35.mixer.down_proj.weight", "backbone.layers.37.mixer.down_proj.weight", "backbone.layers.39.mixer.down_proj.weight", "backbone.layers.41.mixer.down_proj.weight", "backbone.layers.43.mixer.down_proj.weight", "backbone.layers.45.mixer.down_proj.weight", "backbone.layers.47.mixer.down_proj.weight", "backbone.layers.49.mixer.down_proj.weight", "backbone.layers.51.mixer.down_proj.weight", "backbone.layers.53.mixer.down_proj.weight", "backbone.layers.55.mixer.down_proj.weight"], "gguf_tensors_differing": 20}
- screen: {"best": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.71666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 274.15}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}, "edit": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.88333333333333, "trunc_rate": 0.1, "per_source": {"gsm8k": {"acc": 0.8, "n": 20, "mean_tokens": 267.2}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 82.95}, "mmlu_pro": {"acc": 0.45, "n": 20, "mean_tokens": 330.5}}}}
- edit_vs_best: {"dacc_pts": 1.0, "ci_pts": [-2.3333333333333335, 4.333333333333334], "tokens_ratio": 1.036451299031354, "per_source_dacc_pts": {"gsm8k": -1.0, "mbpp": 0.0, "mmlu_pro": 4.0}}
- ctrl_vs_best: {"dacc_pts": -0.6666666666666667, "ci_pts": [-2.666666666666667, 1.3333333333333335], "tokens_ratio": 0.9955270915929653, "per_source_dacc_pts": {"gsm8k": 0.0, "mbpp": 1.0, "mmlu_pro": -3.0}}
- arc_dacc_pts: -0.5
- decision: REJECT (reject: no gain (needs dacc>=+2 with CI>0, or dacc>=0 with tokens<=0.90x))
- wall-clock: 4015.4s
- cumulative: 2.500h
<!-- hours: 1.115398 -->

## e003

- time: 2026-11-14T14:16:24+00:00
- hypothesis: Removing the MMLU truncated-minus-finished residual direction from recurrent MLP writes (layers 17-35, alpha 1) shortens overlong reasoning
- evidence: ["research/wsp/corpus/05-circuits.md", "results/wsp/corpus/precheck_trunc_17-55.json"]
- predicted: MMLU truncation rate falls and MMLU accuracy rises; gsm8k/mbpp within noise
- realized: {"n_requested": 183500800, "n_changed": 133367326, "tensors": ["backbone.layers.17.mixer.down_proj.weight", "backbone.layers.19.mixer.down_proj.weight", "backbone.layers.21.mixer.down_proj.weight", "backbone.layers.23.mixer.down_proj.weight", "backbone.layers.25.mixer.down_proj.weight", "backbone.layers.27.mixer.down_proj.weight", "backbone.layers.29.mixer.down_proj.weight", "backbone.layers.31.mixer.down_proj.weight", "backbone.layers.33.mixer.down_proj.weight", "backbone.layers.35.mixer.down_proj.weight"], "gguf_tensors_differing": 10}
- screen: {"best": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.71666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 274.15}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}, "edit": {"acc": 0.6, "n": 60, "mean_tokens": 242.98333333333332, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.8, "n": 20, "mean_tokens": 271.0}, "mbpp": {"acc": 0.55, "n": 20, "mean_tokens": 109.55}, "mmlu_pro": {"acc": 0.45, "n": 20, "mean_tokens": 348.4}}}}
- decision: REJECT (reject: screen)
- wall-clock: 379.4s
- cumulative: 2.605h
<!-- hours: 0.105399 -->

## e004

- time: 2026-11-14T14:22:50+00:00
- hypothesis: Removing the MMLU truncated-minus-finished residual direction from recurrent MLP writes (layers 37-55, alpha 1) shortens overlong reasoning
- evidence: ["research/wsp/corpus/05-circuits.md", "results/wsp/corpus/precheck_trunc_17-55.json"]
- predicted: MMLU truncation rate falls and MMLU accuracy rises; gsm8k/mbpp within noise
- realized: {"n_requested": 183500800, "n_changed": 127497181, "tensors": ["backbone.layers.37.mixer.down_proj.weight", "backbone.layers.39.mixer.down_proj.weight", "backbone.layers.41.mixer.down_proj.weight", "backbone.layers.43.mixer.down_proj.weight", "backbone.layers.45.mixer.down_proj.weight", "backbone.layers.47.mixer.down_proj.weight", "backbone.layers.49.mixer.down_proj.weight", "backbone.layers.51.mixer.down_proj.weight", "backbone.layers.53.mixer.down_proj.weight", "backbone.layers.55.mixer.down_proj.weight"], "gguf_tensors_differing": 10}
- screen: {"best": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.71666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 274.15}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}, "edit": {"acc": 0.6, "n": 60, "mean_tokens": 239.73333333333332, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.8, "n": 20, "mean_tokens": 290.1}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 106.7}, "mmlu_pro": {"acc": 0.5, "n": 20, "mean_tokens": 322.4}}}}
- decision: REJECT (reject: screen)
- wall-clock: 382.5s
- cumulative: 2.711h
<!-- hours: 0.106238 -->

## e005

- time: 2026-11-14T14:29:20+00:00
- hypothesis: Amplifying the commit-position direction in recurrent MLP writes (layers 17-55, alpha -0.5) makes the model commit to an answer sooner
- evidence: ["research/wsp/corpus/05-circuits.md", "results/wsp/corpus/precheck_commit_17-55.json"]
- predicted: MMLU truncation rate falls and MMLU accuracy rises; gsm8k/mbpp within noise
- realized: {"n_requested": 367001600, "n_changed": 230546142, "tensors": ["backbone.layers.17.mixer.down_proj.weight", "backbone.layers.19.mixer.down_proj.weight", "backbone.layers.21.mixer.down_proj.weight", "backbone.layers.23.mixer.down_proj.weight", "backbone.layers.25.mixer.down_proj.weight", "backbone.layers.27.mixer.down_proj.weight", "backbone.layers.29.mixer.down_proj.weight", "backbone.layers.31.mixer.down_proj.weight", "backbone.layers.33.mixer.down_proj.weight", "backbone.layers.35.mixer.down_proj.weight", "backbone.layers.37.mixer.down_proj.weight", "backbone.layers.39.mixer.down_proj.weight", "backbone.layers.41.mixer.down_proj.weight", "backbone.layers.43.mixer.down_proj.weight", "backbone.layers.45.mixer.down_proj.weight", "backbone.layers.47.mixer.down_proj.weight", "backbone.layers.49.mixer.down_proj.weight", "backbone.layers.51.mixer.down_proj.weight", "backbone.layers.53.mixer.down_proj.weight", "backbone.layers.55.mixer.down_proj.weight"], "gguf_tensors_differing": 20}
- screen: {"best": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.71666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 274.15}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}, "edit": {"acc": 0.6166666666666667, "n": 60, "mean_tokens": 240.91666666666666, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.9, "n": 20, "mean_tokens": 246.4}, "mbpp": {"acc": 0.6, "n": 20, "mean_tokens": 141.75}, "mmlu_pro": {"acc": 0.35, "n": 20, "mean_tokens": 334.6}}}}
- decision: REJECT (reject: screen)
- wall-clock: 387.7s
- cumulative: 2.819h
<!-- hours: 0.107693 -->

## e006

- time: 2026-11-14T15:38:09+00:00
- hypothesis: Removing the MMLU truncated-minus-finished residual direction from recurrent MLP writes (layers 17-55, alpha 0.75) shortens overlong reasoning
- evidence: ["research/wsp/corpus/05-circuits.md", "results/wsp/corpus/precheck_trunc_17-55.json"]
- predicted: MMLU truncation rate falls and MMLU accuracy rises; gsm8k/mbpp within noise
- realized: {"n_requested": 367001600, "n_changed": 241180888, "tensors": ["backbone.layers.17.mixer.down_proj.weight", "backbone.layers.19.mixer.down_proj.weight", "backbone.layers.21.mixer.down_proj.weight", "backbone.layers.23.mixer.down_proj.weight", "backbone.layers.25.mixer.down_proj.weight", "backbone.layers.27.mixer.down_proj.weight", "backbone.layers.29.mixer.down_proj.weight", "backbone.layers.31.mixer.down_proj.weight", "backbone.layers.33.mixer.down_proj.weight", "backbone.layers.35.mixer.down_proj.weight", "backbone.layers.37.mixer.down_proj.weight", "backbone.layers.39.mixer.down_proj.weight", "backbone.layers.41.mixer.down_proj.weight", "backbone.layers.43.mixer.down_proj.weight", "backbone.layers.45.mixer.down_proj.weight", "backbone.layers.47.mixer.down_proj.weight", "backbone.layers.49.mixer.down_proj.weight", "backbone.layers.51.mixer.down_proj.weight", "backbone.layers.53.mixer.down_proj.weight", "backbone.layers.55.mixer.down_proj.weight"], "gguf_tensors_differing": 20}
- screen: {"best": {"acc": 0.5833333333333334, "n": 60, "mean_tokens": 226.71666666666667, "trunc_rate": 0.11666666666666667, "per_source": {"gsm8k": {"acc": 0.85, "n": 20, "mean_tokens": 274.15}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 80.5}, "mmlu_pro": {"acc": 0.4, "n": 20, "mean_tokens": 325.5}}}, "edit": {"acc": 0.6166666666666667, "n": 60, "mean_tokens": 228.05, "trunc_rate": 0.1, "per_source": {"gsm8k": {"acc": 0.9, "n": 20, "mean_tokens": 264.95}, "mbpp": {"acc": 0.5, "n": 20, "mean_tokens": 81.85}, "mmlu_pro": {"acc": 0.45, "n": 20, "mean_tokens": 337.35}}}}
- edit_vs_best: {"dacc_pts": -0.6666666666666667, "ci_pts": [-4.0, 2.666666666666667], "tokens_ratio": 1.0463120289286802, "per_source_dacc_pts": {"gsm8k": 0.0, "mbpp": -1.0, "mmlu_pro": -1.0}}
- ctrl_vs_best: {"dacc_pts": -1.3333333333333335, "ci_pts": [-4.0, 1.3333333333333335], "tokens_ratio": 1.018617755122787, "per_source_dacc_pts": {"gsm8k": 0.0, "mbpp": 1.0, "mmlu_pro": -5.0}}
- arc_dacc_pts: -0.5
- decision: REJECT (reject: no gain (needs dacc>=+2 with CI>0, or dacc>=0 with tokens<=0.90x))
- wall-clock: 4125.7s
- cumulative: 3.965h
<!-- hours: 1.146028 -->
