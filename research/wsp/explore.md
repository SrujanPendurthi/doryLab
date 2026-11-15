
## x-randdir-s0 (random-direction rank-1 control, screen only)

```json
{
 "base": {
  "acc": 0.5833333333333334,
  "n": 60,
  "mean_tokens": 226.71666666666667,
  "trunc_rate": 0.11666666666666667,
  "per_source": {
   "gsm8k": {
    "acc": 0.85,
    "n": 20,
    "mean_tokens": 274.15
   },
   "mbpp": {
    "acc": 0.5,
    "n": 20,
    "mean_tokens": 80.5
   },
   "mmlu_pro": {
    "acc": 0.4,
    "n": 20,
    "mean_tokens": 325.5
   }
  }
 },
 "edit": {
  "acc": 0.5833333333333334,
  "n": 60,
  "mean_tokens": 222.93333333333334,
  "trunc_rate": 0.1,
  "per_source": {
   "gsm8k": {
    "acc": 0.85,
    "n": 20,
    "mean_tokens": 276.55
   },
   "mbpp": {
    "acc": 0.5,
    "n": 20,
    "mean_tokens": 82.3
   },
   "mmlu_pro": {
    "acc": 0.4,
    "n": 20,
    "mean_tokens": 309.95
   }
  }
 },
 "tokens_ratio": 0.9833125045945748
}
```
- wall-clock: 342s

## x-randdir-s1 (random-direction rank-1 control, screen only)

```json
{
 "base": {
  "acc": 0.5833333333333334,
  "n": 60,
  "mean_tokens": 226.71666666666667,
  "trunc_rate": 0.11666666666666667,
  "per_source": {
   "gsm8k": {
    "acc": 0.85,
    "n": 20,
    "mean_tokens": 274.15
   },
   "mbpp": {
    "acc": 0.5,
    "n": 20,
    "mean_tokens": 80.5
   },
   "mmlu_pro": {
    "acc": 0.4,
    "n": 20,
    "mean_tokens": 325.5
   }
  }
 },
 "edit": {
  "acc": 0.5666666666666667,
  "n": 60,
  "mean_tokens": 243.75,
  "trunc_rate": 0.13333333333333333,
  "per_source": {
   "gsm8k": {
    "acc": 0.8,
    "n": 20,
    "mean_tokens": 301.6
   },
   "mbpp": {
    "acc": 0.55,
    "n": 20,
    "mean_tokens": 82.1
   },
   "mmlu_pro": {
    "acc": 0.35,
    "n": 20,
    "mean_tokens": 347.55
   }
  }
 },
 "tokens_ratio": 1.075130485922223
}
```
- wall-clock: 360s

## e001-x (exploratory, screen-rejected; not eligible)

```json
{
 "edit_vs_base": {
  "dacc_pts": -0.33333333333333337,
  "ci_pts": [
   -4.0,
   3.3333333333333335
  ],
  "tokens_ratio": 1.0707532784385483,
  "per_source_dacc_pts": {
   "gsm8k": 2.0,
   "mbpp": -2.0,
   "mmlu_pro": -1.0
  }
 },
 "ctrl_vs_base": {
  "dacc_pts": -1.3333333333333335,
  "ci_pts": [
   -3.6666666666666665,
   1.0
  ],
  "tokens_ratio": 1.0036015626134565,
  "per_source_dacc_pts": {
   "gsm8k": 1.0,
   "mbpp": 0.0,
   "mmlu_pro": -5.0
  }
 },
 "arc_dacc_pts": -2.0,
 "holdout_edit": {
  "acc": 0.5733333333333334,
  "n": 300,
  "mean_tokens": 245.77,
  "trunc_rate": 0.14333333333333334,
  "per_source": {
   "gsm8k": {
    "acc": 0.86,
    "n": 100,
    "mean_tokens": 274.17
   },
   "mbpp": {
    "acc": 0.43,
    "n": 100,
    "mean_tokens": 101.16
   },
   "mmlu_pro": {
    "acc": 0.43,
    "n": 100,
    "mean_tokens": 361.98
   }
  }
 },
 "holdout_base": {
  "acc": 0.5766666666666667,
  "n": 300,
  "mean_tokens": 229.53,
  "trunc_rate": 0.13666666666666666,
  "per_source": {
   "gsm8k": {
    "acc": 0.84,
    "n": 100,
    "mean_tokens": 269.68
   },
   "mbpp": {
    "acc": 0.45,
    "n": 100,
    "mean_tokens": 108.22
   },
   "mmlu_pro": {
    "acc": 0.44,
    "n": 100,
    "mean_tokens": 310.69
   }
  }
 },
 "would_accept": false,
 "rule": "reject: no gain (needs dacc>=+2 with CI>0, or dacc>=0 with tokens<=0.90x)"
}
```
- wall-clock: 3882s
