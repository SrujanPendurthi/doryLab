# Logit lens at the final-answer token

Evidence: `results/wsp/corpus/lens.npz` (rank per virtual layer, key `id|L<loops>`; rank 0 = top-1). Teacher-forced prompt + own output (finished rows only); first_top1 = -1 means never.

| source | loops | items | reach top-1 | median first top-1 layer | n virtual layers |
|---|---|---|---|---|---|
| gsm8k | 1 | 10 | 10 | 56.000 | 72 |
| gsm8k | 2 | 10 | 10 | 96.000 | 112 |
| gsm8k | 4 | 10 | 10 | 176.000 | 192 |
| mbpp | 1 | 10 | 10 | 38.000 | 72 |
| mbpp | 2 | 10 | 10 | 38.000 | 112 |
| mbpp | 4 | 10 | 10 | 38.000 | 192 |
| mmlu_pro | 1 | 10 | 10 | 60.500 | 72 |
| mmlu_pro | 2 | 10 | 10 | 104.000 | 112 |
| mmlu_pro | 4 | 10 | 10 | 178.000 | 192 |

## Fraction already top-1 at each loop boundary (last recurrent half-block of loop c)

| loops | after loop | virtual layer | fraction top-1 |
|---|---|---|---|
| 1 | 1 | 55 | 0.000 |
| 2 | 1 | 55 | 0.000 |
| 2 | 2 | 95 | 0.000 |
| 4 | 1 | 55 | 0.000 |
| 4 | 2 | 95 | 0.000 |
| 4 | 3 | 135 | 0.000 |
| 4 | 4 | 175 | 0.000 |

## Where the answer first reaches top-1

| source | loops | frac first top-1 in output blocks | frac first top-1 before output blocks |
|---|---|---|---|
| gsm8k | 1 | 0.700 | 0.300 |
| gsm8k | 2 | 0.600 | 0.400 |
| gsm8k | 4 | 0.600 | 0.400 |
| mbpp | 1 | 0.000 | 1.000 |
| mbpp | 2 | 0.000 | 1.000 |
| mbpp | 4 | 0.000 | 1.000 |
| mmlu_pro | 1 | 1.000 | 0.000 |
| mmlu_pro | 2 | 1.000 | 0.000 |
| mmlu_pro | 4 | 1.000 | 0.000 |
