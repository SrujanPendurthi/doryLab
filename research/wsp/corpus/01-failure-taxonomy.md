# Failure taxonomy (truncated research rows)

Evidence: `results/wsp/corpus/taxonomy.json`. Rules in order: loop = some 8-gram of whitespace tokens occurs >= 3 times; overlong = no loop and no answer phrase (`answer is`; mbpp: closing fence); other.

| source | loop | overlong | other | rows total |
|---|---|---|---|---|
| gsm8k | 0 | 8 | 0 | 100 |
| mbpp | 0 | 2 | 0 | 100 |
| mmlu_pro | 1 | 34 | 2 | 100 |

## Loop rows

| id | 8-gram | first char offset |
|---|---|---|
| mmlu_pro:test:8186 | `Check $3^2$: $3^2 = 9$. Since $9$ is` | 651 |

## Excerpts: loop

`mmlu_pro:test:8186`:

```
, this is not powerful.

**Option E: 336**
$336 = 2^4 \times 3 \times 7$
Prime factors: 2, 3, and 7.
Check $3^2$: $3^2 = 9$. Since $9$ is not a factor of $336$, this is not powerful.

**Option F: 360**
$360 = 2^3 \times 3^2 \times 5$
Prime factors: 2, 3, and 5.
Check $5^2$: $5^2 = 25$. Since $25$ is
```

## Excerpts: overlong

`gsm8k:train:3835`:

```
litter: $8 \times 6 = 48$ pups.
    3. Total mice after first litter: $8 + 48 = 56$.
    4. Second litter: $56 \times 6 = 336$ pups.
    5. Eating: Each of the 56 adults eats 2 pups.
    6. Pups eaten: $56 \times 2 = 112$.
    7. Pups remaining: $336 - 112 = 224$.
    8. Total mice left: $56$ (adult
```

`gsm8k:train:4508`:

```
ave:
$D = 200$

**Step 5: Verify the result**
Initial dislikes: 200
Initial likes: 200
Check: Is 200 100 more than half as many likes?
Half of 200 is 100. 100 + 100 = 200. This matches.
After change:
Likes: 200 + 0 = 200
Dislikes: 200 + 1000 = 1200
Check: Is 1200 1000 more than the likes?
1200 - 200
```

`gsm8k:train:405`:

```
eet) + (6 gaps × 2 feet) + (2 feet left + 2 feet right)
- 7 × 3 = 21 feet
- 6 × 2 = 12 feet
- 21 + 12 + 4 = 37 feet.
37 feet is less than 42 feet, so 7 boats fit.
- 8 boats would require: (8 boats × 3 feet) + (7 gaps × 2 feet) + (2 feet left + 2 feet right)
- 8 × 3 = 24 feet
- 7 × 2 = 14 feet
- 24 +
```

## Excerpts: other

`mmlu_pro:test:1056`:

```
admissibility here relates to the integrity of the evidence (chain of custody) rather than a general lack of accuracy.
    *   **D:** The timing (three months later) is a fact of the case but does not inherently make the photos inadmissible if they are still authentic.
    *   **F:** Being developed
```

`mmlu_pro:test:1830`:

```
.
    *   **C:** Incorrect. The insufficient funds suggest a lack of funds, not necessarily fraudulent intent.
    *   **D:** Incorrect. The parol evidence rule relates to the interpretation of the contract, not the admissibility of evidence regarding the performance of the consideration.
    *   **
```
