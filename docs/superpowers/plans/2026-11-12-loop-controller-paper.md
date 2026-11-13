# Loop-Controller Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An arXiv-ready paper on per-request loop-count control for Dory1.0-4B on an M4 laptop, built only from measured results, with every number generated from the results files.

**Architecture:** Analysis scripts read `results/*.jsonl|json` and emit LaTeX macros, tables and figures into `paper/generated/` and `paper/figures/`. `paper/main.tex` uses only those macros for numbers, so it can't drift from the data. Tectonic builds the PDF. A claims audit checks every sentence-level claim against its evidence before hand-off.

**Tech Stack:** Python 3.13 `.venv` (numpy, matplotlib added), the existing `dory/` package (`labels`, `evaluate`, `evalsets`), LaTeX via `/opt/homebrew/bin/tectonic`.

**Spec:** None written separately. The scope below was set by the controller from `results/decisions.md`, the loop-controller plan (`docs/superpowers/plans/2026-11-11-dory-loop-controller.md`) and the research report (`reports/Dory loop count controller.md`). Treat it as the binding scope.

## Paper Scope

Working title: **"How Many Loops? Per-Request Compute Allocation for a Looped LLM on a Laptop."**

Claims, in order of weight (each must be backed by a figure or table):
- **C1, flat fixed-k accuracy with large per-item headroom.** Holdout fixed k=1/2/4 = 0.600/0.587/0.610; the per-prompt oracle reaches 0.670 at mean k 1.11 and fewer modeled seconds than fixed k=1 (11.2 vs 10.4 s; fixed 2 is 18.5 s).
- **C2, learned routers capture none of that headroom.** Text-feature policy +0.7 pt vs fixed2 (CI −1.0 to +2.7); hidden-state probe +0.3 pt. Final split (100): text 0.630 vs chosen 0.620, noise.
- **C3, a two-rule heuristic matches fixed k=2 at ~62% of the latency.** Fixed k=1 plus a code floor (k=2 for code): 0.620 at 12.5 s vs 0.620 at 20.3 s on final. It was found after seeing holdout, so it is presented as post hoc.
- **C4, a truncation confound.** About a third of MMLU-Pro answers hit the 512-token cap at every k, so MMLU labels largely measure whether an answer fits. New analysis: accuracy and k-effects among non-truncated items.
- **C5, systems results on Apple silicon.**
  - Self-speculative decoding with a k=1 draft of the same weights (GGUF metadata clone): 1.21x, outputs identical, acceptance 0.84.
  - n-gram lookup 0.74x; partial GPU offload 0.94x; q8_0 KV −1.5 pts.
  - Classifier on the Neural Engine 17.9 vs 44.3 ms.
- **C6, a measurement lesson.** On a shared laptop, sequential A/B arms gave a spurious 0.29x speculative "slowdown". Interleaved arms gave 1.21x.

Non-goals: new controller methods; claims about BF16 or sampled decoding; per-token control.

## Global Constraints

- Every number in `paper/main.tex` comes from a macro in `paper/generated/numbers.tex`; no hand-typed numerals for results (section numbers, years, model sizes in prose excepted).
- Sources of truth: `results/labels_{train,holdout,final}.jsonl`, `results/sweep_*.jsonl`, `results/speed.json`, `results/spec_ab.jsonl`, `results/draft_sweep.jsonl`, `results/spec_k4.json`, `results/offload_ab.jsonl`, `results/sweep_holdout_q8.jsonl`, `results/ane_bench.json`, `results/live_gate.json`, `results/bench.csv`. Where a value is only in `results/decisions.md`, recompute it from these files; if it can't be recomputed, cite decisions.md in a comment beside the macro.
- Report final-split numbers from the post-regrade rerun and disclose that the final split was evaluated twice (before and after a grader fix, Ruling AJ), along with the pre-regrade values.
- Disclose: Q5_K_M weights, greedy decoding, thinking disabled (the card recommends temperature 1.0 and reasoning on), one machine (M4, 24 GB), holdout-informed heuristic (C3), unmerged llama.cpp PR build (pinned commit), MMLU truncation.
- No GPU runs in Tasks 1-5 (the GPU belongs to the weight-space work). Task 6 is the only GPU task and is optional.
- Work on branch `paper` from `main` in a worktree `.worktrees/paper`. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A macro silently computed on the wrong split or the wrong row version (stale pre-regrade rows): the numbers test pins headline macros to values recorded in decisions.md.
- Modeled seconds computed with a different speed table than the controller results used: reuse `dory.evaluate.modeled_seconds` and `results/speed.json`, never re-derive.
- A figure that drops error rows or duplicate (retried) rows differently from the label builder: use `dory.sweep.read_rows`/`latest_rows` semantics.
- Overclaiming from CIs that include zero (C2) or from post hoc choices (C3): the claims audit checks wording against CIs.
- Citations that don't resolve or don't say what the paper says: the citation check fetches each one.

---

### Task 1: Numbers and tables generator

**Files:**
- Create: `scripts/paper/numbers.py`, `paper/generated/` (output), `tests/test_paper_numbers.py`
- Modify: `.venv` (`pip install matplotlib` — used in Task 2)

**Interfaces:**
- Consumes: `dory.evaluate` (`policy_metrics`, `fixed`, `oracle`, `paired_bootstrap`, `modeled_seconds`), `dory.sweep.read_rows`, results files above.
- Produces: `compute() -> dict[str, float | str]` (macro name → value) and `write(out_dir: Path)`, which writes `numbers.tex` (`\newcommand{\NAME}{value}`, names in CamelCase letters only, e.g. `\HoldoutFixedOneAcc`), `numbers.json`, and table files `tab_controllers_holdout.tex`, `tab_controllers_final.tex`, `tab_per_source.tex`, `tab_systems.tex`, `tab_truncation.tex`.
- New analysis inside `compute()`: per source × k, truncation rate and accuracy among non-truncated items (truncated = `n_gen >= MAX_TOKENS[source]`), on holdout and train; and the oracle split into "k=1 already right" vs "rescued by k=2/4".

- [ ] **Step 1: Write failing test** `test_headline_numbers_match_decisions`: `compute()` gives holdout fixed1 acc 0.600, fixed2 0.587, fixed4 0.610, oracle 0.670 (±0.0005); final chosen 0.620, fixed1+floor 0.620 with mean seconds 12.51 (±0.01); spec n4 geometric speedup 1.21 (±0.01); ANE median 17.9 ms (±0.1). Plus `test_macro_names_valid`: every key matches `^[A-Za-z]+$`.
- [ ] **Step 2:** Run `pytest tests/test_paper_numbers.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS; `python -m scripts.paper.numbers` writes `paper/generated/`.
- [ ] **Step 5:** Commit (generated files included, so the paper builds without rerunning).

### Task 2: Figures

**Files:**
- Create: `scripts/paper/figures.py`, `paper/figures/*.pdf`

**Interfaces:**
- Consumes: `paper/generated/numbers.json` plus the results files.
- Produces: `fig_pareto.pdf` (accuracy vs mean modeled seconds: fixed1/2/4, fixed1+floor, text, probe, oracle; holdout and final as two panels), `fig_headroom.pdf` (per-source stacked bars: right at k=1, rescued by k=2/4, wrong at all), `fig_truncation.pdf` (truncation rate and non-truncated accuracy per source × k), `fig_spec.pdf` (left: draft n_max sweep tok/s; right: per-prompt interleaved speedup ratios for n4/n6, with the sequential 0.29x marked).

- [ ] **Step 1:** Implement, following the dataviz skill's palette and mark rules; vector PDF, colorblind-safe, labeled axes with units.
- [ ] **Step 2: Verify:** the script runs end to end, and a smoke test `tests/test_paper_figures.py::test_figures_written` asserts the four PDFs exist and are non-empty after `make_all(tmp_path)`.
- [ ] **Step 3:** Commit.

### Task 3: Bibliography with verified citations

**Files:**
- Create: `paper/refs.bib`, `paper/generated/citation_check.md`

**Interfaces:**
- Consumes: sources in `research_notes/Dory loop count controller/` and `reports/Dory loop count controller.md` (Dory model card and arXiv paper, Ouro, Huginn, Continuous Depth Batching, nGPT, llama.cpp PR #30306, vLLM PR #60149, NVIDIA prompt-task-and-complexity classifier, GSM8K, MMLU-Pro, MBPP, speculative decoding (Leviathan et al.), LayerSkip/self-speculative work, Laya).
- Produces: one BibTeX entry per source actually cited; `citation_check.md` with, per key: URL fetched, resolves (yes/no), the one claim the paper uses it for, and whether the source supports it.

- [ ] **Step 1:** Fetch each source; keep only entries that resolve and support the claim; mark any unsupported claim for Task 4 to drop or rephrase.
- [ ] **Step 2:** Commit.

### Task 4: Paper draft

**Files:**
- Create: `paper/main.tex`, `paper/Makefile` (`tectonic main.tex`)

**Interfaces:**
- Consumes: `paper/generated/*`, `paper/figures/*`, `paper/refs.bib`, `citation_check.md`, `results/decisions.md` (the rulings list for the appendix).
- Produces: `paper/main.pdf`.

- [ ] **Step 1:** Write sections: Abstract; 1 Introduction (C1-C6 as contributions); 2 Background (looped LMs, Dory's per-sample loop curriculum, per-request vs per-token control); 3 Setup (hardware, runtime and pinned build, weights, decoding, datasets and splits, graders and truncation rule, modeled latency from `speed.json`); 4 Controllers and protocol (labels, text and probe policies, gates fixed before results, the one-shot final split and its later rerun); 5 Results (C1-C3); 6 The truncation confound (C4); 7 Systems on Apple silicon (C5); 8 Measuring on a shared laptop (C6); 9 Limitations; 10 Related work; 11 Conclusion. Appendix A: every recorded ruling with its rationale; Appendix B: full tables.
- [ ] **Step 2: Verify:** `cd paper && tectonic main.tex` builds with no undefined references or citations, and `grep -nE '[0-9]+\.[0-9]+' paper/main.tex` shows no result numerals outside macro definitions, figure files and the bibliography.
- [ ] **Step 3:** Commit.

### Task 5: Claims audit

**Files:**
- Create: `paper/generated/claims_audit.md`

- [ ] **Step 1:** A fresh reviewer (most capable model) reads `main.pdf`'s source and, for each claim-bearing sentence, records the macro, table or figure backing it and whether the wording fits the evidence. That means no "improves" when a CI includes zero, post hoc labeled as such, and the truncation caveat attached to every MMLU-derived claim.
- [ ] **Step 2:** Fix every flagged sentence in one pass, rebuild, commit.

### Task 6 (optional, GPU): MMLU-Pro at 1024 tokens

Runs only if the user approves it and the GPU is free of weight-space work.

**Files:**
- Create: `results/sweep_holdout_mmlu1024.jsonl`, a `tab_mmlu1024.tex` addition in Task 1's generator

- [ ] **Step 1:** Holdout mmlu_pro (100 items) at k=1, 2 and 4 with `max_tokens` 1024 (≈3 GPU-hours), via `dory.sweep.run_sweep` with variant `mmlu1024` and a new optional parameter `run_sweep(..., max_tokens: dict[str, int] | None = None)` (default `MAX_TOKENS`; used for both the request cap and the truncation flag passed to `grade`), with a unit test that the override reaches `complete` and `grade`; record truncation and accuracy per k.
- [ ] **Step 2:** Add C4b to the paper: whether loops help once answers fit. Rebuild, re-audit the changed sentences, commit.
