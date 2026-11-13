# Weight-Space Programming for Dory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opus writes direct, verified weight edits that raise Dory1.0-4B's k=1 accuracy under the token caps (or cut its tokens at equal accuracy), proven on a sealed test set.

**Architecture:** BF16 safetensors are the editable source of truth. An edit is a JSON patch applied to them, converted to a Q8_0 GGUF and scored with the existing sweep/grade harness. Dory in PyTorch via the open transformers PR #50099 (parity-tested against llama-server) gives activations for a logit lens, probes and ablations; their outputs form a research corpus from which the controller (Opus) authors edits. Each edit runs through screen, selection, a size-matched random control and an ARC damage check.

**Tech Stack:** Python 3.13 (`.venv-wsp`: system site packages + transformers pinned to PR #50099), torch 2.8 (MPS, bf16), safetensors, numpy, scikit-learn, the vendored llama.cpp (llama-server, `convert_hf_to_gguf.py`, llama-quantize), the existing `dory/` package.

**Spec:** `docs/superpowers/specs/2026-11-12-weight-space-programming-design.md`

## Global Constraints

- Branch `wsp` from `main`. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never delete anything in `~/.cache/huggingface`. Never touch other projects' files. Never change `iogpu.wired_limit_mb`.
- At most one llama-server for evaluation at a time; the PyTorch model (~8 GB) may run alongside one Q8_0 server, never alongside more.
- Evaluation format is Q8_0 GGUF for base and every edit. Old Q5_K_M results are never used as baselines.
- Token caps and graders are the existing ones: `MAX_TOKENS = {"gsm8k": 512, "mmlu_pro": 512, "mbpp": 768}`, `grade(item, output, truncated)`; ARC uses 512 and the mmlu_pro letter grader.
- Primary metric: pooled k=1 accuracy over gsm8k + mmlu_pro + mbpp. Secondary: mean `n_gen`.
- Screen pass: edit correct count ≥ base correct count on `wsp_screen`, and edit mean tokens ≤ 1.05 × base.
- Selection accept (vs current best model, on `holdout`), all of:
  - (Δacc ≥ +2.0 pts and paired-bootstrap 95% CI lower bound > 0) or (Δacc ≥ 0 and mean tokens ≤ 0.90 × current best);
  - edit Δacc minus its random control's Δacc ≥ +2.0 pts, or edit mean tokens ≤ 0.90 × control's if the token path was used;
  - every source's Δacc ≥ −3.0 pts;
  - ARC Δacc ≥ −3.0 pts.
- `wsp_final` loads only when the environment has `WSP_UNSEAL=1`; only Task 9 sets it.
- Mechanism claims from Opus are hypotheses; only harness results decide acceptance. Opus never authors an edit from raw weights alone; each edit file cites corpus evidence.

## Review Focus

- An edit names a tensor or index that doesn't exist (HF name vs GGUF name, recurrent-block index confusion): applying must raise, never silently skip.
- An `add` edit too small to survive bf16 or Q8_0 rounding changes nothing: apply reports changed-parameter count after the bf16 cast, and the GGUF check confirms the dequantized tensor differs from base.
- The sealed final split is loaded during research: `load_split("wsp_final")` raises without `WSP_UNSEAL=1`.
- Cached results reused for a different model or config: result files are keyed by the GGUF's content hash, the split, k and the token caps.
- PyTorch activations that don't match llama-server would poison every probe: no lens, probe or ablation runs until the parity test passes.

---

### Task 1: Weights, Q8_0/BF16 GGUFs, llama-quantize

**Files:**
- Create: `scripts/wsp_get_weights.sh`
- Modify: `scripts/setup_llama.sh` (add `llama-quantize` to build targets)
- Create: `research/wsp/corpus/06-architecture.md`

**Interfaces:**
- Produces: `models/bf16/` (symlinks to the cached `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` snapshot at rev `f6c832e6963ddb74f62c75813a3e8c74b1af67b1`), `models/gguf/dory-base-q8_0.gguf`, `models/gguf/dory-base-bf16.gguf`, `vendor/llama.cpp/build/bin/llama-quantize`.

- [ ] **Step 1:** Abort if free disk < 25 GB. Download the full BF16 snapshot with `huggingface_hub.snapshot_download` at the pinned rev; symlink it as `models/bf16`.
- [ ] **Step 2:** Convert with `vendor/llama.cpp/convert_hf_to_gguf.py models/bf16 --outtype q8_0` and `--outtype bf16` to the two paths above. Build `llama-quantize`.
- [ ] **Step 3: Verify:** `llama-server` with `server_args(models/gguf/dory-base-q8_0.gguf, 1, port, slots=1)` answers "What is 2+2?" containing `4`, at k=1 and k=4.
- [ ] **Step 4:** Write `06-architecture.md`: every safetensors tensor name, shape and dtype, grouped by input (16) / recurrent (40, shared across loops) / output (16) block, plus the HF-name to GGUF-name mapping read from `conversion/dory.py`.
- [ ] **Step 5:** Commit script, setup change and the architecture note.

### Task 2: Evaluation splits (screen, sealed final, ARC)

**Files:**
- Modify: `dory/evalsets.py`
- Test: `tests/test_evalsets.py`

**Interfaces:**
- Consumes: `load`, `load_split`, `SPLITS`, `Item`, `grade`.
- Produces: `load_split("wsp_screen")` (60 items: the first 20 per source of `holdout`, holdout order), `load_split("wsp_final")` (gsm8k test 100, mmlu_pro test 100, mbpp test 100, excluding every id in holdout, final and train), `load_split("arc")` (200 from `allenai/ai2_arc` `ARC-Challenge` test, seed 0), source `"arc"` with `MAX_TOKENS["arc"] = 512`.

- [ ] **Step 1: Write failing tests:**
  - `test_wsp_screen_subset`: 60 items, 20 per source, every id in `holdout`.
  - `test_wsp_final_sealed`: without `WSP_UNSEAL`, `load_split("wsp_final")` raises `PermissionError`; with it set via monkeypatch, 300 items, disjoint from holdout, final and train ids.
  - `test_arc_item_and_grade`: an ARC item's prompt lists options as `A.`–`D.` (numeric labels `1`–`4` mapped to letters), ends with `End with 'The answer is (X)'.`, and `grade(item, "The answer is (B)")` is True when gold is B.
- [ ] **Step 2:** Run `pytest tests/test_evalsets.py -v`. Expected: the three new tests FAIL.
- [ ] **Step 3:** Implement. `wsp_final` and `arc` are not added to `SPLITS` order for existing splits; existing split contents must not change (assert holdout ids unchanged in the test via a fixed id sample).
- [ ] **Step 4:** Run the tests. Expected: PASS, and the full suite still green.
- [ ] **Step 5:** Commit.

### Task 3: Edit patches and random controls

**Files:**
- Create: `dory/wsp/__init__.py`, `dory/wsp/edits.py`
- Test: `tests/test_wsp_edits.py`

**Interfaces:**
- Produces:
  - Edit file format (JSON): `{"id": str, "hypothesis": str, "evidence": [str], "ops": [op]}`; op is one of `{"tensor", "op": "set"|"add"|"scale", "index": [[i, j], ...], "value": [float, ...]}` (scale: value per index or a single float), `{"tensor", "op": "scale_rows"|"scale_cols", "index": [int, ...], "value": float}`, `{"tensor", "op": "lowrank_add", "npz": "path.npz", "u": key, "v": key}` (delta = u @ v.T).
  - `apply_edit(edit: dict, base_dir: Path, out_dir: Path) -> EditStats` with `EditStats(n_requested: int, n_changed: int, tensors: list[str], delta_l2: dict[str, float])`. Untouched shards are APFS clones (`cp -c`); touched shards are rewritten in bf16; config and tokenizer files copied.
  - `random_control(edit: dict, base_dir: Path, seed: int) -> dict`: same tensors; the edit's realized per-parameter deltas (after bf16 cast) as a multiset, placed at uniformly random positions in the same tensor with random signs, as `add` ops; `lowrank_add` becomes random u, v with the same norms.
  - `to_gguf(model_dir: Path, out: Path, outtype: str = "q8_0") -> str` returns the GGUF's sha256.

- [ ] **Step 1: Write failing tests** on a tiny synthetic safetensors dir (two tensors, 8×8 bf16, in a tmp dir):
  - `test_apply_set_add_scale`: values at the given indices change as specified; all others byte-identical.
  - `test_unknown_tensor_or_index_raises`: unknown tensor → `KeyError`; out-of-range index → `IndexError`.
  - `test_n_changed_counts_bf16_noop`: an `add` of 1e-6 on a value of 1.0 reports `n_changed == 0` while `n_requested == 1`.
  - `test_control_matches_size`: control touches the same tensors, `n_changed` equal to the edit's, sorted absolute deltas equal, positions differ from the edit's.
  - `test_untouched_tensor_identical`: a tensor no op names is byte-identical in `out_dir`.
- [ ] **Step 2:** Run `pytest tests/test_wsp_edits.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement with `safetensors.torch` load/save per shard.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 4: Evaluation runner and acceptance rule

**Files:**
- Create: `dory/wsp/run.py`
- Test: `tests/test_wsp_run.py`

**Interfaces:**
- Consumes: `run_sweep(items, ks, variant, gguf, out, slots)` and `read_rows` from `dory/sweep.py`; `paired_bootstrap(a, b)` from `dory/evaluate.py`; `to_gguf`.
- Produces:
  - `evaluate(gguf: Path, split: str, k: int = 1, slots: int = 4) -> list[dict]`: rows from `results/wsp/{sha256[:12]}/{split}_k{k}.jsonl` (variant `wsp`), running only what's missing. The directory holds `meta.json` with sha256, gguf path and MAX_TOKENS; a mismatch with the current MAX_TOKENS raises.
  - `summarize(rows) -> dict` with `acc`, `n`, `mean_tokens`, `trunc_rate`, `per_source: {src: {"acc", "n", "mean_tokens"}}`. Truncated = `n_gen >= MAX_TOKENS[source]`. Error rows raise (never silently dropped).
  - `compare(base_rows, edit_rows) -> dict` paired by id: `dacc_pts`, `ci_pts: (lo, hi)`, `tokens_ratio`, `per_source_dacc_pts`.
  - `screen_pass(base, edit) -> bool` and `accept(base_sel, edit_sel, ctrl_sel, base_arc, edit_arc) -> tuple[bool, str]` implementing the Global Constraints thresholds; the string names the rule that decided.

- [ ] **Step 1: Write failing tests** with synthetic rows:
  - `test_summarize_truncation_and_per_source`.
  - `test_compare_pairs_by_id` (row order shuffled; ids missing on one side raise `ValueError`).
  - `test_screen_pass_rules`: equal correct count and tokens 1.05× → True; one fewer correct → False; tokens 1.06× → False.
  - `test_accept_paths`: accuracy path accepted; token path accepted; rejected when control is within 2 pts; rejected when one source drops 3.1 pts; rejected when ARC drops 3.1 pts.
  - `test_cache_meta_mismatch_raises`.
- [ ] **Step 2:** Run `pytest tests/test_wsp_run.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 5: PyTorch Dory from the transformers PR, with parity to llama-server

No released transformers has Dory (checked through 5.22.2). Open PR huggingface/transformers#50099 ("Add Dory model support", `modeling_dory.py`, references NVIDIA's merged vLLM port) provides it. We pin its head commit in a separate venv so the loop-controller's `.venv` is untouched, and wrap it; we do not reimplement the model.

**Files:**
- Create: `scripts/wsp_setup_venv.sh`, `dory/wsp/model.py`
- Test: `tests/test_wsp_model.py` (parity tests skip when `models/bf16` is absent)

**Interfaces:**
- Consumes: `models/bf16`, `models/gguf/dory-base-bf16.gguf`.
- Produces:
  - `.venv-wsp`: `python3 -m venv --system-site-packages .venv-wsp`, then `pip install "git+https://github.com/huggingface/transformers.git@fcc6924dfbac964f00787859b568c867ef5ad4cb"`. All `dory/wsp` code and `scripts/wsp_*` run under `.venv-wsp`.
  - `DoryTorch.load(model_dir: Path, device: str = "mps", dtype=torch.bfloat16) -> DoryTorch` (wraps `AutoModelForCausalLM.from_pretrained`), and `DoryTorch.from_config(cfg: dict, seed: int = 0, device: str = "cpu") -> DoryTorch` with random weights (for unit tests).
  - `DoryTorch.forward(ids: torch.LongTensor [1, T], loops: int, capture: set[str] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`: logits `[T, V]` (float32) and captured activations via forward hooks. Hooks on shared recurrent layers fire once per loop; captures are indexed by call order into virtual layers. Capture keys: `resid.{v}` (residual after virtual layer v, 0-based over `16 + 40*loops + 16`), `mlp_act.{v}` (MLP hidden after activation), `attn_out.{v}` (attention output before the residual add, per head as `[T, H, D]`). How the loop count is set per call is read from `modeling_dory.py` (config field or forward kwarg) and recorded in the module docstring.
  - `virtual_to_physical(v: int, loops: int) -> tuple[int, int]` returning (physical layer, loop index; -1 outside the recurrent block).
  - `template_ids(server_url: str, prompt: str) -> list[int]` using llama-server `/apply-template` then `/tokenize`, so PyTorch and the server see identical tokens.

- [ ] **Step 1:** Write and run `scripts/wsp_setup_venv.sh`; verify `from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained("models/bf16")` loads under `.venv-wsp`.
- [ ] **Step 2: Write failing tests:**
  - `test_virtual_to_physical`: loops=2: v=0 → (0, -1); v=16 → (16, 0); v=56 → (16, 1); v=96 → (56, -1).
  - `test_capture_counts` (tiny `from_config`, 2+2+2 layers, loops=3): exactly `2 + 2*3 + 2` `resid.*` keys.
  - `test_parity_next_token` (integration): on the first 20 `train` prompts, top-1 next token after the templated prompt matches the BF16 GGUF server's `/completion` (`n_predict: 1`, `n_probs: 5`) on 20/20 at each k in {1, 2, 4}, and top-5 logprob max abs diff < 0.1.
  - `test_parity_greedy_32` (integration): 32-token greedy continuations identical on ≥ 18/20 at k=1.
- [ ] **Step 3:** Run. Expected: FAIL.
- [ ] **Step 4:** Implement the wrapper.
- [ ] **Step 5:** Run the tests. Expected: PASS. The PR port and the llama.cpp port are independent and both unofficial, so agreement validates both. If parity fails, stop and report the first virtual layer whose output diverges (compare per-layer hidden states against llama.cpp's `llama-eval-callback` dump). No later task starts until this passes.
- [ ] **Step 6:** Commit.

### Task 6: Instruments: logit lens, probes, ablations, content swap

**Files:**
- Create: `dory/wsp/lens.py`
- Test: `tests/test_wsp_lens.py`

**Interfaces:**
- Consumes: `DoryTorch`, `virtual_to_physical`, `template_ids`, `Item`.
- Produces:
  - `logit_lens(model, ids, loops, topk=5) -> dict` with `top_ids [Vl, T, topk]`, `top_p`, and `target_rank [Vl, T]` when `target_ids` passed (final norm + lm_head applied to each `resid.{v}`).
  - `teacher_forced_logprob(model, ids, n_prompt, loops) -> np.ndarray` per generated token.
  - `ablate(model, ids, n_prompt, loops, component: str, mode: "zero"|"mean") -> float`: mean change in teacher-forced logprob of the model's own continuation; component strings `head.{phys}.{h}`, `mlp.{phys}`, `neurons.{phys}:{i,j,...}`, `layer.{phys}`; a recurrent physical layer is ablated in every loop.
  - `collect_resid(model, items_ids, loops, layers: list[int], position: "prompt_end"|"all") -> np.ndarray`.
  - `train_probe(X, y, seed=0) -> dict` (`LogisticRegression`, 5-fold CV `auc_mean`, `auc_std`, plus the same with shuffled labels as a control).
  - `permute_options(item: Item, perm: list[int]) -> Item` for mmlu_pro/arc (options reordered, gold remapped).

- [ ] **Step 1: Write failing tests** with a tiny random-weight `DoryTorch.from_config` (2+2+2 layers, hidden 64, no download):
  - `test_lens_final_layer_equals_logits`: lens at the last virtual layer equals forward logits' top-k.
  - `test_ablate_zero_layer_changes_logprob` and `test_ablate_unknown_component_raises`.
  - `test_probe_shuffled_control_near_chance`: on random X with planted signal, real AUC > 0.9 and shuffled AUC in [0.35, 0.65].
  - `test_permute_options_remaps_gold`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 7: Baselines and research corpus (controller-run compute)

**Files:**
- Create: `scripts/wsp_baselines.py`, `scripts/wsp_corpus.py`
- Create: `research/wsp/corpus/00-baselines.md` … `05-circuits.md`

**Interfaces:**
- Consumes: `evaluate`, `summarize`, `compare`, `apply_edit`, `random_control`, `to_gguf`, `DoryTorch`, everything in `lens.py`.
- Produces: the corpus files below and `results/wsp/` caches for the base GGUF.

- [ ] **Step 1: Baselines** (`wsp_baselines.py`, nohup + caffeinate, log in `results/logs/wsp_baselines.log`): base Q8_0 on `wsp_screen` k=1 twice (determinism: ≥ 57/60 identical outputs with slots=4; else rerun with slots=1 and make slots=1 the default in `evaluate`), `holdout` at k=1, 2, 4, `arc` at k=1. Noise floor: three random edits of 0.016% of the recurrent block's MLP down-projection parameters, Gaussian deltas at that tensor's std, each on `wsp_screen`.
- [ ] **Step 2: Research rows:** 300 `train` items (first 100 per source) at k=1 on the base Q8_0.
- [ ] **Step 3: Corpus** (`wsp_corpus.py`), each file a short markdown with numbers and pointers to `.npz`/`.jsonl` evidence:
  - `00-baselines.md`: Step 1-2 tables (acc, mean tokens, truncation per source and k; noise floor).
  - `01-failure-taxonomy.md`: truncated research answers classified by rule (repeated 8-gram ≥ 3 times = loop; answer phrase never reached without repetition = overlong; other), counts and 3 excerpts per class.
  - `02-lens.md`: logit-lens summary across virtual layers and loops on 30 research prompts: at which depth the final answer token first reaches rank 1; how loops 1 vs 2 vs 4 differ.
  - `03-probes.md`: probes at prompt end for `will_truncate` and `will_be_correct` across layers (AUC with shuffled control); onset-of-repetition probe on per-token residuals.
  - `04-ablations.md`: top 20 components by effect on the teacher-forced logprob of truncated vs finished continuations (heads, MLPs, neuron sets ranked by activation difference).
  - `05-circuits.md`: the controller's rough circuit sketch for the looping/overlong behavior, each claim tagged with the evidence and with its content-swap check result (`permute_options` for mmlu_pro items).
- [ ] **Step 4:** Commit scripts and corpus (large arrays stay in gitignored `results/wsp/`).

### Task 8: Edit rounds (controller authors edits)

**Files:**
- Create: `scripts/wsp_try.py`, `research/wsp/edits/eNNN.json`, `research/wsp/ledger.md`

**Interfaces:**
- Consumes: the corpus; `apply_edit`, `random_control`, `to_gguf`, `evaluate`, `screen_pass`, `accept`.
- Produces: `wsp_try.py EDIT.json` runs: apply onto the current best model dir → GGUF (verifying the dequantized tensor differs from base for every touched tensor) → screen → if pass, selection + control (seed 0) + ARC → accept/reject; appends one ledger entry (edit id, hypothesis, evidence, n_changed, screen/selection/control/ARC numbers, decision rule) and deletes rejected GGUFs and model dirs.

- [ ] **Step 1:** Implement `wsp_try.py` as glue over Tasks 3-4 (no new logic beyond orchestration); dry-run it on a no-op edit (expect: rejected at screen or selection, ledger entry written, files cleaned).
- [ ] **Step 2: Rounds.** The controller reads the corpus and the ledger, writes the next edit file citing evidence and a predicted effect, and runs it. Before spending a screen, a cheap check: the edit's teacher-forced effect on 30 research items moves in the predicted direction (`lens.teacher_forced_logprob` with `DoryTorch.load` on the edited model dir vs the current best); if not, log and skip. Accepted edits stack; the current best becomes the comparison base. Results feed back into the corpus (append to `05-circuits.md`).
- [ ] **Step 3: Stop rule:** stop when the stretch goal is met, or after 6 consecutive screened edits without an acceptance, or after 15 GPU-hours of edit evaluation, whichever first. Record the stop reason in the ledger.
- [ ] **Step 4:** Commit edits, ledger, script.

### Task 9: Sealed final test and report

**Files:**
- Create: `scripts/wsp_final.py`, `results/wsp_decisions.md`

**Interfaces:**
- Consumes: `evaluate`, `summarize`, `compare`, the best model dir and its control.

- [ ] **Step 1:** With `WSP_UNSEAL=1`, run base, best and best's random control at k=1 on `wsp_final`; base and best at k=2 and k=4; `arc` for best if not cached. Run once.
- [ ] **Step 2:** Quantize best to Q5_K_M with `llama-quantize` (from its BF16 GGUF) and score it on `holdout` k=1: does the edit survive the router's format.
- [ ] **Step 3:** Write `results/wsp_decisions.md`: objective, baselines, every edit tried (from the ledger) with decision, final-set results with paired CIs and per-source, Q5_K_M survival, what the controller's hypotheses got right and wrong. If no edit was accepted, say so plainly with the evidence.
- [ ] **Step 4:** Commit.
