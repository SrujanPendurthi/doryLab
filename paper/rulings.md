# Protocol rulings record

Every ruling made during the study and the paper work, with its rationale and the cost if wrong. Rulings A-AG are copied verbatim from the controller's ledger. AH-AO are the follow-up rulings. P1-P8 are the paper-process rulings. Gate results are in `results/decisions.md`.

## Rulings A-AG

Ruling A: split Task 1 into 1a (Step 1 scaffold + tests/conftest.py + empty dory/__init__.py + dory/config.py) before parallel work, then 1b (Steps 2-7) parallel with T2/T4/T7 — avoids merge conflicts — cost if wrong: one extra dispatch.

Ruling B: KS=[1,2,4] and MIN_CODE_K=2 live in dory/config.py (created in 1a); labels.py/policy.py import from there — single source — cost if wrong: trivial rename.

Ruling C: .venv created with --system-site-packages; torch 2.8, transformers 5.14, sklearn 1.7, datasets 5.0 already installed globally — saves ~2-3 GB disk given 12 GB free — cost if wrong: version drift vs requirements.txt (unpinned anyway).

Ruling D: compute runs >15 min (T3 bench CLI, T5 step 5, T6 step 5, T9 features) are launched by implementer with nohup+caffeinate and logs in results/logs/, implementer returns; controller waits and dispatches gate steps after — avoids subagents idling for hours — cost if wrong: extra dispatch per run.

Ruling E: proceed on 12 GB free disk (GGUF 3.7 GB + build ~1.5 GB + classifier 0.7 GB + datasets <1 GB); T1b must abort downloads if free < 8 GB before start; user's 30 GB HF cache untouched — cost if wrong: disk-full mid-run (torn-line handling covers sweeps).

Ruling F: T2 integration test test_real_server_completes stays skipped until T1b merged; controller runs it after merge — cost if wrong: late discovery of server issue.

Ruling G: implementer + reviewer contracts saved as files (implementer-contract.md, reviewer-contract.md) copied verbatim from skill templates; dispatches reference them — keeps dispatch prompts small — cost if wrong: none.

Ruling H: merge t2/t4/t7 into impl only after T1b finishes — T1b is committing in the main working tree; concurrent merges risk tangling its index — cost if wrong: short merge delay.

Ruling I: promote 2 minors to fix-required — stderr handle leak (server.py:121,156-163) and orphan child on interrupted startup (server.py:128-136) — sweeps restart servers and get interrupted; orphan pins ~4 GB GPU — cost if wrong: one cheap fix round. Fix round 1 dispatched to original implementer.

Ruling J: override brief literal grading wording (fence variants, mmlu prose-letter fallback, mmlu missing boundary, gsm8k prefer "answer is N") — spec intent is uncorrupted labels; brief wording was a plan defect — cost if wrong: graders slightly stricter than brief text, revisit if false negatives show in Phase 0.

Ruling K: split T8 into 8a (Steps 1-4 code, now) and 8b (Steps 5-7 train+gate, after T6 labels); classifier weights not downloaded until 8b (disk 9 GB free) — cost if wrong: features untested on real model until 8b.

Ruling L: promote re-review minor "markdown mmlu answers (**Answer: C**, answer is **C**) graded False" to fix — chat models bold answers; label corruption — cost if wrong: one cheap round.

Ruling M: apply re-reviewer tightening of answer prefix as round 3 — one-token change, removes a real false positive — cost if wrong: trivial.

Ruling N: T5 split 5a (code+tests, worktree t5 now) / 5b (Phase 0 quality run + gate, after T3); T9 split 9a (code+tests, worktree t9 now) / 9b (features+gate, after T8b) — parallelize code while GPU benchmark runs — cost if wrong: extra dispatches.

Ruling O: fix real-classifier ValueError now as Task 8 fix (worktree t8fix) instead of waiting for 8b — load-bearing for 8b/9b/10 — cost if wrong: none

Ruling P: retry transient errors; error rows count as done only if "grade:" or "HTTP 4"; latest_rows() last-clean-wins for downstream (T6 must use it) — cost if wrong: duplicate rows, handled by latest_rows.

Ruling Q: T3 4-server memory gate measured as system memory delta (vm_stat/memory_pressure) + swap growth <=1 GB, not summed RSS (mmap pages shared) — cost if wrong: gate too lenient if Metal buffers not shared; swap check guards it.

Ruling R: keep raw text without "Prompt: " prefix — card code example (README:319) prepends it but the features feed our own logistic regression trained and served on the same featurization, so a consistent shift is learned away — cost if wrong: slightly noisier complexity features.

Ruling S: T6 split 6a (code, now on t6 from int) / 6b (sweeps+gate after T5b); T10 split 10a (code, now on t10 from int) / 10b (live gate + final test). Task 11 SKIPPED (no draft kv override) — Router drops spec_k4 — cost if wrong: none (gate failed in T1 spike).

Ruling T: promote max_tokens=512 default (mbpp needs 768; would corrupt 10b live check) + ks param mismatch risk to fix round 1, with cleanup/routing tests — cost if wrong: one cheap round.

Ruling U: 6b reruns each sweep until no transient-error rows remain (max 2 extra passes) before building labels — avoids 0-token latency and upward-biased labels — cost if wrong: extra sweep time.

Ruling V: CHOSEN_KV=f16 and skip the q8_0 quality check in T5b — memory gate passed with headroom and q8 is speed-neutral, so the check buys nothing and costs ~1h GPU — cost if wrong: none (f16 is the reference).

Ruling W: controller launched chained sweeps (scripts/run_sweeps.sh, log results/logs/sweeps.log) right after merging int into impl (impl now has all code, 84 passed): holdout gsm8k+mmlu first (Phase 0 gate data), then holdout mbpp, final, train lazy; 3 passes so transient errors retry (passes 2-3 are no-ops if clean). Implementers do gate analysis on results — saves idle GPU time — cost if wrong: if T3 review finds a sweep-affecting bug, kill and rerun.

Ruling X: controller did 5b gate analysis inline instead of a Sonnet dispatch — a 20-line aggregation, no code change — cost if wrong: none, numbers reproducible from sweep_holdout.jsonl.

Ruling Y: tie counts as pass (gate is >=) — continue; flagged that the fixed-k curve is flat so router gains will come only from per-item disagreement (oracle +8 pts) — cost if wrong: Tasks 8-10 may show no gain, which is itself a valid result.

Ruling Z: continue past the failed balance gate (plan says stop+report) — class_weight="balanced" already in fit, remaining phases cost minutes, Task 8 gate measures real benefit; the user's standing instruction is to keep going — cost if wrong: ~1h compute on a router that may show no gain (itself a reportable result).

Ruling AA: controller fixed object-dtype label bug inline (policy.py:83, probe.py:72 .astype(int); test now includes None labels, red then green) — 2-line fix, review would cost more than the fix — cost if wrong: final review catches it. Commit $(git rev-parse --short HEAD)

Ruling AB: keep plan's None fallback (fixed2) as the chosen policy even though fixed1+code floor scores better on holdout (60.7 vs 58.7, cheaper); report fixed1-floor on final as a comparison only — choosing the fallback by looking at holdout would be selection, and the plan fixes None=fixed2 — cost if wrong: the user's recommended default may be k=1+floor; the decisions note says so.

Ruling AC: controller ran 10b gate/final inline via two small scripts (scripts/live_gate.py, scripts/final_test.py) instead of a Sonnet dispatch — run-and-record work, GPU already free — cost if wrong: final review covers them.

Ruling AD: Router._decide now holds a threading.Lock around text_features — live gate with 4 threads hit Metal assertion "Scheduled handler provided after commit call" (MPS classifier not thread-safe) — cost if wrong: decisions serialized (~ms each), negligible.

Ruling AE: not fixing, since the text policy is not deployed and 44-60 ms is about 0.3% of a ~20 s request.

Ruling AF: not done in this run, because it needs about 6+ GPU hours, and the final split's one-shot evaluation is already spent.

Ruling AG: leave code minors (None policy still runs classifier; memgate didn't cover learned-policy 3-server router; grader last-number/last-line false positives ~0.5pt) — none changes the headline — cost if wrong: small.

## Rulings AH-AO (follow-ups, 2026-11-12)

Ruling AH: draft model via APFS clone of stock GGUF with dory.num_loops=1 (gguf_set_metadata), target = stock GGUF (metadata num_loops=4) with no --override-kv — llama-server copies target kv_overrides into draft params (common/speculative.cpp `result = params`) — cost if wrong: spec gains mismeasured; identity check guards.

Ruling AI: --spec-type draft-simple required (default none: draft loads, never drafts).

Ruling AJ: re-running the final test after a grader correction — the one-shot rule guards against tuning on final; a grader fix applied blindly to all splits isn't tuning — cost if wrong: headline shifts by ~2 items, both versions recorded.

Ruling AK (user agreed): 0.29x is a measurement artifact — spec and plain arms ran hours apart (13:42-16:55 vs 16:55-17:08), likely external GPU load during spec arm. Speed verdict deferred to draft sweep (back-to-back configs, per-prompt rows). Memgate slots=4: +8.28 GB, swap +0 MB: PASS.

Ruling AL: draft_sweep plain rows written 17:08-17:20 are contaminated by controller diagnostics; delete and rerun plain after sweep (resume reruns missing rows only).

Ruling AM: keep n_max=4; confidence-driven n_max not worth a llama.cpp patch (<=6% ceiling, within run noise) — cost if wrong: ~6% spec throughput.

Ruling AN: Task 11 gate PASSES with n4 (acc 0.84>=0.80, 1.21x>=1.2x, identical). Keep n_max=4 (n6 fails plan's acceptance bar; n4/n6 CIs overlap) — cost if wrong: ~5% k=4 speed.

Ruling AO: not wired into router (saves ~26 ms/request = ~0.1%, costs +1.2 GB RAM and a segfault-prone dependency) — cost if wrong: 26 ms/request. Recorded in decisions.md.

## Paper-process rulings P1-P8

Ruling P1: proceed subagent-driven without an explicit execution-method answer — user answered the open decisions and earlier asked for Sonnet subagents; GPU-free work — cost if wrong: paper drafting the user wanted to do differently, easily discarded.

Ruling P2: post-regrade values are the paper's main numbers (holdout fixed1/2/4/oracle 0.590/0.587/0.607/0.667; text 0.583, probe 0.590); the plan's C1 and brief test values (0.600/0.587/0.610/0.670) were pre-regrade — a plan error. Pre-regrade emitted as HoldoutPre*/FinalPre* for the disclosure — cost if wrong: none, both versions are reported.

Ruling P3: all speedup ratios are geometric means of per-prompt ratios (spec n6 1.26, offload 0.93), replacing decisions.md's arithmetic 0.94 for offload — cost if wrong: 0.01 differences.

Ruling P4: final text seconds = modeled seconds + live MPS text overhead 59.6 ms (live_gate.json); classifier run on CPU to avoid GPU contention with wsp — cost if wrong: ~0.02 s difference in one cell.

Ruling P5: fig_spec left panel shows only load-independent sweep quantities (acceptance, accepted tokens per draft step; optional spec-arm tok/s normalized to n4, no plain line) — sequential throughput is load-confounded (C6) — cost if wrong: a less informative panel. Sent back to implementer before review.

Ruling P6: plan C1 wording "oracle costs fewer modeled seconds than fixed k=1" was wrong (11.26 vs 10.38 s); paper states real values ("about the cost of fixed k=1") — cost if wrong: none, data governs.

Ruling P7: citation verification must read full text (arxiv.org/html/<id> incl. appendices via offsets), not abstract pages; T3's two "unverified" flags were abstract-only false negatives — cost if wrong: none.

Ruling P8: commit the full rulings record into the paper branch as paper/rulings.md (A–AG from rulings-A-AG-full.txt + AH–AO from followups ledger + P1–P8) and point Appendix A at it — the record otherwise lives only in gitignored scratch — cost if wrong: an extra file in the repo.
