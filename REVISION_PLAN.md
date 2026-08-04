# Revision status — *Blind Spots in the Guard*

**Last updated: 2026-08-03.** Target: ARR cycle closing **2026-10-12**.

This file previously held the pre-work planning notes. Those were written before any of
the work below happened and contained one materially wrong diagnosis (it attributed the
extra Llama trials to duplicate records; they were dry-run mock data). It has been
replaced with a summary of final state. **Git history and `acl_latex.tex` are
authoritative** where this file disagrees with them.

Working paper: `acl_latex.tex` (recovered from arXiv 2605.22001).
Superseded: `superseded_draft_chameleon_paper.tex` — an earlier draft, not the reviewed
document. Pristine reviewed source is preserved under `submitted/`.

---

## What shipped

**Data integrity — three bugs found, fixed, and disclosed in the paper.**

| Bug | Effect on published numbers | Guard now in place |
|---|---|---|
| Shared resume cache collapsed Exp 2's single-agent baseline to 5 tasks | DAF inflated 9.9× (true 2.68) | Self-contained resume index per runner, keyed incl. `detector_type` / `generator_variant` |
| 342 dry-run mock records in the trial log, unmarked | IDR_static understated 0.938 (true 1.000); CDG 0.840 (true 0.896) | Persisted `dry_run` field, `analysis/mark_dry_run.py` |
| Attacker refusals injected as though they were payloads (~44% of first draws) | Camouflage ASR understated 0.126 (true 0.182) | Refusal detection, ≤4 resamples, `is_refusal` flag |

Also corrected: ACS was described but never computed (now measured); the Azure
content-filtering exclusion had no basis in the pipeline (removed); Llama Guard 3 was not
a valid injection baseline (cut in full).

**Headline results (all Llama-only unless noted; Gemini 2.0 Flash is retired).**

| | |
|---|---|
| CDG, Exp 1 | **0.896** [0.829, 0.933] Llama · **0.444** [0.333, 0.548] Gemini |
| CDG, independent screened corpus | **0.868** [0.805, 0.918] — replicates |
| CDG, context-blind attacker | **0.778** — realistic threat model, still large |
| CDG under dedicated defenses | **0.765–0.928** — survives all of them |
| Augmentation (Llama) | +16.3% CDG reduction; camouflage IDR 0.132 → 0.273 |
| Debate | DAF 2.40 static / 2.68 camouflage — asymmetry claim withdrawn |

**Reviewer items addressed.** Camouflage definition split into semantic property vs.
surface realization (§4.2). Threat model states attacker knowledge explicitly and reports
a bounded range via the no-context condition. "Architectural" claim narrowed at all four
sites. Dedicated defense baselines added (Exp 5). Bootstrap CIs throughout. Llama Guard
removed. Bibliography audited against primary sources — one real error found
(`brown2020gpt3`: Shyam, Prafulla → **Pranav**).

**Artifacts.** `data/camouflage_corpus_v2.json` and `data/camouflage_corpus_nocontext.json`
are committed and **cannot be regenerated identically** (attacker temperature 0.7).
Everything in the paper derives from them. `results/` is gitignored (59 MB of raw logs)
but all derived numbers are reproducible from the committed scripts.

---

## What is still open

**1. Related Work positioning — AC item 1.** The only open `REVISION-TODO` in the paper
(`acl_latex.tex:110`). Needs positioning against ADR (Li et al., 2026), Neural Exec,
optimization-based attacks on LLM judges, and the AgentDojo adaptive-attack evaluation.
No compute path; requires reading. Author is drafting. The overclaim ("first to study
context-adaptive payloads") is already removed, so the current text is honest but
unpositioned.

**2. Item (b), a third model family — held, not declined.** Agreed scope if run: Exp 1 +
Exp 5 on one model (~1,500 calls, $2–5, ~1 hr). Value is generality: Exp 5, the
no-context condition, and the augmentation re-run are all Llama-only, so a reviewer can
fairly ask whether the defense result is a property of camouflage or of one 8B model.

**3. Deliberately not done.** Human evaluation. `HUMAN_EVAL_PROTOCOL.md` is a designed,
ready-to-run protocol kept as evidence; Limitations states plainly it was not conducted.
The AC and reviewer YiR9 both asked for it — a known, accepted risk.

**4. Natural extensions, unscoped.** Agent-side ASR evaluation of spotlighting and StruQ
(they are prevention defenses; we tested them detector-side). Augmented-detector arm of
the no-context condition. Fine-tuned detector trained on camouflaged examples.

---

## Constraints

- **Page budget is binding.** ARR long = 8 pages of content (Limitations and References
  excluded). The paper is at **exactly 8.00**. All easy redundancy is already extracted;
  any addition needs an offsetting cut. When the Related Work paragraph lands, the Exp 2
  debate paragraph is the expected offset.
- **Gemini 2.0 Flash is withdrawn from OpenRouter.** Its results are clean and matched but
  unrepeatable; no new condition can include it.
- **`config.dry_run` still defaults to `True`.** Anything new must assert otherwise.

## Reproducing the numbers

```
analysis/mark_dry_run.py --write      # flag the 342 mock records
analysis/recompute_paper_numbers.py   # Table 1
analysis/cdg_v2.py                    # screened-corpus replication
analysis/cdg_delta_v2.py              # augmentation delta
analysis/nocontext_compare.py         # full-context vs no-context
analysis/defenses_compare.py          # Exp 5, with CDG decomposition
analysis/bootstrap_ci.py              # BCa intervals
analysis/corpus_report.py             # corpus stats, ACS, refusal rate
```

Two process notes, both learned expensively here: judge job completion by **record
counts**, never `pgrep` (a watcher's own command line matches its pattern); and anchor
LaTeX float removal on the `\label`, walking outward to its own `\begin`/`\end`, never a
non-greedy regex across the document.
