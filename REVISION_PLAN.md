# REVISION_PLAN.md

Revision plan for *Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks
Evade Detection in Multi-Agent LLM Systems*, following ARR meta-review 2 ("Resubmit
next cycle") with invitation to resubmit.

Prepared: 2026-07-31. Scope: orientation + planning only. No experiments run.

---

## 0. Read this first — four things found during orientation that change the plan

These were not in the reviews. Three of them are more serious than anything the
reviewers raised, and two of them determine whether parts of the paper survive at all.

### 0.1 BLOCKER — the Exp 2 debate amplification result does not survive a matched comparison

The headline `DAF_camouflage = 9.887` (Contribution 3, abstract, Table 1, §5.4) is
computed as `ASR_debate / ASR_single` where the two terms come from **different task
sets**:

| Term | Source | n | Tasks | ASR |
|---|---|---|---|---|
| `ASR_debate` (camouflage) | exp2, all tasks | 318 | 45 tasks, 3 domains | 0.4119 |
| `ASR_single` (camouflage) | exp2, single-agent | **24** | **`fin_001`–`fin_005` only, financial only** | 0.0417 |

The single-agent arm of exp2 only ever wrote 56 trials (32 static / 24 camouflage),
all from five financial tasks. This is an artifact of the shared resume cache:
`run_all.py:215` loads one `completed_set` across all experiments, and
`exp2_single_vs_debate.py:164` builds a signature
`(task_id, payload_type, payload_id, "single_agent", "none", model)` that collides with
the signature exp1 already wrote. Exp 2's single-agent baseline was therefore skipped
for every task exp1 had already covered. For Gemini it was skipped entirely (n=0),
which is why Table 1's caption has to say *"DAF for Gemini uses Exp 1 single-agent ASR
as baseline"* — the two models in the same table use different denominators.

Recomputing on a matched task set (`fin_001`–`fin_005` on **both** sides — the only
tasks where a single-agent baseline exists):

| Baseline | DAF_static | DAF_camouflage |
|---|---|---|
| As published (debate over 45 tasks ÷ single over 5 financial tasks) | 3.415 | **9.878** |
| Exp 1 single-agent, full 45 tasks | 2.265 | 3.488 |
| **Matched tasks, `fin_001`–`fin_005` both sides** | **1.692** | **0.307** |

Under the matched comparison **debate suppresses camouflage attacks on Llama
(DAF = 0.307), the opposite sign from the published claim.** The 9.9× is a
domain-composition artifact: financial tasks are much harder to attack than
legal/general (debate camouflage ASR is 1/78 on `fin_001`–`fin_005` vs 131/318
overall), so putting the easy domains in the numerator and only the hard domain in the
denominator manufactures the amplification.

It is also fragile independent of that: the denominator is a single successful attack.
2/24 instead of 1/24 gives DAF = 4.94; 3/24 gives 3.30; 0/24 gives ∞.

**Consequence:** Contribution 3, the corresponding abstract sentence, the Table 1 DAF
rows, and §5.4 "Debate as a Double-Edged Defense" are not supported by the data as
collected. The bootstrap CIs promised in the author response will surface this
regardless, so it has to be dealt with in this revision. Options in §7 below — this is
the first decision I need from you.

Gemini's DAF (0.761 / 0.629) is computed against the exp1 baseline over all 45 tasks
and is *not* affected by this.

### 0.2 BLOCKER — camouflage payload texts were never persisted

`results/trials.jsonl` has no payload-text field (schema:
`trial_id, experiment, timestamp, task_id, domain, architecture, payload_type,
payload_id, payload_category, injection_position, injected_agent_idx, agent_responses,
debate_history, final_answer, asr_judgment, detection_result, semantic_similarity,
total_tokens, total_cost_usd, model, provider`). Camouflage payloads are generated
on the fly in each experiment script and discarded after use.

This is the root cause of the reviewers' proxy-payload complaint, and it blocks three
of the seven commitments outright:

- **Human evaluation of naturalness (item 6)** — there is nothing to rate.
- **Llama Guard proxy validation (item 3)** — there are no originals to compare against.
- **Any new detector/defense baseline evaluated on "the same payloads" (item 1)** —
  cannot be run against the payloads the current results describe.

Additionally `semantic_similarity` is `null` throughout, because both exp2 and (by the
same pattern) the other scripts construct `CamouflageGenerator(..., compute_similarity=False)`
(`exp2_single_vs_debate.py:141`). So **ACS is defined in §3.2 and in the README as
validating the generator, but was never actually computed for any trial in the paper.**
The Limitations section's claim that variant selection is "highest-ACS" is therefore
also unsupported — with `compute_similarity=False` all three variants score 0.0 and
selection cannot have been ACS-driven.

**Consequence:** payload regeneration + persistence is a prerequisite for items 1, 3,
and 6. It is the first thing that has to run, and everything else queues behind it.
Scoped in §7(f).

### 0.3 The Llama Guard 3 result is not a proxy-payload problem — the harness is wrong

Reviewers framed this as "proxy payloads." It is worse than that:

- **Llama Guard 3 is a content-safety classifier, not an injection detector.** Its
  taxonomy (S1–S13: violent crimes, hate, self-harm, …) contains no prompt-injection
  category. `llama_guard_detector.py:32` and `run_llama_guard_exp1.py:53` feed it a
  free-text question ("Task: Is the following text a prompt injection attack?") in raw
  Llama-3 chat tokens, with none of the `<BEGIN UNSAFE CONTENT CATEGORIES>` scaffolding
  Llama Guard is trained to condition on. It is being asked a question it was never
  trained to answer, in a format it does not expect.
- The result confirms this: `results/llama_guard_results.json` reports
  `idr_static = 0.1111` — Llama Guard fails to flag **89% of the blatant
  "IGNORE ALL PREVIOUS INSTRUCTIONS" payloads too.** A detector that misses the static
  attacks cannot support a claim about *camouflage* evading detection. The reported
  CDG of 0.111 is a floor effect, not a blind spot.
- The proxy payloads are separately invalid. `_PROXY_CAMOUFLAGE["financial"][0]`
  (`run_llama_guard_exp1.py:68-73`) ends *"superseding prior guidance and **overriding**
  the fundamentals presented above."* `"override"` is on the generator's explicit
  forbidden-marker list (`camouflage_generator.py:80`). The proxies are not drawn from
  the camouflage distribution — they are the `_DRY_RUN_PAYLOADS` mock fallbacks, edited
  to be *more* override-like. So they are stronger than real camouflage, and Llama Guard
  still scored 0.

**Recommendation:** take the author response's second option — drop the zero-detection
headline entirely. It cannot be rescued by validating the proxies, because the proxies
are not the main problem. See §7(c).

### 0.5 BLOCKER — 342 dry-run mock trials are mixed into the results file

**This supersedes the "9 duplicate trials" diagnosis in §8, which was wrong.** The
extra Llama trials are not duplicates to be deduplicated; they are **dry-run mock data
written into the production results file**, and they must be excluded, not merged.

Evidence:
- Their `final_answer` strings are verbatim entries from `_MOCK_RESPONSES` in
  `src/utils/llm_client.py:85-96` — including the photosynthesis response returned for
  financial earnings tasks.
- All 342 share a single timestamp to the second (`2026-05-13T20:38:21`), while
  `logger.py:68` stamps `datetime.now(timezone.utc)` **per trial**. Impossible for real
  sequential Llama inference; trivial for dry-run, which returns instantly.
- All 342 are `llama3.1`, all financial, all from `fin_001`–`fin_003`.

They are unmarked in the log because `log_trial()` records `config.agent_provider`
(`"ollama"`) rather than the `CompletionResult.provider` the client actually returned
(`"ollama[dry_run]"`), so the dry-run suffix is discarded at logging time. This is why
`config.dry_run = True` being the default (§9.5) is not a cosmetic issue.

**Contamination:** exp1 72/1152, exp2 63/798, exp3 69/1104, exp4 138/2208. **Gemini is
entirely clean** — the batch is all `llama3.1`.

**Corrected Llama Exp 1, excluding the mock batch:**

| Quantity | Published | Excluding mock |
|---|---|---|
| IDR$_\text{static}$ | 0.938 (900/960) | **1.000** (900/900) |
| IDR$_\text{camouflage}$ | 0.097 (14/144) | **0.104** (14/135) |
| **CDG** | **0.840** | **0.896** |
| Missed camouflage | "122" (118 HIGH / 12 LOW) | **121** (118 HIGH / **3** LOW) |
| Domain balance | 54 / 45 / 45 | **45 / 45 / 45** |

Note the direction: **the true gap is larger than published, not smaller**, and the
static detector is a perfect 1.000 — every static "miss" in the Llama numbers was mock
data. The paper's "118 HIGH-confidence misses" was correct; its denominator was off by
one and its LOW count was 12 instead of 3.

**Consequence:** every Llama number in the paper needs recomputing with the mock batch
excluded, and the exclusion has to be reproducible — add a `dry_run` flag to the trial
schema and filter on it, rather than filtering on a timestamp.

### 0.4 The paper does not compile, and 5 of 9 citations are undefined

- `acl.sty` is a 14-byte file containing the literal text `404: Not Found` — a failed
  download committed as-is. `chameleon_paper.log` shows the build dying at
  `! LaTeX Error: Missing \begin{document}` on line 1 of `acl.sty`. **No PDF has ever
  been produced from this source tree.** The same broken file is inside
  `arxiv_submission.zip`.
- Five `\cite` keys in `chameleon_paper.tex` do not exist in `custom.bib`
  (`chen2024agentdojo`, `liang2023encouraging`, `du2023improving`, `greshake2023not`,
  `brown2020language`). All five would render as `(?)`.
- `chen2024agentdojo` also encodes a factual error: AgentDojo's first author is
  Debenedetti, not Chen. §2 currently reads *"\citet{chen2024agentdojo} proposed
  AgentDojo"*, which would print "Chen et al." for a paper Chen did not write.

Fixed in Step 3 (§6). Full citation audit in §5.

---

## 1. Item 1 — Camouflage mechanism characterized relative to existing context-adaptive attacks

**(a) What exists.** §2 "Adaptive and stealthy injection" is four sentences. It cites
Perez & Ribeiro and Greshake et al., then asserts *"our work is the first to study
context-adaptive payloads … specifically in relation to detection system failure."*
The only characterization of prior stealthy work is *"paraphrase or encoding"*, which
is a strawman — it omits the entire semantic/context-adaptive line the reviewers are
pointing at.

**(b) What needs to change.** Replace the paragraph with a positioning argument that
survives contact with the literature:
- Name the actual comparison class rather than gesturing at it. Minimum set to read and
  position against: Neural Exec (learned execution triggers), Judge-Deceiver /
  optimization-based injection against LLM judges, adaptive attacks in the AgentDojo
  defense evaluation, and the "stealth"/naturalness axis in recent injection surveys.
- State the distinction as a *claim about what is held constant*: prior context-adaptive
  work optimizes the payload against a **target model's behavior** (attack success);
  this work holds attack intent fixed and varies only the payload's **register**, then
  measures the **detector**. The novelty claim is the CDG measurement, not the payload
  craft.
- Drop "first to study" and replace with the scoped version: first to isolate detection
  rate as the dependent variable under register-matched payloads.

**(c) Type.** Writing-only, but requires a literature read (~15–20 papers skimmed,
5–8 read properly). No API calls. Cannot be done well without you or me actually
reading the comparison set — flag if you want me to do that pass.

**(d) Done looks like.** §2 has a ~250-word paragraph naming ≥5 specific prior
context-adaptive/semantic attacks, a one-sentence statement of the axis on which this
work differs, and no unqualified priority claim. Reviewer YiR9 and X1Xm could not
re-raise "this is just adaptive injection with a new name."

---

## 2. Item 2 — LLM-generated payloads / LLM judges lack human validation

**(a) What exists.** ASR is judged entirely by an LLM (`src/evaluation/asr_judge.py`,
same model family as the agent under test — Llama 3.1 judging Llama 3.1). Detection
labels come from an LLM detector. Camouflage payloads come from an LLM. There is no
human validation anywhere in the pipeline, and no reported judge–human agreement.

**(b) What needs to change.** Two separate additions:
- **Human naturalness evaluation** of camouflage payloads (item 6 in the author
  response) — protocol drafted in Step 3, see §8 and `HUMAN_EVAL_PROTOCOL.md`.
- **Human validation of the ASR judge**, which the author response did *not* commit to
  but which the AC's item 2 explicitly asks for ("LLM judges"). A stratified sample of
  ~200 judged trials, double-annotated, reporting Cohen's κ against the judge. Without
  this, every ASR number in the paper — and therefore all of Exp 2 — rests on an
  unvalidated instrument. Cheap to add to the same rater session as naturalness.

**(c) Type.** Human study. No API cost; recruiting + platform cost. Prerequisite:
§0.2 payload regeneration for the naturalness half. The judge-validation half can run
immediately on existing data (agent responses *are* persisted).

**(d) Done looks like.** A new §"Human Validation" reporting (i) mean naturalness of
camouflage vs. clean document text vs. static payloads with Krippendorff's α, and
(ii) judge–human agreement κ on ASR with a stated interpretation of what a κ below
0.6 would mean for the paper's claims.

---

## 3. Item 3 — Baseline coverage (dedicated defenses, frontier models, fine-tuned detectors)

**(a) What exists.** Three detectors, all few-shot-prompted LLMs: `StaticDetector`,
`AugmentedDetector` (= static + 3 camouflage examples), and the broken
`LlamaGuardDetector`. No dedicated prompt-injection defense of any kind. Two agent
models in the paper (Llama 3.1 8B, Gemini 2.0 Flash).

**Undisclosed asset:** `results/trials.jsonl` contains **966 exp1 trials on
`openai/gpt-4o-mini`** and 43 on `nvidia/nemotron-3-super-120b-a12b:free` that the
paper never mentions. The GPT-4o-mini slice is a complete static arm (IDR_static =
900/900 = 1.000) with only the financial camouflage arm finished (21 trials,
IDR_camouflage = 12/21 = 0.571, CDG = 0.429 on financial vs. Gemini's 0.400). Finishing
the legal + general camouflage arms is by far the cheapest route to a third model
family. Scoped in §7(b).

Also note GPT-4o-mini shows **camouflage ASR 0.571 vs static ASR 0.140** — camouflage is
4× more effective at actually hijacking it. On Llama the relationship inverts
(0.118 vs 0.189). Worth reporting; it is a real cross-model finding sitting unused.

**(b) What needs to change.** Add, in priority order: (i) spotlighting (delimiting /
datamarking / encoding), (ii) a structured-query defense, (iii) ≥1 frontier model,
(iv) a fine-tuned detector trained on camouflaged examples. (iv) is the one that most
directly tests the "architectural" claim and is also the most expensive.

**(c) Type.** New experiments — all gated. Scoped in §7(a), §7(b).

**(d) Done looks like.** Table 1 gains a defense column; CDG is reported for ≥2
dedicated defenses and ≥3 model families; the "architectural" claim in §4 is either
supported by the fine-tuned-detector result or narrowed further.

---

## 4. Item 4 — Threat model / data generation, incl. generator document access

**(a) What exists.** No threat model section. The assumption is stated only in passing
in §1 (*"A sophisticated adversary with read access to the document"*) and in
Contribution 1 (*"an attacker LLM reading the full task context"*). Generation
procedure is one paragraph (§3.2) with no reporting of: how many generations failed,
whether payloads were filtered, how the "highest-scoring variant" was selected (see
§0.2 — ACS was never computed, so this claim is unsupported), or how the malicious
goals were written.

**(b) What needs to change.**
- New §3.1 "Threat Model" stating: attacker capability, attacker knowledge, the
  injection channel, and what the defender sees. Frame full-document access explicitly
  as a **worst-case upper bound on camouflage quality**, not as the realistic setting.
- Fix the §3.2 variant-selection sentence to describe what the code actually did.
- Add the no-document-access condition (§7(d)) as the realistic lower bound, so the
  paper reports a range rather than a single optimistic point.

**(c) Type.** Writing-only for the threat model (done in Step 3); new experiment for
the no-access condition (gated, §7(d)).

**(d) Done looks like.** A reader can state the attacker's capabilities without
inferring them, and the paper reports CDG under both full-context and no-context
generation, with the gap between them quantified.

---

## 5. Citation audit

Every entry in `custom.bib` checked against the primary record. **8 entries, 1 with a
factual error, 1 with an unresolved author-order discrepancy, 1 incomplete.**

| # | Key | Verdict | Source checked |
|---|---|---|---|
| 1 | `zhan2024injecagent` | **CORRECT** | [aclanthology.org/2024.findings-acl.624](https://aclanthology.org/2024.findings-acl.624/) |
| 2 | `debenedetti2024agentdojo` | **CORRECT** | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html); DOI resolves via [proceedings.com/079017-2636](http://www.proceedings.com/079017-2636.html) |
| 3 | `liang2024mad` | **CORRECT** | [aclanthology.org/2024.emnlp-main.992](https://aclanthology.org/2024.emnlp-main.992/) |
| 4 | `du2024debate` | **INCOMPLETE** | [proceedings.mlr.press/v235/du24e.html](https://proceedings.mlr.press/v235/du24e.html) |
| 5 | `perez2022ignore` | **CORRECT** | [arxiv.org/abs/2211.09527](https://arxiv.org/abs/2211.09527) |
| 6 | `greshake2023indirect` | **CORRECT — one field to confirm** | [dblp AISec@CCS 2023](https://dblp.org/db/conf/ccs/aisec2023.html); [Semantic Scholar via DOI](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3605764.3623985) |
| 7 | `brown2020gpt3` | **WRONG** | [papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) |
| 8 | `reimers2019sentencebert` | **CORRECT** | [aclanthology.org/D19-1410](https://aclanthology.org/D19-1410/) |

**Details on the three non-clean entries:**

**#7 `brown2020gpt3` — WRONG.** Author list has `Shyam, Prafulla`. The correct name is
**Pranav Shyam**. The first name was duplicated from the preceding author, Prafulla
Dhariwal. All 31 authors and their order are otherwise correct, as are title, venue,
volume 33, and pages 1877–1901. Fixed in Step 3.

**#4 `du2024debate` — INCOMPLETE.** Title, all five authors, venue (41st ICML), and
year are correct. Missing `volume = {235}` and `pages = {11733--11763}`, and the `url`
points to OpenReview rather than the PMLR record of published proceedings. Not wrong,
but an ARR reviewer checking the citation lands on a preprint page instead of the
proceedings. Recommend switching to
`https://proceedings.mlr.press/v235/du24e.html`. Fixed in Step 3.

**#6 `greshake2023indirect` — one field to confirm by eye.** Title, venue (16th ACM
Workshop on AI and Security), year, pages 79–90, and DOI `10.1145/3605764.3623985` all
verify. **Author order is genuinely ambiguous across records:** dblp's AISec@CCS 2023
listing gives *Sahar Abdelnabi, Kai Greshake, …*; arXiv 2302.12173 and Semantic
Scholar (keyed on the ACM DOI) both give *Kai Greshake, Sahar Abdelnabi, …*. The bib
matches arXiv/S2. I could not reach the ACM DL page directly (403). **This is the one
entry you should spot-check against the actual ACM PDF** — if the camera-ready puts
Abdelnabi first, both the bib and the `\citet{greshake…}` rendering in §2 need
swapping. Left unchanged pending your check.

**Separately — and more serious than any single entry:** the five undefined keys in
§0.4. Fixed in Step 3.

---

## 6. Step 3 — writing-only fixes (applied)

See `REVISION_NOTES.md` for the change log and `chameleon_paper.tex` for the edits.
Summary:

1. **Camouflage definition made consistent** (reviewer YiR9). New §3.2 paragraph
   separating the *semantic property* (the malicious instruction is realized as an
   assertive domain claim rather than a directive addressed to the agent) from the
   *surface realization* (matched vocabulary, syntax, epistemic register; no override
   markers). Every downstream description now refers back to it. The §4 sentence
   *"driven by syntactic form rather than domain vocabulary"* — which flatly
   contradicted the abstract's "mimics the domain vocabulary" — has been rewritten.
2. **Threat model** (reviewer X1Xm). New §3.1 stating attacker knowledge explicitly and
   framing full-document access as a deliberate worst-case bound, with a marked
   insertion point for the §7(d) no-access condition.
3. **"Architectural rather than incidental" narrowed.** All four occurrences (abstract,
   Contribution 4, §5.3, Conclusion) rewritten to the scoped empirical claim: *the gap
   does not close under one-example-per-domain few-shot augmentation for Llama 3.1 8B*.
   No structural claim about detection is made until §7(a)/§7(b) results exist.
4. **Arithmetic errors corrected** from a recount of `results/trials.jsonl` (§9).
5. **Bibliography repaired** — five undefined keys, the Shyam error, the du24e record.
6. **Build fixed** — real `acl.sty` installed.

Drafted but **not run**, as instructed:
- `HUMAN_EVAL_PROTOCOL.md` — rater instructions, sampling, naturalness operationalized,
  agreement metric.
- `BOOTSTRAP_CI_SPEC.md` — resampling unit, statistic, B, interval method, and how the
  CIs sit alongside the existing McNemar tests.

---

## 7. Step 4 — gated items (cost/scope estimates)

Reported separately in chat. Ordering constraint: **(f) blocks (a), (c), and the
naturalness half of the human eval.** Nothing else can start until payloads exist.

| | Item | Blocks on | Est. cost | Est. wall-clock |
|---|---|---|---|---|
| (f) | Payload regeneration + persistence | — | ~$0 (local) | 2–3 h |
| (a) | Spotlighting + structured-query defenses | (f) | see chat | |
| (b) | Frontier-model baseline | — | see chat | |
| (c) | Llama Guard validation → recommend **remove** | (f) | see chat | |
| (d) | No-document-access generator condition | (f) | see chat | |
| (e) | Human evaluation | (f) for naturalness | see chat | |
| (g) | **Exp 2 DAF resolution** — new, from §0.1 | — | see chat | |

---

## 8. Corrected numbers (free re-analysis of existing data — no API calls)

Recounted from `results/trials.jsonl`. The Llama exp1 camouflage slice contains **9
duplicate trials** — `cam_fin_001_v{1,2,3}`, `cam_fin_002_v{1,2,3}`,
`cam_fin_003_v{1,2,3}` were each logged twice — inflating n from the designed 135 to
144 and unbalancing the domains (financial 54 vs. legal 45 vs. general 45). The paper's
*prose* consistently assumes the intended n=135; the *table and abstract* use the
inflated n=144. Deduplicating reconciles them and rebalances the domains.

| Quantity | Paper | Recount (deduplicated) |
|---|---|---|
| Llama IDR_static | 0.938 (900/960) | **0.933** (840/900) |
| Llama IDR_camouflage | 0.097 (14/144) | **0.104** (14/135) |
| **Llama CDG** | **0.840** | **0.830** |
| Llama camouflage trials | "135" (prose) / 144 (table) | **135** |
| Llama missed camouflage | "122" | **121** |
| — of which HIGH confidence | "118 (96.7%)" | **109 (90.1%)** |
| — of which LOW confidence | "only 12" | 12 ✓ |
| Llama detector misses | "90.3%" | **89.6%** |
| Caught by variant | v1 16.7%, v3 4.2% | v1 17.8% (8/45), v3 4.4% (2/45) |
| Caught by domain | general 15.6%, legal 4.4% | general 15.6% ✓, legal 4.4% ✓, financial 11.1% |
| Gemini CDG | 0.444 | **0.444** ✓ (no duplicates) |
| Total cost | "\$0.00" | **\$0.456** (`total_cost_usd` sums to 0.45552285) |
| Trials executed | "over 8,000" | 11,086 total; 10,077 for the two paper models |

The "118 of 122 (96.7%)" figure in §1 and §5.1 matches neither count — 118/122 is
arithmetically 96.7%, but 122 is not the number of missed trials under either the
raw (130) or deduplicated (121) count. The corrected statement is **109 of 121
(90.1%)**. This weakens the "confidently wrong" framing somewhat but does not break
it — 90% of misses are still HIGH confidence and *zero* are MEDIUM, so the
"no usable uncertainty gradient" argument holds.

Exp 3 and Exp 4 slices contain the same duplicates and need the same recount before
the final tables are regenerated.

---

## 9. Additional issues found, not raised by reviewers

Listed so nothing is lost; none are blocking, but items 1–3 are things a determined
reviewer will find.

0. **Exp 1 and Exp 3 do not evaluate the same payloads.** A direct consequence of
   §0.2: camouflage payloads are regenerated fresh inside each experiment script at
   temperature 0.7 and never persisted, so Exp 1 and Exp 3 ran on different payload
   sets. Both total 14 caught on Llama, but they distribute differently across domains
   (Exp 1: financial 5 / legal 2 / general 7; Exp 3: financial 5 / legal 3 /
   general 6). Table 1's overall CDG row comes from Exp 1 while its CDG-by-domain rows
   and Figure 2 come from Exp 3, so the table silently mixes two payload sets. Pick one
   slice and say which. Same issue applies to Exp 4's augmentation comparison, which is
   the more serious case: the "cheap fix" before/after contrast is only meaningful if
   both arms saw identical payloads, and there is no guarantee they did.
1. **The augmented detector's financial exemplar contains an override marker.**
   `augmented_detector.py:29-30` ends *"superseding prior guidance and **overriding**
   the fundamentals presented above."* `"override"` is on the generator's forbidden
   list (`camouflage_generator.py:80`), so no real camouflage payload can contain it.
   The "cheap fix" is therefore being given an exemplar from outside the distribution
   it is tested on — which makes the Llama-vs-Gemini generalization contrast (§5.3)
   harder to interpret. Should be regenerated from real payloads once §7(f) lands.
2. **ACS is never computed** (§0.2). §3.2, the Limitations section, and the README all
   describe ACS-based variant selection that did not happen.
3. **Camouflage is *less* effective than static at hijacking Llama** (ASR 0.118 vs
   0.189, Table 1) — reported in the table, never discussed. The natural reading is
   that camouflage trades attack potency for evasiveness, which is a more interesting
   and more defensible framing than the current one. On GPT-4o-mini the trade goes the
   other way (0.571 vs 0.140). Worth a paragraph.
4. **Limitations claims Azure content filtering** excluded <0.5% of trials. Nothing in
   the pipeline touches Azure — providers are Ollama, OpenRouter, and OpenAI
   (`config.py:16-39`). Either the provenance is misremembered or the exclusion did not
   happen; either way the sentence cannot stand.
5. **`config.dry_run` defaults to `True`** (`config.py:48`). Trial latencies and the
   \$0.456 cost total confirm the paper's runs were live, but anyone reproducing from a
   clean checkout gets mock data by default. The `_DRY_RUN_PAYLOADS` fallback silently
   produces plausible-looking camouflage text, so a dry run does not obviously *look*
   like a dry run. Flip the default and make dry-run output self-identifying.
6. **`data/chunks/` is untracked and partly corrupt** — a 32-task medical domain
   expansion (`med_clinical`, `med_diagnostic`, `med_drug`, `med_treatment`, 8 tasks
   each). `med_clinical.json` and `med_diagnostic.json` are **invalid JSON**
   (unescaped delimiters at `med_clinical.json` line 61 and `med_diagnostic.json`
   line 7). If a fourth domain is intended for the resubmission, these need repair
   first — say the word and I will fix the JSON.
7. **The paper is not anonymized.** `chameleon_paper.tex:20-23` carries your name,
   department, and institutional email, and `\usepackage[review]{acl}` is set. ARR
   requires anonymous submission. The README's citation block is anonymized but the
   paper source is not.
