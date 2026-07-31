# Bootstrap Confidence Interval Specification — DRAFT, NOT RUN

Addresses the author-response commitment to "bootstrap confidence intervals alongside
the existing significance tests."

**Status:** spec only, not run, per instruction. Note that this is pure re-analysis of
`results/trials.jsonl` — **no API calls, no cost, runs in under a minute.** It is the
cheapest item in the entire revision and it is also the one most likely to change what
the paper claims.

---

## 1. Why this matters more than it sounds

The existing McNemar tests answer "is the gap non-zero?" and the answer is an
emphatic yes ($\chi^2 = 38.03$, $p < 0.001$). That was never in doubt. What the paper
currently has no way to express is **how precisely each number is estimated**, and
several headline numbers are estimated very imprecisely:

- `DAF_camouflage = 9.887` has a denominator of **one successful attack out of 24**
  (REVISION_PLAN §0.1). Any honest interval on this is enormous and probably includes
  1.0. This is the number the bootstrap will most visibly destroy, and it is better
  that we find that than a reviewer.
- The variant breakdown (v1 17.8% / v2 8.9% / v3 4.4%) rests on 45 trials per cell with
  8, 4, and 2 positive events. The v1-vs-v3 difference is 6 events.
- `CDG_legal = 0.956` for Llama rests on 2 caught out of 45.

Anywhere the numerator is single digits, the point estimate is being reported with a
false air of precision. That is the gap this fills.

## 2. Resampling unit

**The task, not the trial.** This is the single most important decision in this spec
and the easiest to get wrong.

Trials are not independent: 3 camouflage variants share a task, a document, and a
malicious goal; 20 static payloads share the same. Resampling trials treats 135
camouflage observations as 135 independent draws when they are 45 clusters of 3, which
understates variance — by roughly $\sqrt{1 + (m-1)\rho}$ for cluster size $m$ and
intra-cluster correlation $\rho$. With $m = 3$ and the strong per-task clustering
visible in the data, that is a meaningful underestimate.

**Procedure — cluster bootstrap:**
1. Let $T$ = the 45 task IDs.
2. Draw $T^*$ = 45 task IDs with replacement from $T$.
3. Assemble the resampled dataset from **all** trials belonging to the drawn tasks
   (a task drawn twice contributes its trials twice).
4. Recompute the statistic on the resampled dataset.
5. Repeat $B$ times.

Stratify the draw by domain (15 financial / 15 legal / 15 general) so every replicate
preserves the design's domain balance. Without stratification some replicates will draw
few legal tasks and the domain-conditional statistics become unstable for reasons that
have nothing to do with the underlying uncertainty.

## 3. Statistics to bootstrap

| Statistic | Definition | Note |
|---|---|---|
| IDR$_\text{static}$, IDR$_\text{camouflage}$ | per model, per detector | |
| **CDG** | IDR$_s -$ IDR$_c$ | primary; report per model and per domain |
| CDG$_\Delta$ | $(\text{gap}_\text{before} - \text{gap}_\text{after}) / \text{gap}_\text{before}$ | ratio — see §5 |
| ASR | per model × architecture × payload type | |
| **DAF** | ASR$_\text{debate}$ / ASR$_\text{single}$ | ratio with a near-zero denominator — see §5 |
| Catch rate by variant | v1 / v2 / v3 | and the v1$-$v3 difference directly |
| Catch rate by domain | | |

For any **difference** or **contrast** (v1 $-$ v3, CDG$_\text{legal}$ $-$
CDG$_\text{financial}$, CDG$_\text{Llama}$ $-$ CDG$_\text{Gemini}$), bootstrap the
contrast itself rather than each arm separately and eyeballing overlap. Non-overlapping
intervals imply a significant difference but overlapping ones do not imply the absence
of one; reading two separate intervals against each other is the classic error and a
reviewer will catch it.

## 4. Interval method

**BCa (bias-corrected and accelerated), $B = 10{,}000$.**

- Percentile intervals are inadequate here: several statistics are bounded (IDR ∈ [0,1]),
  skewed near their bounds (IDR$_\text{camouflage}$ ≈ 0.10 on Llama), and one is a ratio
  of small proportions. BCa corrects for both bias and skew and is the right default.
- Acceleration $\hat{a}$ from jackknife-over-tasks (leave-one-task-out, 45 replicates),
  consistent with the cluster resampling unit.
- $B = 10{,}000$ for reported intervals. Cheap — the whole grid runs in well under a
  minute on 11k records.
- Fixed seed (42, matching `config.seed`); record it in the output.

**Degenerate cases.** Where a resample yields an all-zero denominator (entirely
plausible for DAF given 1/24), BCa will fail or return absurd endpoints. Do not silently
drop those replicates — that biases the interval toward the finite values. Instead:
report the **proportion of replicates that are undefined or infinite** alongside the
interval, and if it exceeds ~1%, report the statistic as unestimable rather than
reporting an interval at all. For `DAF_camouflage` on Llama I expect this to trigger.
That is the correct outcome and should be stated as such in the paper.

## 5. Ratio statistics specifically

DAF and CDG$_\Delta$ are ratios and need care:

- Bootstrap the **log** of the ratio where the ratio is strictly positive, then
  exponentiate the interval endpoints. This respects the multiplicative scale — an
  interval like [0.8, 12.0] is meaningless on a linear scale and interpretable on a
  log one.
- Report the ratio's interval, never the ratio alone. `DAF = 9.9` presented without an
  interval is the current problem in miniature.
- For DAF, also report both numerator and denominator with their own intervals, so a
  reader can see which arm carries the uncertainty.

## 6. Relationship to the existing McNemar tests

The bootstrap **supplements**, it does not replace. Keep both and say what each is for:

- **McNemar** — a paired test on task-level detection outcomes, answering *"does the
  detector behave differently on static vs. camouflaged payloads?"* Correctly paired,
  correctly reported, and it stays.
- **Bootstrap CIs** — interval estimates for effect magnitude, answering *"how large,
  and how precisely do we know?"*

One caveat to add to the existing McNemar reporting: `significance_tests.py:60`
aggregates to a per-task majority vote before the test, which is the right way to get
independent pairs, but it also discards within-task variation. Note this in the paper —
the McNemar $n$ is 45 paired tasks, not 135 or 144 trials, and the current text does
not make that clear.

## 7. Reporting

- Table 1: every proportion becomes `0.830 [0.771, 0.879]`.
- Prose: intervals on every headline number in the abstract, intro, and conclusion.
- New Appendix table: full grid of statistic × model × domain with $B$, method,
  seed, and undefined-replicate proportion.
- **Pre-commit to reporting intervals for statistics that turn out to be wide.**
  The temptation once the numbers land will be to keep intervals for CDG (which will
  be tight and supportive) and quietly omit them for the variant breakdown and DAF
  (which will not be). Do not.

## 8. Implementation note

Add as `analysis/bootstrap_ci.py`, reading `results/trials.jsonl`, writing
`results/bootstrap_cis.json`. Deduplicate first (REVISION_PLAN §8) — bootstrapping over
a dataset with 9 duplicate trials resamples the duplication too and produces intervals
that are subtly too narrow for the affected financial cells.

Depends on nothing outside `numpy` and `scipy`, both already in `requirements.txt`.
