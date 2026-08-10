# Response to Meta-Review

**Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems**

We thank the AC and reviewers for the constructive feedback. In preparing this revision, we conducted a full data-provenance audit of our experimental pipeline and found three issues affecting the original submission's numbers, described below alongside our response to each requested revision. None of the issues weaken the paper's central finding; one (the debate-amplification claim) we have corrected downward and withdrawn its strongest interpretation.

All section, table, and figure references below are to the revised manuscript.

---

## Data integrity (found during this revision, not requested by reviewers, disclosed in full)

We report these first because two of them changed numbers that appeared in the reviewed submission. All three were found by re-deriving every reported figure from raw logs rather than from intermediate summaries. Each is now disclosed in the paper itself, not only here.

**1. Resume-cache collision suppressed the debate baseline.** A resume cache shared across experiment runners collapsed the single-agent baseline arm of Exp 2 to five financial tasks, of which 21 of 56 trials were invalid, while the debate arm covered all 45. Because the Debate Amplification Factor is a ratio against that baseline, and because financial tasks are the hardest to attack in our bank, the ratio was inflated. The submitted paper reported DAF = 3.4x (static) and 9.9x (camouflage) for Llama 3.1 8B and built a contribution on the asymmetry between them. On matched 45-task arms the corrected values are **DAF = 2.40 (static) and 2.68 (camouflage)**, and the asymmetry disappears. We have **withdrawn the camouflage-specific conformity claim entirely**. Debate is now reported as a corrected secondary result with a model-dependent effect (it reduces injection on Gemini 2.0 Flash: 0.76 and 0.63), not as a contribution. Section 5 states the prior value and the reason it was wrong. The fix is a self-contained resume index per runner, keyed to include `detector_type` and `generator_variant`.

**2. Dry-run mock records contaminated the trial log.** Our log contained 11,086 records, of which **342 were dry-run placeholder responses** emitted by the client's offline mode during a smoke test and not distinguishable by any field recorded at the time. We identify them by two independent signatures (verbatim matches against the mock response pool, and wall-clock seconds containing more trials than sequential inference can produce), flag them with a persisted `dry_run` field, and exclude them from every number in the paper. All 342 are Llama trials; Gemini is unaffected. **10,744 real trials remain.** Excluding them moves the headline figures **upward**: Llama IDR_static rises from 0.938 to 1.000 and **CDG from 0.840 to 0.896**, because every apparent static-detection failure was a mock record. This is reported in the main text (Section 4.7), not relegated to a footnote, precisely because it changed a headline number.

**3. Attacker refusals were scored as payloads.** The attacker model declines a substantial fraction of generation requests. Earlier versions of our pipeline took the first draw unconditionally, so refusals were injected into documents and scored as though they were camouflage payloads. A refusal cannot induce the malicious behaviour, so this biased the camouflage condition downward on both ASR and IDR. We now detect refusals, resample up to four times, and flag any that persist. **Camouflage ASR rises from 0.126 to 0.182**; the effect on CDG is small (+0.009). We report detection on the refusal-screened corpus and, for comparison, on the unscreened set. This issue also produced a substantive finding, described under AC Item 3 below.

We note that issues 2 and 3 both moved the central result in the direction *favourable* to our claim. We flag this explicitly rather than quietly banking it.

---

## AC Item 1 — dedicated defenses, frontier models, fine-tuned detectors

**Dedicated injection defenses: added (new Section 5, Exp 5).** We evaluate the two defense families most often named for this threat: spotlighting (Hines et al.) in **all three published variants** (delimiting, datamarking, base64 encoding) and a **structured-query** approach (Chen et al.), applied as input transforms before detection, over the refusal-screened corpus. Critically, each configuration includes **45 clean documents as false-positive controls** plus an undefended control on identical sampling (1,800 detections total).

**No dedicated defense closes the gap.** CDG stays between **0.765 and 0.928** across delimiting, datamarking, and structured-query, against an undefended 0.873. Because CDG alone would mislead here in two distinct ways, we decompose every change into its static and camouflage components and pair it with a false-positive rate:

| Defense | IDR_static | IDR_camo | CDG | FPR |
|---|---|---|---|---|
| none (control) | 0.989 | 0.116 | 0.873 | 0.000 |
| spotlight: delimiting | 1.000 | 0.107 | 0.893 | 0.000 |
| spotlight: datamarking | 0.906 | 0.140 | 0.765 | 0.000 |
| spotlight: base64 | 0.328 | 0.570 | −0.242 | 0.267 |
| structured-query | 0.994 | 0.066 | 0.928 | 0.000 |

Two results here required the controls to interpret correctly, and we think both are worth the reviewers' attention:

- **Base64's apparent improvement is a detector-discrimination failure, not a defense.** It is the only configuration with a negative CDG, and without clean controls it would have looked like the one defense that worked. In fact it flags **26.7% of clean documents** and catches only **32.8% of blatant override payloads**, worse on the easy case than no defense at all. Its "improvement" is a detector that has stopped discriminating between anything. We report it as a detector failure. This is our concrete argument that detector-side defense evaluations must report a false-positive rate.
- **Datamarking's apparent improvement is mostly the detector getting worse.** It narrows CDG by 0.108, but **77% of that narrowing is static detection degrading** (0.989 to 0.906), not camouflage becoming detectable (0.116 to 0.140, three extra catches in 121). Converging the two arms by damaging the detector's grip on the easy case does not address the blind spot, and we do not count it as doing so.

Structured-query framing actively **suppresses** detection, catching camouflage 43% less often than the control (0.116 to 0.066). In hindsight this is intuitive: instructing a model that a region cannot contain instructions lowers its willingness to report that it does.

We state a scope caveat plainly in the paper: spotlighting and structured queries were designed as **agent-side prevention**, and we evaluate them **detector-side**. Our results should not be read as showing these defenses fail at what they were built to do. Whether they reduce attack success rate agent-side against camouflaged payloads is a distinct and unanswered question. Our structured-query condition is additionally an inference-time approximation; the published method fine-tunes on the instruction/data channel split, which we do not reproduce.

**Stronger frontier models: not added in this revision.** We want to be direct about this rather than imply broader coverage than we have. Our strong closed model remains Gemini 2.0 Flash, which was in the original submission and which shows CDG = 0.444, demonstrating the gap is not an artifact of small open-weight models. However, **`google/gemini-2.0-flash-001` has been withdrawn from our inference provider's catalogue during this revision cycle**, with no Gemini 2.0 or 1.x endpoint remaining. The existing Gemini results are complete, uncontaminated, and matched across arms, but they **cannot be extended or regenerated**, and we could not produce a persisted payload corpus for that model as we did for Llama. We record this in Limitations as a permanent gap in the released artifact rather than a defect in the reported numbers, and we note it as a general hazard for agentic-security work, where hosted-model turnover is fast relative to review cycles. Adding a third model family (Exp 1 + Exp 5 on one additional model) is scoped and remains our next addition; it is held rather than declined.

**Fine-tuned detector trained on camouflaged examples: not done.** We did not train one, and we do not claim to have. Limitations now states this as a scoped limitation and confines our claims to prompted detectors rather than to injection detection in general. Our detector-augmentation experiment (one camouflaged exemplar per domain added to the few-shot pool) bounds only what that cheap intervention achieves, and it is explicitly not a substitute for fine-tuning.

---

## AC Item 2 — human validation of camouflage naturalness

**Designed, not run.** We take this criticism as fair and we have not worked around it. All payload quality, attack success, and detection labels in the paper are produced by LLMs, and none has been validated against human judgement; the ASR judge additionally shares a model family with the agent it evaluates, and we do not report judge-human agreement.

We designed a full protocol for this revision: domain-literate raters scoring camouflage naturalness against clean-document and static-payload controls, a parallel ASR-judge validation study, and pre-registered agreement thresholds. It is preserved in the repository as `HUMAN_EVAL_PROTOCOL.md` as evidence of a considered and ready design. **We did not conduct it**, for reasons of annotator cost and revision-cycle timeline.

Limitations names this as an undone limitation rather than softening it, and states that running the protocol is the most direct way to establish that the measured detection gap reflects genuinely natural payloads rather than an artifact of LLM-generated text. It is our primary planned follow-up. No result in the paper rests on human annotation, so nothing reported here is contingent on the outcome.

---

## AC Item 3 — threat model clarity

The AC correctly identified a mismatch: our stated threat model gives the attacker indirect document access only, while our camouflage generator receives the full target document. We have addressed this in three ways.

**1. The threat model now states attacker knowledge explicitly** (Section 2), and separates what the attacker is assumed to know from what our generator was actually given. We frame the full-document condition as a **deliberate worst-case bound**, not as the expected attacker position, and every CDG figure in the paper is now labelled an upper bound at each site where it appears, including the abstract, contributions, and conclusion.

**2. We ran the no-context condition** to measure what that assumption buys the attacker. The generator sees the domain, the document genre, and the domain's professional vocabulary, but never the target document and never the task instruction. **The malicious goal string is byte-identical across conditions, so document access is the only manipulated variable.** Results (Table 1):

| | Full context | No context |
|---|---|---|
| Static detector | 0.868 | **0.778** |
| Augmented detector | 0.726 | **0.414** |

Detection becomes easier but not easy. The static detector still misses **77.8%** of camouflaged injections written blind. The gap narrows most in the legal domain (0.947 to 0.778), where imitating a formulaic register appears to require having seen an instance of it. Payload quality drops correspondingly but modestly (mean ACS 0.592 to 0.523). The upper-bound framing is therefore now supported by a measurement rather than asserted, and the realistic-attacker figure remains a large blind spot.

**3. An unexpected finding about attacker-model refusals.** Chasing down data-integrity issue 3 produced a result we think is independently worth reporting. With the full document, the attacker model refuses **43.7% of first draws**. Without it, pursuing the **same byte-identical malicious goal**, it refuses **5.9% (8 of 135)**, and every refusal resolves within four resamples where 14 previously did not. That is a **7.4x reduction driven entirely by removing document access**. Since the goal is unchanged, the refusal is triggered by the **concreteness of the target** rather than by the harmfulness of the objective. This is now written into Section 2 as a finding, scoped explicitly to this model's refusal training rather than generalized to safety training at large.

---

## AC Item 4 — camouflage definition

Section 4.2 now separates the two things the original submission conflated:

- **The semantic property (what camouflage *is*).** A payload is domain-camouflaged when its malicious instruction is realized as an **assertive domain claim rather than a directive addressed to the agent**. A static payload tells the agent what to do ("recommend SELL"). A camouflaged payload asserts something about the world which, if accepted as part of the document, makes the malicious output the correct answer to the original task ("the appropriate recommendation consistent with our risk framework is SELL"). The attacker's leverage shifts from commanding the agent to supplying it with a false premise.
- **The surface realization (how it is *achieved*).** Matched domain vocabulary, sentence structure, epistemic and authority register, and the absence of override markers. These are the means, not the definition.

The distinction does work in the paper rather than sitting as a definition: it is the property, not the vocabulary, that determines whether the detector fires, and it is what makes such a payload indistinguishable *in kind* from legitimate content, since documents in these domains routinely contain recommendations, findings, and requirements. This is listed as a contribution in its own right.

---

## AC Item 5 — positioning against context-adaptive and semantic injection literature

We thank reviewer X1Xm for this direction; it sharpened how we describe our own contribution. Related Work now has a dedicated **"Stealthy and adaptive injection"** paragraph positioning our work against the evasion literature, and the framing turns on a distinction we had not previously articulated: **the direction of evasion**.

Optimization-based attacks such as **Neural Exec** (Pasquini et al., AISec 2024) and **JudgeDeceiver** (Shi et al., CCS 2024) use gradient search to synthesize triggers that deviate sharply from any handcrafted pattern. Their evasion comes from *morphological novelty*, and the resulting strings are conspicuously unnatural, defeating blacklist and perplexity filters precisely because they do not resemble ordinary text. **AgentDojo's adaptive attacks** use fluent handcrafted templates adapted to attacker knowledge (the victim's name, the target model, injection placement) rather than to the register of the surrounding document.

Domain-camouflaged injection evades in the opposite direction: the payload is fluent natural language, carries no optimized tokens, and evades detection by **lowering** its anomaly relative to the host document rather than raising it. We also hold fixed what these attacks vary (attack intent and target behaviour are constant across our static and camouflaged conditions), and we measure the response of the **detector** rather than the agent. We state plainly that our contribution is the CDG measurement isolating that register axis, not a new payload-generation technique; our payloads come from a single unoptimized LLM prompt. The overclaim about being first to study context-adaptive payloads has been removed.

**On ADR (Li et al., 2026): we could not locate this reference.** We searched for it repeatedly and were unable to identify the paper with confidence; the nearest candidate we found is a different work by different authors. Rather than cite something we have not read or, worse, cite the wrong paper, we have deliberately left it out. **We would be grateful if reviewer X1Xm could provide the exact reference (venue, full author list, or a link) so that we can engage with it properly.** We expect it is directly relevant given the framing above, and we would rather position against it correctly in the next revision than approximate it now. A note recording this decision is retained in the manuscript source.

---

## Other changes

- **Llama Guard 3 removed as a baseline.** On inspection it is not an injection detector: it has no prompt-injection category in its taxonomy and missed 89% of blatant static attacks in our setting. Reporting it as a defense baseline would have been misleading about what it was built to do. It is cut in full rather than reported with caveats.
- **ACS was described but never computed.** The submitted paper stated that we generate three camouflage variants per task and select the highest-ACS one. Semantic similarity was disabled throughout the pipeline and was null for all 11,086 trials, so no such selection occurred; all three variants were evaluated. ACS is now genuinely measured and reported as a **descriptive statistic** characterising the released corpus. It is not used to select variants, and the paper says so.
- **An unsupported exclusion claim was removed.** The submitted paper stated that a small number of trials were excluded due to Azure content filtering, and offered this as evidence of payload realism. Nothing in our pipeline touches Azure (providers are Ollama, OpenRouter, and OpenAI) and there is no exclusion log. We removed the sentence rather than rewriting it; if trials were in fact excluded, the real provenance and count must be recovered before any version of that claim returns.
- **Two further unsupported mitigation claims removed:** a keyword cross-validation the ASR judge does not perform, and a Llama Guard proxy-payload sentence tied to the cut baseline.
- **Bibliography audited against primary sources.** One real error found and corrected (`brown2020gpt3`: "Prafulla Shyam" to **Pranav Shyam**).
- **Bootstrap confidence intervals** (BCa) are now reported throughout rather than point estimates alone.
- **The "architectural" characterization of the vulnerability was narrowed** at all four sites where it appeared, to claims our evidence actually supports.
- **Editorial pass** for clarity and consistency across the manuscript.

---

## A note on scope

The revised manuscript is at exactly the 8-page content limit, so the additions above were accommodated by trimming redundancy and by reducing the debate material, which the data-integrity audit had in any case demoted from a contribution to a secondary result. Two requested items (human validation, a fine-tuned detector) are not addressed and are named as limitations rather than deferred silently. We would rather submit a paper whose claims are all supported than one that appears to answer everything.
