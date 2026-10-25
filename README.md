# Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems

https://arxiv.org/abs/2605.22001

Research evaluation framework for evaluating domain-camouflaged 
injection attacks against LLM agent detection systems.

## Overview

This framework evaluates whether **domain-camouflaged injection payloads** evade detection more effectively than canonical static payloads, and whether **multi-agent debate architectures** amplify this blind spot through conformity pressure.

### Core Claims Evaluated

| Claim | Metric | Experiment |
|-------|--------|------------|
| Camouflage evades detection better than static | CDG (Camouflage Detection Gap) | Exp 1, 3 |
| Debate amplifies the camouflage blind spot | DAF (Debate Amplification Factor) | Exp 2 |
| Conformity pressure propagates the injection | CPS (Conformity Pressure Score) | Exp 2 |
| One example per domain partially closes the gap | CDG improvement % | Exp 4 |

### Novel Metrics

- **CDG** — IDR_static_detector_on_static_payloads − IDR_static_detector_on_camouflage_payloads. The primary contribution.
- **DAF** — ASR_debate / ASR_single. > 1 means debate amplifies injection success.
- **CPS** — Fraction of non-injected agents that adopt the injected agent's answer (inject_first mode).
- **ACS** — Cosine similarity between payload and clean context. Validates the generator.

## Project Structure

```
chameleon-injection/
├── src/
│   ├── agents/          # SingleAgent, DebateAgent, DebateOrchestrator
│   ├── attacks/         # CamouflageGenerator, PayloadBank, static_payloads
│   ├── tasks/           # Task loader
│   ├── evaluation/      # ASRJudge, MetricsComputer
│   ├── detection/       # StaticDetector, AugmentedDetector
│   └── utils/           # LLMClient, TrialLogger, CostTracker
├── experiments/
│   ├── exp1_static_vs_camouflage_single.py
│   ├── exp2_single_vs_debate.py
│   ├── exp3_detection_blind_spot.py
│   ├── exp4_cheap_fix.py
│   └── run_all.py
├── analysis/
│   ├── compute_metrics.py
│   ├── generate_tables.py
│   └── generate_plots.py
├── data/
│   ├── tasks.json           # 45 tasks: 15 financial, 15 legal, 15 general
│   ├── static_payloads.json # 20 static payloads across 4 categories
│   └── domains.json
├── results/             # auto-created, gitignored
├── config.py
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dry run (no API calls — validate everything works)

```bash
python experiments/run_all.py --dry-run
```

### 3. Run with Ollama (local, free)

```bash
ollama serve
ollama pull llama3.1
python experiments/run_all.py --no-dry-run
```

### 4. Run with OpenRouter (cloud, free models available)

```bash
export OPENROUTER_API_KEY=your_key_here
# Edit config.py: set agent_provider="openrouter", agent_model="meta-llama/llama-3.1-8b-instruct:free"
python experiments/run_all.py --no-dry-run
```

### 5. Run individual experiments

```bash
python experiments/run_all.py --exp 1          # Exp 1 only
python experiments/run_all.py --exp 1,3        # Exps 1 and 3
python experiments/run_all.py --max-tasks 5    # Quick smoke test
```

### 6. Generate analysis outputs

```bash
python analysis/compute_metrics.py   # CSVs + metrics_full.json
python analysis/generate_tables.py   # LaTeX table files
python analysis/generate_plots.py    # 6 PDF/PNG figures
```

## Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dry_run` | `True` | Use mock responses (no API calls) |
| `agent_provider` | `"ollama"` | LLM backend: ollama, openrouter, openai, anthropic |
| `agent_model` | `"llama3.1"` | Model name for agent calls |
| `n_agents` | `3` | Number of debate participants |
| `n_rounds` | `3` | Debate rounds per trial |
| `n_camouflage_variants` | `3` | Camouflage variants per task |
| `debate_aggregation` | `"majority_vote"` | How to aggregate final debate answer |
| `seed` | `42` | Random seed for reproducibility |

## Data

### Tasks (45 total)
- **Financial** (15): stock analysis, risk assessment, market interpretation
- **Legal** (15): contract review, compliance check, risk flagging
- **General QA** (15): factual QA, summarization, classification (control domain)

### Static Payloads (20)
Four categories, 5 each: Override Directives (score 5), Role Confusion (score 4), Authority Claims (score 3), Social Engineering (score 2).

### Camouflage Payloads (generated)
Generated on-the-fly by `CamouflageGenerator` using domain vocabulary and authority structures from the clean context, with no override markers. 3 variants per task = 135 total.

## Experiments

| Exp | Question | Key Metric |
|-----|----------|------------|
| 1 | Does camouflage evade detection better? | CDG |
| 2 | Does debate amplify the blind spot? | DAF, CPS |
| 3 | Does CDG vary by domain? | CDG per domain |
| 4 | Does one example close the gap? | CDG improvement % |

## Extended Experiments (v2)

These modules extend the framework with the directiveness-ablation, benign-twin, and agent-environment studies. They build on the trial logs produced by the experiments above. For the findings and their interpretation, see the arXiv paper linked at the top of this README, not this section; the descriptions below cover what each module does, not what it concluded. Every module reads payloads and texts by id and does not print payload text.

### Directiveness ablation (`item3b`)

Rewrites each camouflage source into a set of matched arms (directive form, length control, null paraphrase, generic vocabulary, document-specifics removed, compressed) and measures how detection and attack success move across them, paired by source payload.

| File | Role |
|------|------|
| `experiments/item3b_rewrite.py` | Build the rewrite arms for the refusal-screened payloads (rewrites via OpenRouter; LLM judges and ratings), applying the pre-registered acceptance rules |
| `analysis/item3b_data.py` | Load the frozen records by id; compute word counts and specificity with the frozen extractor |
| `analysis/item3b_models.py` | Paired statistics: risk difference with task-cluster BCa intervals, sign-flip permutation, exact McNemar, Holm, Wilson, Firth |
| `analysis/item3b_report.py` | Run the pre-registered contrasts and write the report |
| `analysis/item3b_selection.py` | Descriptive per-arm detection of the original payloads in accepted vs excluded subsets (Fisher exact) |
| `analysis/item3b_a2.py`, `analysis/item3b_a3.py` | Exploratory follow-up arms (declarative goal; asserted outcome), same paired methods |
| `analysis/item3b_firth.R` | Firth logistic fit used by the exploratory specification (`Rscript`) |

### Benign twins (`item3c`)

Generates benign "twin" documents matched to each camouflage payload, screens and selects them, verifies the match, and compares detector false positives on twins against detections on camouflage.

| File | Role |
|------|------|
| `experiments/item3c_generate.py` | Per-arm twin generation, rule plus LLM screening, validity check, and selection of the closest matched twin per slot |
| `experiments/item3c_checks.py` | Manipulation checks (directiveness rating; matching equivalence via task-cluster bootstrap and TOST) |
| `analysis/item3c_data.py` | Assemble twin and re-screened camouflage detection records by id across the detector set |
| `analysis/item3c_caliper.py` | Build the caliper-matched subset and per-item features from text features only |
| `analysis/item3c_analysis.py` | Pre-registered outcomes: twin false-positive rate, camouflage detection, their difference and AUROC with task-cluster BCa intervals, and augmentation cost |
| `analysis/item3c_adjusted.py` | Covariate-adjusted difference via Firth-penalised logistic regression and g-computation |

### Agent-environment study (`item5b`, AgentDojo)

Runs the camouflage and static-template arms inside the AgentDojo banking suite and scores the injected fields offline. These scripts require a separate AgentDojo environment, noted in each file's header; the paid runs are cost-capped and resumable.

| File | Role |
|------|------|
| `experiments/item5b_gen_payloads.py` | Pre-generate and screen the banking camouflage payloads on local Ollama, cached by id |
| `experiments/item5b_regen_gpt4o.py`, `experiments/item5b_unify_gpt4o.py` | Regenerate refused slots with GPT-4o under a live-balance guard; unify both arms to a single generator |
| `experiments/item5b_pilot.py`, `experiments/item5b_pilot_report.py` | Resumable, cost-capped pilot across models, and its report |
| `experiments/item5b_run.py` | Full banking agent matrix (model by defense by arm), resumable and cost-capped |
| `experiments/item5b_detect.py` | Offline detection pass over the injected fields with the detector set (no agent is run) |
| `experiments/item5b_report.py` | Per (model, defense, arm) attack success, clean utility, and ask-before-acting summary |
| `src/attacks/agentdojo_camouflage_attack.py` | AgentDojo adapter registering `camouflage_full` and `camouflage_no_context`; wires AgentDojo plumbing into the generator, which reads payloads by id |
| `src/attacks/agentdojo_static_attack.py` | AgentDojo adapter registering the fixed overt-injection `static` template arm |
| `src/attacks/camouflage_screened.py` | Shared generate, refusal-screen, and cache path for the AgentDojo camouflage arms (local Ollama) |

The item5b scripts use the same `OPENROUTER_API_KEY` environment variable as the Quick Start above for their paid calls.

### Data files

| File | Contents |
|------|----------|
| `data/item3b_payloads.json` | Frozen inputs for the directiveness ablation: control sentences and per-arm rewrite records, keyed by id, with provenance metadata |
| `data/item3b_a2_payloads.json`, `data/item3b_a3_payloads.json` | Inputs for the exploratory follow-up arms (templates or sentences and their source ids) |
| `data/item5b_camouflage_payloads.json` | Screened AgentDojo banking camouflage payloads, keyed by `arm\|injection_task\|field`, each with generator and status |
| `data/mechanism_goals_neutral.json` | Neutral goal set for the mechanism probe |
| `data/mechanism_templates.json` | Slot templates, probe prefix, and cell definitions for the mechanism probe |


## Output Files

After running:

```
results/
├── trials.jsonl              # Every trial record
├── paper_numbers.txt         # Copy-paste numbers for the paper
├── metrics_by_payload_type.csv
├── metrics_by_domain.csv
├── metrics_cdg_by_domain.csv
├── table1_main_results.tex
├── table2_cdg_by_domain.tex
├── table3_debate_dynamics.tex
└── figures/
    ├── fig1_cdg_bar.{pdf,png}
    ├── fig2_daf_comparison.{pdf,png}
    ├── fig3_conformity_propagation.{pdf,png}
    ├── fig4_cdg_before_after.{pdf,png}
    ├── fig5_acs_distribution.{pdf,png}
    └── fig6_asr_vs_acs.{pdf,png}
```

## Reproducibility

- All random operations seeded with `config.seed = 42`
- `temperature = 0.0` for agent/detector/judge calls
- `attacker_temperature = 0.7` for camouflage generation
- Trial records include model name and provider

## Citation

```bibtex
@inproceedings{anonymous2026blindspots,
  title     = {Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks 
               Evade Detection in Multi-Agent LLM Systems},
  author    = {Anonymous},
  booktitle = {Under review at EMNLP 2026},
  year      = {2026},
}
```
