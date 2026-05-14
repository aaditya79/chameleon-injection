# Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems

Research evaluation framework for EMNLP 2026 Industry Track.

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
@inproceedings{pai2026blindspots,
  title     = {Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems},
  author    = {Pai, Aaditya},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
}
```