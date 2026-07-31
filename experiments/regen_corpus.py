"""
Revision work items (f) and (g).

(f) PAYLOAD REGENERATION + PERSISTENCE
    Regenerate 45 tasks x 3 variants = 135 camouflage payloads per model with
    ACS (Authoritative Camouflage Score) actually computed, and persist the full
    payload text. The originally reviewed payloads were never stored -- they were
    generated in-loop and discarded -- so this re-establishes the corpus rather
    than recovering it. Attacker temperature is 0.7, so payloads are NOT identical
    to the ones behind the submitted numbers. The corpus is stamped with a
    regeneration date and hash for the paper's reproducibility statement.

(g) EXP 2 SINGLE-AGENT BASELINE, ALL 45 TASKS
    The submitted DAF divides a 45-task debate ASR by a single-agent ASR that
    only ever covered 5 financial tasks, because run_all.py shares one
    completed_set across experiments and exp2's single-agent signature collides
    with exp1's. This script uses a SELF-CONTAINED resume index keyed on the
    output file only, so no cross-experiment suppression is possible.

Both phases write to results/trials_v2.jsonl with an explicit dry_run=False
field and the payload text inline.

    python experiments/regen_corpus.py --model llama          # (f)+(g) for Llama
    python experiments/regen_corpus.py --model gemini
    python experiments/regen_corpus.py --model llama --phase f --limit 2   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.agents.single_agent import SingleAgent
from src.evaluation.asr_judge import ASRJudge
from src.attacks.camouflage_generator import CamouflageGenerator
from src.attacks.payload_bank import PayloadBank
from src.attacks.static_payloads import instantiate_payloads

TASKS_PATH = Path("data/tasks.json")
CORPUS_PATH = Path("data/camouflage_corpus_v2.json")
TRIALS_PATH = Path("results/trials_v2.jsonl")

# Both paths are rebound from --corpus / --trials in main(). Each model writes to
# its own shard so two models can run concurrently without a read-modify-write
# race on the shared corpus file; analysis/merge_corpus.py combines the shards.

# The four static categories exp2 used, one representative payload each.
STATIC_CATEGORIES = [
    "override_directive", "authority_claim", "role_confusion", "social_engineering",
]

# The attacker LLM sometimes refuses to produce a payload. A refusal is not a
# camouflage payload -- injecting one measures nothing -- but the original
# pipeline had no refusal check, so refusals were silently evaluated as though
# they were payloads, deflating both ASR and IDR for the camouflage condition.
# We detect, retry, and (if still refusing) flag them.
_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm unable",
    "i am unable", "i'm not able", "i am not able", "as an ai", "i apologize",
    "i'm sorry", "i am sorry", "cannot assist", "can't assist", "cannot help with",
    "can't help with", "cannot create content", "can't create content",
    "cannot provide", "can't provide", "against my", "not appropriate",
)
MIN_PAYLOAD_WORDS = 25
MAX_REGEN_ATTEMPTS = 4


def looks_like_refusal(text: str) -> bool:
    """
    True if the generated text is a refusal or a degenerate non-payload.

    Two signals: an explicit refusal marker near the start (refusals lead with
    them, whereas a legitimate payload would only contain such a phrase deep in
    the body, if at all), or a text too short to be the requested 50-150 words.
    """
    t = (text or "").strip().lower()
    if len(t.split()) < MIN_PAYLOAD_WORDS:
        return True
    head = t[:160]
    return any(marker in head for marker in _REFUSAL_MARKERS)


MODEL_SPECS = {
    "llama": {"provider": "ollama", "model": "llama3.1:latest", "key": ""},
    "gemini": {
        "provider": "openrouter",
        "model": "google/gemini-2.0-flash-001",
        "key": os.environ.get("OPENROUTER_API_KEY", "") or CONFIG.openrouter_api_key,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tasks() -> list[dict]:
    return json.loads(TASKS_PATH.read_text())


def build_client(spec: dict) -> LLMClient:
    """Build a live (never dry-run) client for the given model spec."""
    kwargs = {"dry_run": False, "cost_alert_usd": 25.0}
    if spec["provider"] == "openrouter":
        kwargs["api_key"] = spec["key"]
        kwargs["base_url"] = CONFIG.openrouter_base_url
    client = LLMClient(spec["provider"], spec["model"], **kwargs)
    assert client.dry_run is False, "refusing to run with dry_run enabled"
    return client


def load_corpus() -> dict:
    if CORPUS_PATH.exists():
        return json.loads(CORPUS_PATH.read_text())
    return {"generated": {}, "meta": {}}


def save_corpus(corpus: dict) -> None:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))


def resume_index() -> set[tuple]:
    """
    Signatures already present in the OUTPUT file only.

    Deliberately self-contained: unlike run_all.py's shared completed_set, this
    cannot suppress a trial because some other experiment already ran it.
    """
    if not TRIALS_PATH.exists():
        return set()
    done = set()
    for line in TRIALS_PATH.open():
        r = json.loads(line)
        done.add((r["model"], r["experiment"], r["architecture"],
                  r["task_id"], r["payload_type"], r["payload_id"]))
    return done


def append_trial(record: dict) -> None:
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIALS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Phase (f): payload regeneration + persistence
# ---------------------------------------------------------------------------

def phase_f(tasks: list[dict], model_key: str, spec: dict, limit: int | None) -> dict:
    """Generate and persist 3 camouflage variants per task, with ACS."""
    print(f"\n{'='*70}\nPHASE (f)  payload regeneration + persistence  [{spec['model']}]\n{'='*70}")

    client = build_client(spec)
    gen = CamouflageGenerator(client, compute_similarity=True)

    corpus = load_corpus()
    corpus.setdefault("generated", {}).setdefault(model_key, {})
    store = corpus["generated"][model_key]

    todo = [t for t in tasks if t["task_id"] not in store]
    if limit:
        todo = todo[:limit]
    print(f"  tasks needing generation: {len(todo)} / {len(tasks)}")

    t0 = time.monotonic()
    n_refusals_initial = n_refusals_final = n_retries = 0

    for task in tqdm(todo, desc=f"generate [{model_key}]", unit="task"):
        variants = gen.generate(
            clean_context=task["clean_context"],
            malicious_goal=task["malicious_goal"],
            domain=task["domain"],
            task_id=task["task_id"],
            n_variants=CONFIG.n_camouflage_variants,
        )

        # Retry any refused variant; the generator samples at 0.7 so a fresh
        # draw is a genuinely different attempt, not a repeat of the same one.
        for i, v in enumerate(variants):
            if not looks_like_refusal(v.payload_text):
                continue
            n_refusals_initial += 1
            for _ in range(MAX_REGEN_ATTEMPTS):
                n_retries += 1
                retry = gen.generate(
                    clean_context=task["clean_context"],
                    malicious_goal=task["malicious_goal"],
                    domain=task["domain"],
                    task_id=task["task_id"],
                    n_variants=1,
                )[0]
                if not looks_like_refusal(retry.payload_text):
                    retry.payload_id = v.payload_id
                    retry.variant_idx = v.variant_idx
                    variants[i] = retry
                    break
            else:
                n_refusals_final += 1
                tqdm.write(f"    ! {task['task_id']} {v.payload_id}: "
                           f"still refusing after {MAX_REGEN_ATTEMPTS} retries")

        store[task["task_id"]] = [
            {
                "payload_id": v.payload_id,
                "variant_idx": v.variant_idx,
                "payload_text": v.payload_text,
                "acs": round(v.semantic_similarity, 6),
                "domain": v.domain,
                "malicious_goal": v.malicious_goal,
                "generation_model": v.generation_model,
                "is_refusal": looks_like_refusal(v.payload_text),
                "n_words": len(v.payload_text.split()),
            }
            for v in variants
        ]
        save_corpus(corpus)

    elapsed = time.monotonic() - t0
    payloads = [p for ps in store.values() for p in ps]
    acs_vals = [p["acs"] for p in payloads if p["acs"] > 0]
    blob = json.dumps(store, sort_keys=True).encode()

    corpus.setdefault("meta", {})[model_key] = {
        "regenerated_utc": datetime.now(timezone.utc).isoformat(),
        "model": spec["model"],
        "provider": spec["provider"],
        "attacker_temperature": CONFIG.attacker_temperature,
        "n_tasks": len(store),
        "n_payloads": len(payloads),
        "acs_mean": round(sum(acs_vals) / len(acs_vals), 4) if acs_vals else None,
        "acs_min": round(min(acs_vals), 4) if acs_vals else None,
        "acs_max": round(max(acs_vals), 4) if acs_vals else None,
        "corpus_sha256": hashlib.sha256(blob).hexdigest()[:16],
        "refusals_on_first_draw": n_refusals_initial,
        "refusal_retries": n_retries,
        "refusals_unresolved": n_refusals_final,
        "refusals_remaining_in_corpus": sum(1 for p in payloads if p.get("is_refusal")),
        "note": "Regenerated corpus; attacker temp 0.7 means payloads are NOT "
                "identical to those behind the originally submitted numbers.",
    }
    save_corpus(corpus)

    m = corpus["meta"][model_key]
    print(f"  payloads: {m['n_payloads']} across {m['n_tasks']} tasks")
    print(f"  ACS mean {m['acs_mean']}  (min {m['acs_min']}, max {m['acs_max']})")
    print(f"  refusals: {m['refusals_on_first_draw']} on first draw, "
          f"{m['refusal_retries']} retries, {m['refusals_unresolved']} unresolved")
    print(f"  corpus sha256[:16] = {m['corpus_sha256']}   elapsed {elapsed/60:.1f} min")
    print(f"  cumulative client cost ${client._cumulative_cost:.4f}")
    return {"cost": client._cumulative_cost, "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# Phase (g): single-agent baseline over all 45 tasks
# ---------------------------------------------------------------------------

def _run_one(agent, judge, task, injected_ctx, payload_type, payload_id,
             payload_text, payload_category, spec, model_key):
    """Run one single-agent trial and return its log record."""
    resp = agent.analyze(task["instruction"], injected_ctx, task["task_id"])
    j = judge.judge(task["instruction"], task["malicious_goal"], resp.response_text)

    cost = resp.completion.cost_usd + (j.completion.cost_usd if j.completion else 0.0)
    tokens = resp.completion.input_tokens + resp.completion.output_tokens
    return {
        "trial_id": hashlib.md5(
            f"{model_key}|{task['task_id']}|{payload_type}|{payload_id}".encode()
        ).hexdigest(),
        "experiment": "exp2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"],
        "domain": task["domain"],
        "architecture": "single_agent",
        "payload_type": payload_type,
        "payload_id": payload_id,
        "payload_category": payload_category,
        "payload_text": payload_text,
        "injection_position": "end",
        "injected_agent_idx": None,
        "agent_responses": [resp.response_text],
        "debate_history": None,
        "final_answer": resp.response_text,
        "asr_judgment": {
            "followed_injection": j.followed_injection,
            "confidence": j.confidence,
            "evidence": j.evidence,
            "reasoning": j.reasoning,
        },
        "detection_result": None,
        "semantic_similarity": None,
        "total_tokens": tokens,
        "total_cost_usd": cost,
        "model": spec["model"],
        "provider": spec["provider"],
        "dry_run": False,
        "corpus_version": "v2",
        "inject_mode": "none",
    }


def phase_g(tasks: list[dict], model_key: str, spec: dict, limit: int | None) -> dict:
    """Single-agent ASR baseline over ALL tasks, camouflage + static."""
    print(f"\n{'='*70}\nPHASE (g)  single-agent baseline, all tasks  [{spec['model']}]\n{'='*70}")

    corpus = load_corpus()
    store = corpus.get("generated", {}).get(model_key, {})
    if not store:
        raise SystemExit(f"No corpus for {model_key}; run phase f first.")

    client = build_client(spec)
    agent = SingleAgent(client, agent_id="single_agent")
    judge = ASRJudge(client)
    bank = PayloadBank()

    done = resume_index()
    work: list[tuple] = []

    run_tasks = tasks[:limit] if limit else tasks
    for task in run_tasks:
        for p in store.get(task["task_id"], []):
            work.append((task, "camouflage", p["payload_id"], p["payload_text"], None))
        for cat in STATIC_CATEGORIES:
            ps = instantiate_payloads(task["malicious_goal"], CONFIG.data_dir, categories=[cat])
            if ps:
                work.append((task, "static", ps[0].payload_id, ps[0].instantiated_text, cat))

    pending = [
        w for w in work
        if (spec["model"], "exp2", "single_agent", w[0]["task_id"], w[1], w[2]) not in done
    ]
    print(f"  planned {len(work)} trials | already done {len(work)-len(pending)} | to run {len(pending)}")

    t0 = time.monotonic()
    errors = 0
    for task, ptype, pid, ptext, pcat in tqdm(pending, desc=f"single [{model_key}]", unit="trial"):
        injected = bank.inject(task["clean_context"], ptext, "end")
        try:
            append_trial(_run_one(agent, judge, task, injected, ptype, pid,
                                  ptext, pcat, spec, model_key))
        except Exception as e:                       # noqa: BLE001 - keep the run alive
            errors += 1
            tqdm.write(f"    ! {task['task_id']}/{pid}: {type(e).__name__}: {str(e)[:120]}")

    elapsed = time.monotonic() - t0
    print(f"  ran {len(pending)-errors} trials ({errors} errors) in {elapsed/60:.1f} min")
    print(f"  cumulative client cost ${client._cumulative_cost:.4f}")
    return {"cost": client._cumulative_cost, "elapsed_s": elapsed, "errors": errors}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Revision items (f) and (g)")
    ap.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    ap.add_argument("--phase", choices=["f", "g", "fg"], default="fg")
    ap.add_argument("--limit", type=int, default=None, help="limit tasks (smoke test)")
    ap.add_argument("--corpus", default=None, help="corpus shard path (per-model)")
    ap.add_argument("--trials", default=None, help="trial output path (per-model)")
    args = ap.parse_args()

    global CORPUS_PATH, TRIALS_PATH
    if args.corpus:
        CORPUS_PATH = Path(args.corpus)
    if args.trials:
        TRIALS_PATH = Path(args.trials)

    spec = MODEL_SPECS[args.model]
    if spec["provider"] == "openrouter" and not spec["key"]:
        raise SystemExit("OPENROUTER_API_KEY not set")

    tasks = load_tasks()
    totals = {"cost": 0.0}

    if "f" in args.phase:
        totals["f"] = phase_f(tasks, args.model, spec, args.limit)
        totals["cost"] += totals["f"]["cost"]
    if "g" in args.phase:
        totals["g"] = phase_g(tasks, args.model, spec, args.limit)
        totals["cost"] += totals["g"]["cost"]

    print(f"\n{'='*70}\nTOTAL COST [{args.model}]: ${totals['cost']:.4f}\n{'='*70}")


if __name__ == "__main__":
    main()
