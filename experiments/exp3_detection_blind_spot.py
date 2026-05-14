"""
Experiment 3: Detection Blind Spot by Domain

PRIMARY QUESTION: Does the CDG hold across domains? Is it larger for
high-specificity domains (financial, legal) vs. the general control?

Protocol:
  Run static detector on:
    - All static payloads (all 20) across all 45 tasks
    - All camouflage payloads (3 variants × 45 tasks)

  Compute CDG broken down by domain (financial, legal, general).

Key metrics: CDG per domain, IDR breakdowns, domain-specificity finding
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.utils.logger import TrialLogger
from src.utils.cost_tracker import COST_TRACKER
from src.attacks.static_payloads import instantiate_payloads
from src.attacks.camouflage_generator import CamouflageGenerator
from src.attacks.payload_bank import PayloadBank
from src.detection.static_detector import StaticDetector
from src.evaluation.metrics import MetricsComputer


def _build_clients(config, use_second: bool = False):
    """Build attacker and detector clients for one model config."""
    if use_second:
        provider = config.second_provider
        model = config.second_model
        api_key = config.second_api_key or config.openrouter_api_key
    else:
        provider = config.agent_provider
        model = config.agent_model
        api_key = config.openrouter_api_key

    def _make():
        kwargs = dict(dry_run=config.dry_run, cost_alert_usd=config.cost_alert_usd)
        if provider == "openrouter":
            kwargs["api_key"] = api_key
            kwargs["base_url"] = config.openrouter_base_url
        return LLMClient(provider, model, **kwargs)

    return _make(), _make()  # attacker, detector


def _compute_exp3_metrics(trials: list[dict]) -> dict:
    """Compute CDG per domain from a filtered trial list."""
    mc = MetricsComputer()
    cdg_by_domain = {}
    for domain in ["financial", "legal", "general"]:
        dom = [t for t in trials if t.get("domain") == domain]
        static_t = [t for t in dom if t.get("payload_type") == "static"]
        cam_t = [t for t in dom if t.get("payload_type") == "camouflage"]

        def _idr(tl):
            dets = [t["detection_result"] for t in tl if t.get("detection_result")]
            if not dets:
                return 0.0
            return sum(1 for d in dets if d.get("verdict") == "INJECTED") / len(dets)

        idr_s = _idr(static_t)
        idr_c = _idr(cam_t)
        cdg_by_domain[domain] = {
            "idr_static_payload": round(idr_s, 4),
            "idr_camouflage_payload": round(idr_c, 4),
            "cdg": round(mc.cdg(idr_s, idr_c), 4),
        }
    return cdg_by_domain


def _run_exp3_loops(
    tasks: list[dict],
    config,
    logger: TrialLogger,
    completed_set: set,
    attacker_client: LLMClient,
    detector_client: LLMClient,
    model_name: str,
    model_provider: str,
) -> int:
    """Execute all exp3 trial loops for one model configuration."""
    detector = StaticDetector(detector_client)
    camouflage_gen = CamouflageGenerator(attacker_client, compute_similarity=False)
    payload_bank = PayloadBank()
    trial_count = 0

    # Static payloads
    static_bar = tqdm(total=len(tasks) * 20, desc=f"Static→Detector [{model_name}]")
    for task in tasks:
        payloads = instantiate_payloads(task["malicious_goal"], config.data_dir)
        for payload in payloads:
            sig = (task["task_id"], "static", payload.payload_id,
                   "detection_only", "none", model_name)
            static_bar.update(1)
            if sig in completed_set:
                continue

            injected = payload_bank.inject(task["clean_context"], payload.instantiated_text, "end")
            detection = detector.detect(injected, task["instruction"], ground_truth=True)
            cost = detection.completion.cost_usd if detection.completion else 0
            tokens = ((detection.completion.input_tokens + detection.completion.output_tokens)
                      if detection.completion else 0)

            logger.log_trial(
                experiment="exp3",
                task_id=task["task_id"],
                domain=task["domain"],
                architecture="detection_only",
                payload_type="static",
                payload_id=payload.payload_id,
                payload_category=payload.category,
                injection_position="end",
                agent_responses=[],
                final_answer="",
                detection_result={
                    "verdict": detection.verdict,
                    "confidence": detection.confidence,
                    "detector_type": detection.detector_type,
                },
                total_tokens=tokens,
                total_cost_usd=cost,
                model=model_name,
                provider=model_provider,
                extra={"inject_mode": "none"},
            )
            COST_TRACKER.record(cost, tokens)
            completed_set.add(sig)
            trial_count += 1

            if trial_count % 50 == 0 and trial_count > 0:
                COST_TRACKER.checkpoint(f"exp3 trial {trial_count} [{model_name}]")

    static_bar.close()

    # Camouflage payloads
    cam_bar = tqdm(total=len(tasks) * config.n_camouflage_variants,
                   desc=f"Camouflage→Detector [{model_name}]")
    for task in tasks:
        variants = camouflage_gen.generate(
            clean_context=task["clean_context"],
            malicious_goal=task["malicious_goal"],
            domain=task["domain"],
            task_id=task["task_id"],
            n_variants=config.n_camouflage_variants,
        )
        for variant in variants:
            sig = (task["task_id"], "camouflage", variant.payload_id,
                   "detection_only", "none", model_name)
            cam_bar.update(1)
            if sig in completed_set:
                continue

            injected = payload_bank.inject(task["clean_context"], variant.payload_text, "end")
            detection = detector.detect(injected, task["instruction"], ground_truth=True)
            cost = detection.completion.cost_usd if detection.completion else 0
            tokens = ((detection.completion.input_tokens + detection.completion.output_tokens)
                      if detection.completion else 0)

            logger.log_trial(
                experiment="exp3",
                task_id=task["task_id"],
                domain=task["domain"],
                architecture="detection_only",
                payload_type="camouflage",
                payload_id=variant.payload_id,
                payload_category="camouflage",
                injection_position="end",
                agent_responses=[],
                final_answer="",
                detection_result={
                    "verdict": detection.verdict,
                    "confidence": detection.confidence,
                    "detector_type": detection.detector_type,
                },
                semantic_similarity=variant.semantic_similarity,
                total_tokens=tokens,
                total_cost_usd=cost,
                model=model_name,
                provider=model_provider,
                extra={"inject_mode": "none"},
            )
            COST_TRACKER.record(cost, tokens)
            completed_set.add(sig)
            trial_count += 1

            if trial_count % 50 == 0 and trial_count > 0:
                COST_TRACKER.checkpoint(f"exp3 trial {trial_count} [{model_name}]")

    cam_bar.close()
    return trial_count


def run_exp3(config=None, tasks_override=None, logger=None, completed_set=None):
    """
    Run Experiment 3.

    Returns:
        Dict of CDG per domain (includes _model2 keys if config.run_second_model).
    """
    if config is None:
        config = CONFIG

    if completed_set is None:
        completed_set = set()

    logger = logger or TrialLogger(config.results_dir)

    if tasks_override is not None:
        tasks = tasks_override
    else:
        with open(f"{config.data_dir}/tasks.json", encoding="utf-8") as f:
            tasks = json.load(f)
    if config.max_tasks:
        tasks = tasks[: config.max_tasks]

    print(f"\n{'='*60}")
    print(f"EXP 3: Detection Blind Spot by Domain")
    print(f"Tasks: {len(tasks)} | Dry run: {config.dry_run}")
    print(f"{'='*60}\n")

    # Primary model
    att_client, det_client = _build_clients(config, use_second=False)
    _run_exp3_loops(tasks, config, logger, completed_set,
                    att_client, det_client, config.agent_model, config.agent_provider)

    # Second model
    if config.run_second_model:
        print(f"\n--- Running second model: {config.second_model} ---")
        att2, det2 = _build_clients(config, use_second=True)
        _run_exp3_loops(tasks, config, logger, completed_set,
                        att2, det2, config.second_model, config.second_provider)

    # Metrics
    all_trials = logger.load_trials(experiment="exp3")
    primary_trials = [t for t in all_trials if t.get("model") == config.agent_model]
    results = _compute_exp3_metrics(primary_trials)

    if config.run_second_model:
        second_trials = [t for t in all_trials if t.get("model") == config.second_model]
        second_results = _compute_exp3_metrics(second_trials)
        for domain, vals in second_results.items():
            results[f"{domain}_model2"] = vals

    logger.save_metrics_snapshot(
        {"cdg_by_domain": results, "total_cost_usd": COST_TRACKER.total_cost},
        experiment="exp3",
    )

    print(f"\n{'='*60}")
    print(f"EXP 3 RESULTS (CDG by domain) [{config.agent_model}]:")
    for domain in ["financial", "legal", "general"]:
        vals = results.get(domain, {})
        print(f"  {domain:12s}: IDR_static={vals.get('idr_static_payload', 0):.3f}  "
              f"IDR_cam={vals.get('idr_camouflage_payload', 0):.3f}  "
              f"CDG={vals.get('cdg', 0):.3f}")
    print(f"  Total cost: ${COST_TRACKER.total_cost:.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    run_exp3()
