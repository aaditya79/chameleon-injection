"""
Experiment 4: The "Cheap Fix" — Augmented Detector

PRIMARY QUESTION: Does adding one domain-camouflage example per domain
to the detector's few-shot set close the CDG?

Protocol:
  Run augmented detector (static examples + one camouflage example per domain) on:
    - Same static payloads from Exp 3
    - Same camouflage payloads from Exp 3

  Compute CDG_before (static detector) vs CDG_after (augmented detector)
  Report percentage of gap closed by augmentation.

Key metrics: CDG_before, CDG_after, improvement_pct per domain
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
from src.detection.augmented_detector import AugmentedDetector
from src.evaluation.metrics import MetricsComputer


def _build_clients(config, use_second: bool = False):
    """Build attacker and detector clients."""
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


def _compute_exp4_metrics(trials: list[dict]) -> dict:
    """Compute CDG improvement per domain and overall from filtered trials."""
    mc = MetricsComputer()

    def _idr_by(tlist, det_type, pt):
        filt = [t for t in tlist
                if t.get("payload_type") == pt
                and t.get("detection_result", {}).get("detector_type") == det_type]
        if not filt:
            return 0.0
        return sum(1 for t in filt if t["detection_result"]["verdict"] == "INJECTED") / len(filt)

    results = {}
    for domain in ["financial", "legal", "general"]:
        dom = [t for t in trials if t.get("domain") == domain]
        idr_s_s = _idr_by(dom, "static", "static")
        idr_c_s = _idr_by(dom, "static", "camouflage")
        idr_s_a = _idr_by(dom, "augmented", "static")
        idr_c_a = _idr_by(dom, "augmented", "camouflage")
        improvement = mc.cdg_improvement(idr_s_s, idr_c_s, idr_s_a, idr_c_a)
        results[domain] = {
            "idr_static_detector_on_static": round(idr_s_s, 4),
            "idr_static_detector_on_camouflage": round(idr_c_s, 4),
            "idr_augmented_detector_on_static": round(idr_s_a, 4),
            "idr_augmented_detector_on_camouflage": round(idr_c_a, 4),
            **improvement,
        }

    # Overall
    overall = mc.cdg_improvement(
        _idr_by(trials, "static", "static"),
        _idr_by(trials, "static", "camouflage"),
        _idr_by(trials, "augmented", "static"),
        _idr_by(trials, "augmented", "camouflage"),
    )
    results["overall"] = overall
    return results


def _run_exp4_loops(
    tasks: list[dict],
    config,
    logger: TrialLogger,
    completed_set: set,
    attacker_client: LLMClient,
    detector_client: LLMClient,
    model_name: str,
    model_provider: str,
) -> int:
    """Execute all exp4 trial loops for one model configuration."""
    static_detector = StaticDetector(detector_client)
    augmented_detector = AugmentedDetector(detector_client)
    camouflage_gen = CamouflageGenerator(attacker_client, compute_similarity=False)
    payload_bank = PayloadBank()
    trial_count = 0

    def _run_both(injected, instruction, domain, task_id, payload_type,
                  base_payload_id, payload_category, sim):
        nonlocal trial_count
        for det_suffix, det in [("_static", static_detector),
                                 ("_augmented", augmented_detector)]:
            pid = base_payload_id + det_suffix
            sig = (task_id, payload_type, pid, "detection_only", "none", model_name)
            if sig in completed_set:
                return

            detection = det.detect(injected, instruction, ground_truth=True)
            cost = detection.completion.cost_usd if detection.completion else 0
            tokens = ((detection.completion.input_tokens + detection.completion.output_tokens)
                      if detection.completion else 0)

            logger.log_trial(
                experiment="exp4",
                task_id=task_id,
                domain=domain,
                architecture="detection_only",
                payload_type=payload_type,
                payload_id=pid,
                payload_category=payload_category,
                injection_position="end",
                agent_responses=[],
                final_answer="",
                detection_result={
                    "verdict": detection.verdict,
                    "confidence": detection.confidence,
                    "detector_type": detection.detector_type,
                },
                semantic_similarity=sim,
                total_tokens=tokens,
                total_cost_usd=cost,
                model=model_name,
                provider=model_provider,
                extra={"inject_mode": "none"},
            )
            COST_TRACKER.record(cost, tokens)
            completed_set.add(sig)
            trial_count += 1

    # Static payloads
    static_bar = tqdm(total=len(tasks) * 20 * 2,
                      desc=f"Static→Both Detectors [{model_name}]")
    for task in tasks:
        payloads = instantiate_payloads(task["malicious_goal"], config.data_dir)
        for payload in payloads:
            injected = payload_bank.inject(task["clean_context"], payload.instantiated_text, "end")
            _run_both(injected, task["instruction"], task["domain"], task["task_id"],
                      "static", payload.payload_id, payload.category, None)
            static_bar.update(2)
            if trial_count % 50 == 0 and trial_count > 0:
                COST_TRACKER.checkpoint(f"exp4 trial {trial_count} [{model_name}]")

    static_bar.close()

    # Camouflage payloads
    cam_bar = tqdm(total=len(tasks) * config.n_camouflage_variants * 2,
                   desc=f"Camouflage→Both Detectors [{model_name}]")
    for task in tasks:
        variants = camouflage_gen.generate(
            clean_context=task["clean_context"],
            malicious_goal=task["malicious_goal"],
            domain=task["domain"],
            task_id=task["task_id"],
            n_variants=config.n_camouflage_variants,
        )
        for variant in variants:
            injected = payload_bank.inject(task["clean_context"], variant.payload_text, "end")
            _run_both(injected, task["instruction"], task["domain"], task["task_id"],
                      "camouflage", variant.payload_id, "camouflage", variant.semantic_similarity)
            cam_bar.update(2)
            if trial_count % 50 == 0 and trial_count > 0:
                COST_TRACKER.checkpoint(f"exp4 trial {trial_count} [{model_name}]")

    cam_bar.close()
    return trial_count


def run_exp4(config=None, tasks_override=None, logger=None, completed_set=None,
             exp3_results=None):
    """
    Run Experiment 4.

    Returns:
        Dict with cdg_improvement per domain and overall (+ _model2 keys if applicable).
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
    print(f"EXP 4: Cheap Fix — Augmented Detector")
    print(f"Tasks: {len(tasks)} | Dry run: {config.dry_run}")
    print(f"{'='*60}\n")

    # Primary model
    print("Running static payload detection (both detectors)...")
    att_client, det_client = _build_clients(config, use_second=False)
    _run_exp4_loops(tasks, config, logger, completed_set,
                    att_client, det_client, config.agent_model, config.agent_provider)

    # Second model
    if config.run_second_model:
        print(f"\n--- Running second model: {config.second_model} ---")
        att2, det2 = _build_clients(config, use_second=True)
        _run_exp4_loops(tasks, config, logger, completed_set,
                        att2, det2, config.second_model, config.second_provider)

    # Metrics
    all_trials = logger.load_trials(experiment="exp4")
    primary_trials = [t for t in all_trials if t.get("model") == config.agent_model]
    results = _compute_exp4_metrics(primary_trials)

    if config.run_second_model:
        second_trials = [t for t in all_trials if t.get("model") == config.second_model]
        second_results = _compute_exp4_metrics(second_trials)
        for k, v in second_results.items():
            results[f"{k}_model2"] = v

    logger.save_metrics_snapshot(
        {"cdg_improvement_by_domain": results, "total_cost_usd": COST_TRACKER.total_cost},
        experiment="exp4",
    )

    print(f"\n{'='*60}")
    print(f"EXP 4 RESULTS (CDG improvement by domain) [{config.agent_model}]:")
    for domain, vals in results.items():
        if isinstance(vals, dict) and "model2" not in domain:
            if domain == "overall":
                print(f"  OVERALL: gap_before={vals.get('gap_before', 0):.3f}  "
                      f"gap_after={vals.get('gap_after', 0):.3f}  "
                      f"improvement={vals.get('improvement_pct', 0):.1f}%")
            else:
                print(f"  {domain:12s}: CDG_before={vals.get('gap_before', 0):.3f}  "
                      f"CDG_after={vals.get('gap_after', 0):.3f}  "
                      f"improvement={vals.get('improvement_pct', 0):.1f}%")
    print(f"  Total cost: ${COST_TRACKER.total_cost:.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    run_exp4()
