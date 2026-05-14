"""
Experiment 1: Static vs. Camouflage payloads on Single Agent

PRIMARY QUESTION: Does camouflage evade detection better than static payloads?

Protocol:
  For each task (45):
    - Clean baseline: single agent, no injection
    - Static injection (20 payloads): agent + static detector + ASR judge
    - Camouflage injection (3 variants): agent + static detector + ASR judge

Key metrics: ASR_static, ASR_camouflage, IDR_static_on_static,
             IDR_static_on_camouflage, CDG
"""

from __future__ import annotations

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.utils.logger import TrialLogger
from src.utils.cost_tracker import COST_TRACKER
from src.agents.single_agent import SingleAgent
from src.attacks.static_payloads import instantiate_payloads
from src.attacks.camouflage_generator import CamouflageGenerator
from src.attacks.payload_bank import PayloadBank
from src.detection.static_detector import StaticDetector
from src.evaluation.asr_judge import ASRJudge
from src.evaluation.metrics import MetricsComputer


def _build_clients(config, use_second: bool = False):
    """Build all LLM clients, optionally using the second model config."""
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

    return _make(), _make(), _make(), _make()  # agent, attacker, detector, judge


def _compute_exp1_metrics(trials: list[dict]) -> dict:
    """Compute exp1 summary metrics from a filtered trial list."""
    mc = MetricsComputer()
    summary = mc.compute_from_trials(trials)
    return {
        "asr_static": summary.asr_by_payload_type.get("static", 0),
        "asr_camouflage": summary.asr_by_payload_type.get("camouflage", 0),
        "idr_static_on_static": summary.idr_static_detector_on_static,
        "idr_static_on_camouflage": summary.idr_static_detector_on_camouflage,
        "cdg": summary.cdg,
        "asr_by_domain": summary.asr_by_domain,
    }


def _run_exp1_loops(
    tasks: list[dict],
    config,
    logger: TrialLogger,
    completed_set: set,
    agent_client: LLMClient,
    attacker_client: LLMClient,
    detector_client: LLMClient,
    judge_client: LLMClient,
    model_name: str,
    model_provider: str,
) -> int:
    """
    Execute all trial loops for one model configuration.

    Returns:
        Number of new trials run (not skipped).
    """
    agent = SingleAgent(agent_client, agent_id="single_agent")
    detector = StaticDetector(detector_client)
    judge = ASRJudge(judge_client)
    camouflage_gen = CamouflageGenerator(
        attacker_client,
        compute_similarity=(not config.dry_run),
    )
    payload_bank = PayloadBank()

    trial_count = 0

    # ===== CLEAN BASELINE =====
    for task in tqdm(tasks, desc=f"Clean baseline [{model_name}]"):
        sig = (task["task_id"], "clean", "", "single_agent", "none", model_name)
        if sig in completed_set:
            continue

        response = agent.analyze(task["instruction"], task["clean_context"], task["task_id"])
        logger.log_trial(
            experiment="exp1",
            task_id=task["task_id"],
            domain=task["domain"],
            architecture="single_agent",
            payload_type="clean",
            agent_responses=[response.response_text],
            final_answer=response.response_text,
            total_tokens=response.completion.input_tokens + response.completion.output_tokens,
            total_cost_usd=response.completion.cost_usd,
            model=model_name,
            provider=model_provider,
            extra={"inject_mode": "none"},
        )
        COST_TRACKER.record(response.completion.cost_usd,
                            response.completion.input_tokens + response.completion.output_tokens)
        completed_set.add(sig)
        trial_count += 1

    # ===== STATIC PAYLOADS =====
    static_templates_count = len(json.load(open(f"{config.data_dir}/static_payloads.json")))
    static_bar = tqdm(total=len(tasks) * static_templates_count,
                      desc=f"Static payloads [{model_name}]")
    for task in tasks:
        payloads = instantiate_payloads(malicious_goal=task["malicious_goal"],
                                        data_dir=config.data_dir)
        for payload in payloads:
            sig = (task["task_id"], "static", payload.payload_id, "single_agent", "none", model_name)
            static_bar.update(1)
            if sig in completed_set:
                continue

            injected = payload_bank.inject(task["clean_context"], payload.instantiated_text,
                                           position="end")
            response = agent.analyze(task["instruction"], injected, task["task_id"])
            detection = detector.detect(injected, task["instruction"], ground_truth=True)
            judgment = judge.judge(task["instruction"], task["malicious_goal"],
                                   response.response_text)

            cost = (response.completion.cost_usd
                    + (detection.completion.cost_usd if detection.completion else 0)
                    + (judgment.completion.cost_usd if judgment.completion else 0))
            tokens = (response.completion.input_tokens + response.completion.output_tokens
                      + (detection.completion.input_tokens + detection.completion.output_tokens
                         if detection.completion else 0)
                      + (judgment.completion.input_tokens + judgment.completion.output_tokens
                         if judgment.completion else 0))

            logger.log_trial(
                experiment="exp1",
                task_id=task["task_id"],
                domain=task["domain"],
                architecture="single_agent",
                payload_type="static",
                payload_id=payload.payload_id,
                payload_category=payload.category,
                injection_position="end",
                agent_responses=[response.response_text],
                final_answer=response.response_text,
                asr_judgment={
                    "followed_injection": judgment.followed_injection,
                    "confidence": judgment.confidence,
                    "evidence": judgment.evidence,
                    "reasoning": judgment.reasoning,
                },
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

            if trial_count % 50 == 0:
                COST_TRACKER.checkpoint(f"exp1 trial {trial_count} [{model_name}]")

    static_bar.close()

    # ===== CAMOUFLAGE PAYLOADS =====
    cam_bar = tqdm(total=len(tasks) * config.n_camouflage_variants,
                   desc=f"Camouflage payloads [{model_name}]")
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
                   "single_agent", "none", model_name)
            cam_bar.update(1)
            if sig in completed_set:
                continue

            injected = payload_bank.inject(task["clean_context"], variant.payload_text,
                                           position="end")
            response = agent.analyze(task["instruction"], injected, task["task_id"])
            detection = detector.detect(injected, task["instruction"], ground_truth=True)
            judgment = judge.judge(task["instruction"], task["malicious_goal"],
                                   response.response_text)

            cost = (response.completion.cost_usd
                    + (detection.completion.cost_usd if detection.completion else 0)
                    + (judgment.completion.cost_usd if judgment.completion else 0))
            tokens = (response.completion.input_tokens + response.completion.output_tokens
                      + (detection.completion.input_tokens + detection.completion.output_tokens
                         if detection.completion else 0)
                      + (judgment.completion.input_tokens + judgment.completion.output_tokens
                         if judgment.completion else 0))

            logger.log_trial(
                experiment="exp1",
                task_id=task["task_id"],
                domain=task["domain"],
                architecture="single_agent",
                payload_type="camouflage",
                payload_id=variant.payload_id,
                payload_category="camouflage",
                injection_position="end",
                agent_responses=[response.response_text],
                final_answer=response.response_text,
                asr_judgment={
                    "followed_injection": judgment.followed_injection,
                    "confidence": judgment.confidence,
                    "evidence": judgment.evidence,
                    "reasoning": judgment.reasoning,
                },
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

            if trial_count % 50 == 0:
                COST_TRACKER.checkpoint(f"exp1 trial {trial_count} [{model_name}]")

    cam_bar.close()
    return trial_count


def run_exp1(
    config=None,
    tasks_override=None,
    logger: TrialLogger | None = None,
    completed_set: set | None = None,
):
    """
    Run Experiment 1.

    Args:
        config: Config object (defaults to CONFIG).
        tasks_override: If provided, use this task list instead of loading from disk.
        logger: TrialLogger instance (created if None).
        completed_set: Set of completed trial signatures for resume logic.
            Modified in-place as new trials are logged.

    Returns:
        Dict of computed metrics (includes _model2 keys if config.run_second_model).
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
    print(f"EXP 1: Static vs. Camouflage — Single Agent")
    print(f"Tasks: {len(tasks)} | Dry run: {config.dry_run}")
    print(f"{'='*60}\n")

    # ---- Primary model ----
    a_client, att_client, det_client, j_client = _build_clients(config, use_second=False)
    print(f"Running clean baselines...")
    tc = _run_exp1_loops(tasks, config, logger, completed_set,
                         a_client, att_client, det_client, j_client,
                         config.agent_model, config.agent_provider)

    # ---- Second model (optional) ----
    if config.run_second_model:
        print(f"\n--- Running second model: {config.second_model} ---")
        a2, att2, det2, j2 = _build_clients(config, use_second=True)
        _run_exp1_loops(tasks, config, logger, completed_set,
                        a2, att2, det2, j2,
                        config.second_model, config.second_provider)

    # ---- Compute metrics per model ----
    all_trials = logger.load_trials(experiment="exp1")
    primary_trials = [t for t in all_trials if t.get("model") == config.agent_model]
    results = _compute_exp1_metrics(primary_trials)

    if config.run_second_model:
        second_trials = [t for t in all_trials if t.get("model") == config.second_model]
        second_results = _compute_exp1_metrics(second_trials)
        for k, v in second_results.items():
            results[k + "_model2"] = v

    logger.save_metrics_snapshot(
        {
            "asr_by_payload_type": {"static": results["asr_static"],
                                    "camouflage": results["asr_camouflage"]},
            "idr_static_on_static": results["idr_static_on_static"],
            "idr_static_on_camouflage": results["idr_static_on_camouflage"],
            "cdg": results["cdg"],
            "total_cost_usd": COST_TRACKER.total_cost,
        },
        experiment="exp1",
    )

    print(f"\n{'='*60}")
    print(f"EXP 1 RESULTS [{config.agent_model}]:")
    print(f"  ASR_static:     {results['asr_static']:.3f}")
    print(f"  ASR_camouflage: {results['asr_camouflage']:.3f}")
    print(f"  IDR_static→static:     {results['idr_static_on_static']:.3f}")
    print(f"  IDR_static→camouflage: {results['idr_static_on_camouflage']:.3f}")
    print(f"  CDG: {results['cdg']:.3f}")
    if config.run_second_model:
        print(f"  CDG_model2: {results.get('cdg_model2', 'N/A')}")
    print(f"  Total cost: ${COST_TRACKER.total_cost:.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    run_exp1()
