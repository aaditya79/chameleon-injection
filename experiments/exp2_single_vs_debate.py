"""
Experiment 2: Single Agent vs. Multi-Agent Debate

PRIMARY QUESTION: Does debate amplify the camouflage blind spot?

Protocol:
  For each task (45):
    For each camouflage variant (3):
      - Single agent with injection
      - Debate inject_all: all 3 agents receive injected context
      - Debate inject_first: only Agent_A receives injected context

  For each task, one representative static payload per category (4 categories):
      - Same three conditions above

Key metrics: DAF_camouflage, DAF_static, CPS (inject_first), CSS (consensus shift score)
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
from src.agents.single_agent import SingleAgent
from src.agents.debate_orchestrator import DebateOrchestrator
from src.attacks.static_payloads import instantiate_payloads
from src.attacks.camouflage_generator import CamouflageGenerator
from src.attacks.payload_bank import PayloadBank
from src.evaluation.asr_judge import ASRJudge
from src.evaluation.metrics import MetricsComputer


def _build_clients(config, use_second: bool = False):
    """Build agent, attacker, and judge clients for one model config."""
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

    return _make(), _make(), _make()  # agent, attacker, judge


def _log_debate_result(logger, debate_result, task, payload_type, payload_id,
                       judgment, cost, tokens, config, model_name, model_provider,
                       inject_mode: str):
    """Log a debate trial with inject_mode included in the record."""
    history_dicts = [
        {"round": rnd.round_num, "responses": [r.response_text for r in rnd.responses]}
        for rnd in debate_result.round_history
    ]
    logger.log_trial(
        experiment="exp2",
        task_id=task["task_id"],
        domain=task["domain"],
        architecture="debate",
        payload_type=payload_type,
        payload_id=payload_id,
        injection_position="end",
        injected_agent_idx=debate_result.injected_agent_idx,
        agent_responses=debate_result.final_positions,
        debate_history=history_dicts,
        final_answer=debate_result.final_answer,
        asr_judgment={
            "followed_injection": judgment.followed_injection,
            "confidence": judgment.confidence,
            "evidence": judgment.evidence,
            "reasoning": judgment.reasoning,
        } if judgment else None,
        total_tokens=tokens,
        total_cost_usd=cost,
        model=model_name,
        provider=model_provider,
        extra={"consensus_reached": debate_result.consensus_reached,
               "inject_mode": inject_mode},
    )


def _compute_exp2_metrics(trials: list[dict]) -> dict:
    """Compute exp2 summary metrics from a filtered trial list."""
    mc = MetricsComputer()
    summary = mc.compute_from_trials(trials)
    cps_scores = [
        t.get("cps_score", 0.0) for t in trials
        if t.get("inject_mode") == "inject_first" and t.get("architecture") == "debate"
    ]
    mean_cps = sum(cps_scores) / len(cps_scores) if cps_scores else 0.0
    return {
        "daf_camouflage": summary.daf_camouflage,
        "daf_static": summary.daf_static,
        "cps_mean": mean_cps,
        "asr_by_architecture": summary.asr_by_architecture,
    }


def _run_exp2_loops(
    tasks: list[dict],
    config,
    logger: TrialLogger,
    completed_set: set,
    agent_client: LLMClient,
    attacker_client: LLMClient,
    judge_client: LLMClient,
    model_name: str,
    model_provider: str,
) -> list[float]:
    """
    Execute all exp2 trial loops for one model configuration.

    Returns:
        List of CPS scores from inject_first debate trials.
    """
    single_agent = SingleAgent(agent_client, agent_id="single_agent")
    orchestrator = DebateOrchestrator(
        n_agents=config.n_agents,
        n_rounds=config.n_rounds,
        llm_client=agent_client,
        aggregation=config.debate_aggregation,
        seed=config.seed,
    )
    judge = ASRJudge(judge_client)
    camouflage_gen = CamouflageGenerator(attacker_client, compute_similarity=False)
    payload_bank = PayloadBank()

    all_cps_scores: list[float] = []
    trial_count = 0

    # ===== CAMOUFLAGE PAYLOAD TRIALS =====
    cam_total = len(tasks) * config.n_camouflage_variants * 3
    cam_bar = tqdm(total=cam_total, desc=f"Camouflage trials [{model_name}]")

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

            # 1) Single agent
            sig_single = (task["task_id"], "camouflage", variant.payload_id,
                          "single_agent", "none", model_name)
            cam_bar.update(1)
            if sig_single not in completed_set:
                response = single_agent.analyze(task["instruction"], injected, task["task_id"])
                judgment = judge.judge(task["instruction"], task["malicious_goal"],
                                       response.response_text)
                cost = (response.completion.cost_usd
                        + (judgment.completion.cost_usd if judgment.completion else 0))
                tokens = response.completion.input_tokens + response.completion.output_tokens
                logger.log_trial(
                    experiment="exp2",
                    task_id=task["task_id"],
                    domain=task["domain"],
                    architecture="single_agent",
                    payload_type="camouflage",
                    payload_id=variant.payload_id,
                    agent_responses=[response.response_text],
                    final_answer=response.response_text,
                    asr_judgment={
                        "followed_injection": judgment.followed_injection,
                        "confidence": judgment.confidence,
                        "evidence": judgment.evidence,
                        "reasoning": judgment.reasoning,
                    },
                    total_tokens=tokens,
                    total_cost_usd=cost,
                    model=model_name,
                    provider=model_provider,
                    extra={"inject_mode": "none"},
                )
                completed_set.add(sig_single)
                trial_count += 1

            # 2) Debate inject_all
            sig_all = (task["task_id"], "camouflage", variant.payload_id + "_all",
                       "debate", "inject_all", model_name)
            cam_bar.update(1)
            if sig_all not in completed_set:
                debate_all = orchestrator.run_debate(
                    instruction=task["instruction"],
                    context=task["clean_context"],
                    task_id=task["task_id"],
                    injected_context=injected,
                    inject_mode="inject_all",
                )
                j_all = judge.judge(task["instruction"], task["malicious_goal"],
                                    debate_all.final_answer)
                _log_debate_result(logger, debate_all, task, "camouflage",
                                   variant.payload_id + "_all", j_all,
                                   debate_all.total_cost_usd, debate_all.total_tokens,
                                   config, model_name, model_provider, "inject_all")
                completed_set.add(sig_all)
                trial_count += 1

            # 3) Debate inject_first
            sig_first = (task["task_id"], "camouflage", variant.payload_id + "_first",
                         "debate", "inject_first", model_name)
            cam_bar.update(1)
            if sig_first not in completed_set:
                debate_first = orchestrator.run_debate(
                    instruction=task["instruction"],
                    context=task["clean_context"],
                    task_id=task["task_id"],
                    injected_context=injected,
                    inject_mode="inject_first",
                    injected_agent_idx=0,
                )
                j_first = judge.judge(task["instruction"], task["malicious_goal"],
                                      debate_first.final_answer)
                cps_score = orchestrator.compute_conformity_pressure(
                    debate_first, injected_agent_idx=0
                ) if len(debate_first.final_positions) > 1 else 0.0
                all_cps_scores.append(cps_score)
                _log_debate_result(logger, debate_first, task, "camouflage",
                                   variant.payload_id + "_first", j_first,
                                   debate_first.total_cost_usd, debate_first.total_tokens,
                                   config, model_name, model_provider, "inject_first")
                # Stash CPS score in the record via extra (already logged above via _log_debate_result)
                # We track it in-memory in all_cps_scores for the final metrics dict
                completed_set.add(sig_first)
                trial_count += 1

            if trial_count % 50 == 0 and trial_count > 0:
                COST_TRACKER.checkpoint(f"exp2 cam trial {trial_count} [{model_name}]")

    cam_bar.close()

    # ===== STATIC PAYLOAD TRIALS (one per category) =====
    static_categories = ["override_directive", "authority_claim", "role_confusion",
                         "social_engineering"]
    static_total = len(tasks) * len(static_categories) * 3
    static_bar = tqdm(total=static_total, desc=f"Static trials [{model_name}]")

    for task in tasks:
        for category in static_categories:
            payloads = instantiate_payloads(task["malicious_goal"], config.data_dir,
                                            categories=[category])
            if not payloads:
                static_bar.update(3)
                continue
            payload = payloads[0]
            injected = payload_bank.inject(task["clean_context"], payload.instantiated_text, "end")

            # Single
            sig_single = (task["task_id"], "static", payload.payload_id,
                          "single_agent", "none", model_name)
            static_bar.update(1)
            if sig_single not in completed_set:
                response = single_agent.analyze(task["instruction"], injected, task["task_id"])
                judgment = judge.judge(task["instruction"], task["malicious_goal"],
                                       response.response_text)
                logger.log_trial(
                    experiment="exp2",
                    task_id=task["task_id"],
                    domain=task["domain"],
                    architecture="single_agent",
                    payload_type="static",
                    payload_id=payload.payload_id,
                    payload_category=category,
                    agent_responses=[response.response_text],
                    final_answer=response.response_text,
                    asr_judgment={
                        "followed_injection": judgment.followed_injection,
                        "confidence": judgment.confidence,
                        "evidence": judgment.evidence,
                        "reasoning": judgment.reasoning,
                    },
                    total_cost_usd=response.completion.cost_usd,
                    total_tokens=(response.completion.input_tokens
                                  + response.completion.output_tokens),
                    model=model_name,
                    provider=model_provider,
                    extra={"inject_mode": "none"},
                )
                completed_set.add(sig_single)
                trial_count += 1

            # Debate inject_all
            sig_all = (task["task_id"], "static", payload.payload_id + "_all",
                       "debate", "inject_all", model_name)
            static_bar.update(1)
            if sig_all not in completed_set:
                d_all = orchestrator.run_debate(
                    task["instruction"], task["clean_context"], task["task_id"],
                    injected_context=injected, inject_mode="inject_all",
                )
                j_all = judge.judge(task["instruction"], task["malicious_goal"],
                                    d_all.final_answer)
                _log_debate_result(logger, d_all, task, "static",
                                   payload.payload_id + "_all", j_all,
                                   d_all.total_cost_usd, d_all.total_tokens,
                                   config, model_name, model_provider, "inject_all")
                completed_set.add(sig_all)
                trial_count += 1

            # Debate inject_first
            sig_first = (task["task_id"], "static", payload.payload_id + "_first",
                         "debate", "inject_first", model_name)
            static_bar.update(1)
            if sig_first not in completed_set:
                d_first = orchestrator.run_debate(
                    task["instruction"], task["clean_context"], task["task_id"],
                    injected_context=injected, inject_mode="inject_first",
                    injected_agent_idx=0,
                )
                j_first = judge.judge(task["instruction"], task["malicious_goal"],
                                      d_first.final_answer)
                _log_debate_result(logger, d_first, task, "static",
                                   payload.payload_id + "_first", j_first,
                                   d_first.total_cost_usd, d_first.total_tokens,
                                   config, model_name, model_provider, "inject_first")
                completed_set.add(sig_first)
                trial_count += 1

            if trial_count % 50 == 0 and trial_count > 0:
                COST_TRACKER.checkpoint(f"exp2 static trial {trial_count} [{model_name}]")

    static_bar.close()
    return all_cps_scores


def run_exp2(config=None, tasks_override=None, logger=None, completed_set=None):
    """
    Run Experiment 2.

    Args:
        completed_set: Set of completed trial signatures for resume logic.

    Returns:
        Dict of metrics (includes _model2 keys if config.run_second_model).
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
    print(f"EXP 2: Single Agent vs. Multi-Agent Debate")
    print(f"Tasks: {len(tasks)} | Agents: {config.n_agents} | Rounds: {config.n_rounds}")
    print(f"{'='*60}\n")

    # Primary model
    print("Running camouflage trials (single + debate)...")
    a_client, att_client, j_client = _build_clients(config, use_second=False)
    cps_scores_primary = _run_exp2_loops(
        tasks, config, logger, completed_set,
        a_client, att_client, j_client,
        config.agent_model, config.agent_provider,
    )

    # Second model
    cps_scores_second: list[float] = []
    if config.run_second_model:
        print(f"\n--- Running second model: {config.second_model} ---")
        a2, att2, j2 = _build_clients(config, use_second=True)
        cps_scores_second = _run_exp2_loops(
            tasks, config, logger, completed_set,
            a2, att2, j2,
            config.second_model, config.second_provider,
        )

    # Metrics
    all_trials = logger.load_trials(experiment="exp2")
    primary_trials = [t for t in all_trials if t.get("model") == config.agent_model]
    mean_cps = (sum(cps_scores_primary) / len(cps_scores_primary)
                if cps_scores_primary else 0.0)
    mc = MetricsComputer()
    summary = mc.compute_from_trials(primary_trials)
    results = {
        "daf_camouflage": summary.daf_camouflage,
        "daf_static": summary.daf_static,
        "cps_mean": mean_cps,
        "asr_by_architecture": summary.asr_by_architecture,
    }

    if config.run_second_model:
        second_trials = [t for t in all_trials if t.get("model") == config.second_model]
        s2 = mc.compute_from_trials(second_trials)
        mean_cps2 = (sum(cps_scores_second) / len(cps_scores_second)
                     if cps_scores_second else 0.0)
        results["daf_camouflage_model2"] = s2.daf_camouflage
        results["daf_static_model2"] = s2.daf_static
        results["cps_mean_model2"] = mean_cps2

    logger.save_metrics_snapshot(
        {
            "daf_camouflage": results["daf_camouflage"],
            "daf_static": results["daf_static"],
            "cps_mean": results["cps_mean"],
            "asr_by_architecture": results["asr_by_architecture"],
            "total_cost_usd": COST_TRACKER.total_cost,
        },
        experiment="exp2",
    )

    print(f"\n{'='*60}")
    print(f"EXP 2 RESULTS [{config.agent_model}]:")
    print(f"  DAF_camouflage: {results['daf_camouflage']:.3f}")
    print(f"  DAF_static:     {results['daf_static']:.3f}")
    print(f"  CPS (mean):     {results['cps_mean']:.3f}")
    print(f"  Total cost: ${COST_TRACKER.total_cost:.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    run_exp2()
