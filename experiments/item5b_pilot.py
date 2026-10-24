"""
Item 5b pilot (ITEM5B_AGENTDOJO_DESIGN.md, "Pilot"). Run with the AgentDojo venv (agentdojo @ 089ed468c):

    ~/research/agentdojo-venv/bin/python experiments/item5b_pilot.py run      # resumable; stops at the cap
    ~/research/agentdojo-venv/bin/python experiments/item5b_pilot.py report   # -> results/item5b_pilot_report.{json,md}

Sets: GLM-5.3 and DeepSeek V4 Pro: 16 clean banking tasks x 2 runs + 9 important_instructions pairs (one per injection
task, user task drawn with seed 42); GPT-5.6 Terra: the 16 clean tasks once, seeded order, sub-cap $0.50.
Every API call is logged with OpenRouter's billed cost (results/item5b_pilot_api_log.jsonl). A call is refused once
the pilot total reaches CAP; no new episode starts within MARGIN of it.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from agentdojo import benchmark
from agentdojo.agent_pipeline import AgentPipeline, OpenAILLM, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

VERSION, SUITE = "v1.2.2", "banking"
CAP, MARGIN, TERRA_CAP = 4.50, 0.25, 0.50  # key had $4.70 left at the start (2026-10-21)
MODELS = {  # model id -> (short name, pinned OpenRouter endpoint)
    "z-ai/glm-5.3": ("glm-5.3", "z-ai/fp8"),
    "deepseek/deepseek-v4-pro-0813": ("deepseek-v4-pro", "deepinfra/fp8"),  # first-party blocked by the account data policy (user chose DeepInfra fp8, 2026-10-21)
    "openai/gpt-5.6-terra": ("gpt-5.6-terra", "openai"),
}
ROOT = Path("results/item5b_pilot")
API_LOG = Path("results/item5b_pilot_api_log.jsonl")
CTX: dict = {}


class BudgetStop(Exception):
    pass


def spent(model: str | None = None) -> float:
    if not API_LOG.exists():
        return 0.0
    rs = [json.loads(x) for x in API_LOG.open() if x.strip()]
    return sum(r["cost"] or 0 for r in rs if model is None or r["model"] == model)


def client_for(model: str) -> openai.OpenAI:
    c = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    create = c.chat.completions.create

    def logged(**kw):
        if spent() >= CAP:
            raise BudgetStop(f"pilot cap ${CAP} reached")
        kw["extra_body"] = {"provider": {"order": [MODELS[model][1]], "allow_fallbacks": False},
                            "reasoning": {"effort": "low"}, "usage": {"include": True}}
        t0 = time.time()
        for wait in (15, 30, 60, 120, 240, None):  # transient upstream rate limits (429): retry with backoff, not billed
            try:
                r = create(**kw)
                break
            except openai.RateLimitError:
                if wait is None:
                    raise
                time.sleep(wait)
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        with API_LOG.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **CTX, "model": model,
                                "response_model": r.model, "provider": getattr(r, "provider", None),
                                "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                                "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
                                "cost": getattr(u, "cost", None), "latency_s": round(time.time() - t0, 2)}) + "\n")
        return r

    c.chat.completions.create = logged
    return c


def pipeline(model: str) -> AgentPipeline:
    llm = OpenAILLM(client_for(model), model)
    p = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None,
                                                 system_message=None))
    p.name = "openai-compatible"  # as the openai-compatible provider names it: important_instructions -> "AI assistant"
    return p


def plan(suite) -> list[tuple]:
    users = sorted(suite.user_tasks, key=lambda s: int(s.split("_")[-1]))
    rng = random.Random(42)
    pairs = [(rng.choice(users), i) for i in sorted(suite.injection_tasks, key=lambda s: int(s.split("_")[-1]))]
    terra = users[:]
    random.Random(42).shuffle(terra)
    steps = []
    for run in ("run1", "run2"):
        for m in ("z-ai/glm-5.3", "deepseek/deepseek-v4-pro-0813"):
            steps += [(m, run, u, None) for u in users]
        if run == "run1":
            for m in ("z-ai/glm-5.3", "deepseek/deepseek-v4-pro-0813"):
                steps += [(m, "ii", u, i) for u, i in pairs]
            steps += [("openai/gpt-5.6-terra", "run1", u, None) for u in terra]
    return steps


def _episode(suite, p, u, inj, logdir) -> None:
    if inj:
        benchmark.run_task_with_injection_tasks(suite, p, suite.user_tasks[u], load_attack(
            "important_instructions", suite, p), logdir, False, [inj], VERSION)
    else:
        benchmark.run_task_without_injection_tasks(suite, p, suite.user_tasks[u], logdir, False, VERSION)


def run() -> None:
    suite = get_suite(VERSION, SUITE)
    pipes = {}
    start = datetime.now(timezone.utc).isoformat()
    for model, rset, u, inj in plan(suite):
        short = MODELS[model][0]
        logdir = ROOT / short / rset
        name = f"{inj}.json" if inj else "none.json"
        sub = "important_instructions" if inj else "none"
        if (logdir / "openai-compatible" / SUITE / u / sub / name).exists():
            continue
        if spent() > CAP - MARGIN:
            print(f"[pilot] stopping: ${spent():.3f} spent, within ${MARGIN} of the ${CAP} cap (checkpoint; rerun to resume)")
            return
        if model.startswith("openai/") and spent(model) >= TERRA_CAP:
            continue
        logdir.mkdir(parents=True, exist_ok=True)
        p = pipes.setdefault(model, pipeline(model))
        CTX.clear()
        CTX.update(set=rset, user_task=u, injection_task=inj, run_start=start)
        try:
            with OutputLogger(str(logdir)):  # traces are written to the enclosing logger's dir, not the logdir argument
                _episode(suite, p, u, inj, logdir)
        except BudgetStop as e:

            print(f"[pilot] {e}; checkpointed")
            return
        except Exception as e:  # noqa: BLE001  (logged; the episode is reported as an error)
            with (ROOT / "episode_errors.jsonl").open("a") as f:
                f.write(json.dumps({**CTX, "model": model, "error": f"{type(e).__name__}: {str(e)[:300]}"}) + "\n")
        print(f"[pilot] {short} {rset} {u} {inj or ''}  total ${spent():.3f}", flush=True)
    print(f"[pilot] done; total ${spent():.3f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1] == "run":
        run()
    else:
        import item5b_pilot_report
        item5b_pilot_report.main()
