"""
Item 5b full banking run (ITEM5B_AGENTDOJO_DESIGN.md). AgentDojo venv (agentdojo @ 089ed468c):

    OR_PROVIDER unused here; providers are pinned per model below.
    ~/research/agentdojo-venv/bin/python experiments/item5b_run.py run      # resumable; stops at CAP
    ~/research/agentdojo-venv/bin/python experiments/item5b_run.py report   # -> results/item5b_report.{json,md}

Paid agent matrix only: (model, defense, arm) x 144 banking pairs. Detection on the injected fields is a separate,
unpaid offline pass (experiments/item5b_detect.py) over the generated injections -- it is a property of the payload,
not of the episode. CaMeL is run from the camel-venv harness, not here.

Every API call is logged with OpenRouter's billed cost. A call is refused once the total reaches CAP; no new episode
starts within MARGIN of it. Payload text is read by the attack adapter from data files by id and never appears here.
"""

from __future__ import annotations

import json
import os
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so `src` resolves when run as a script

import src.attacks.agentdojo_camouflage_attack as cam  # noqa: E402  registers camouflage_full / camouflage_no_context
import src.attacks.agentdojo_static_attack  # noqa: E402,F401  registers the "static" arm

VERSION, SUITE = "v1.2.2", "banking"
CAP, MARGIN = float(os.environ.get("ITEM5B_CAP", "0")), 0.50  # set ITEM5B_CAP after the top-up; 0 => refuse all
MODELS = {  # model id -> (short, pinned OpenRouter endpoint)
    "deepseek/deepseek-v4-pro-0813": ("deepseek-v4-pro", "deepinfra/fp8"),  # main agent (diag2: 8/30 baseline)
    "z-ai/glm-5.3": ("glm-5.3", "z-ai/fp8"),                                 # baseline-vs-camouflage contrast only
}
ARMS = ("important_instructions", "static", "camouflage_full", "camouflage_no_context")
DEFENSES = (None, "tool_filter", "transformers_pi_detector", "spotlighting_with_delimiting")  # CaMeL is separate
# contrast model runs no-defense only, and only the two arms needed for the baseline-vs-camouflage contrast
CONTRAST = {"model": "z-ai/glm-5.3", "defenses": (None,), "arms": ("important_instructions", "camouflage_full")}
ROOT = Path("results/item5b")
API_LOG = Path("results/item5b_api_log.jsonl")
EXCLUDED = Path("results/item5b_excluded.jsonl")
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
            raise BudgetStop(f"cap ${CAP} reached")
        kw["extra_body"] = {"provider": {"order": [MODELS[model][1]], "allow_fallbacks": False},
                            "reasoning": {"effort": "low"}, "usage": {"include": True}}
        t0 = time.time()
        for wait in (15, 30, 60, 120, 240, None):  # transient upstream 429s: backoff, not billed
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


def pipeline(model: str, defense: str | None) -> AgentPipeline:
    llm = OpenAILLM(client_for(model), model)
    p = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=None, defense=defense,
                                                 system_message_name=None, system_message=None))
    p.name = "openai-compatible" if defense is None else f"openai-compatible-{defense}"
    return p


def _episode(suite, p, attack, u, inj, logdir) -> None:
    if inj is None:  # clean utility: run the user task with no injection, to measure a defense's utility cost
        benchmark.run_task_without_injection_tasks(suite, p, suite.user_tasks[u], logdir, False, VERSION)
    else:
        benchmark.run_task_with_injection_tasks(suite, p, suite.user_tasks[u], attack, logdir, False, [inj], VERSION)


def run() -> None:
    suite = get_suite(VERSION, SUITE)
    k = lambda x: int(x.split("_")[-1])
    users = sorted(suite.user_tasks, key=k)
    injs = sorted(suite.injection_tasks, key=k)
    start = datetime.now(timezone.utc).isoformat()
    # full matrix for the main model; contrast model gets only its subset. arm "clean" = no-attack utility baseline.
    jobs = [("deepseek/deepseek-v4-pro-0813", d, a) for d in DEFENSES for a in ("clean",) + ARMS]
    jobs += [(CONTRAST["model"], d, a) for d in CONTRAST["defenses"] for a in ("clean",) + CONTRAST["arms"]]
    # optional filters so a single block can run alone, e.g. the no-defense keystone first:
    #   ITEM5B_DEFENSES=none ITEM5B_MODELS=deepseek-v4-pro
    only_def = os.environ.get("ITEM5B_DEFENSES")  # comma list of "none"/defense names
    only_mod = os.environ.get("ITEM5B_MODELS")    # comma list of short names
    if only_def:
        keep = set(only_def.split(","))
        jobs = [j for j in jobs if (j[1] or "none") in keep]
    if only_mod:
        keep = set(only_mod.split(","))
        jobs = [j for j in jobs if MODELS[j[0]][0] in keep]
    print(f"[5b] {len(jobs)} cells to run (cap ${CAP})", flush=True)
    pipes: dict = {}
    for model, defense, arm in jobs:
        short = MODELS[model][0]
        logdir = ROOT / short / (defense or "none") / arm
        pname = "openai-compatible" if defense is None else f"openai-compatible-{defense}"
        clean = arm == "clean"
        pairs = [(u, None) for u in users] if clean else [(u, inj) for u in users for inj in injs]
        for u, inj in pairs:
            sub = "none" if clean else arm
            name = "none.json" if clean else f"{inj}.json"
            if (logdir / pname / SUITE / u / sub / name).exists():
                continue
            if spent() > CAP - MARGIN:
                print(f"[5b] stop: ${spent():.3f} within ${MARGIN} of ${CAP} (rerun to resume)")
                return
            logdir.mkdir(parents=True, exist_ok=True)
            p = pipes.setdefault((model, defense), pipeline(model, defense))
            attack = None if clean else load_attack(arm, suite, p)
            CTX.clear()
            CTX.update(model_short=short, defense=defense or "none", arm=arm, user_task=u,
                       injection_task=inj, run_start=start)
            try:
                with OutputLogger(str(logdir)):  # traces go to the enclosing logger's dir, not the arg
                    _episode(suite, p, attack, u, inj, logdir)
            except cam.NoScreenedPayload:  # no screened payload for this slot -> exclude the pair for this arm
                with EXCLUDED.open("a") as f:
                    f.write(json.dumps({**CTX}) + "\n")
            except BudgetStop as e:
                print(f"[5b] {e}; checkpointed")
                return
            except Exception as e:  # noqa: BLE001 (logged; reported as an error, re-scored out)
                with (ROOT / "episode_errors.jsonl").open("a") as f:
                    f.write(json.dumps({**CTX, "error": f"{type(e).__name__}: {str(e)[:300]}"}) + "\n")
            print(f"[5b] {short} {defense or 'none'} {arm} {u} {inj}  total ${spent():.3f}", flush=True)
    print(f"[5b] done; total ${spent():.3f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.argv[1] == "run":
        run()
    else:
        import item5b_report
        item5b_report.main()
