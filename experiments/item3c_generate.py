"""
Item 3c benign twins: generation, screening, J1 and selection (ITEM3C_BENIGN_TWINS_DESIGN.md, amendment 1).

Per arm (llama: Llama 3.1 8B local, matched to the v2 Llama corpus; gpt4o: GPT-4o, matched to the GPT-4o full-context
corpus), per slot (43 tasks x 3 variants), 5 candidates from the frozen twin prompt (data/item3c_prompts.json).
Each candidate: fixed rule screen + LLM screen (analysis/refusal_screen_v2), then J1 (data/item3c_j1_prompt.json)
on screen-passing candidates, then the matching features. J1 is also run on the arm's re-screened camouflage
(validity). Selection keeps, per slot, the passing (screen payload + J1 AGREES) candidate closest to the slot's
camouflage payload on the amendment-1 distance. Texts are stored in the data files only, never printed.

    python3 experiments/item3c_generate.py generate --arm llama|gpt4o [--limit N]
    python3 experiments/item3c_generate.py judge-camouflage --arm llama|gpt4o
    python3 experiments/item3c_generate.py select      -> data/item3c_twins.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from tqdm import tqdm

from analysis.refusal_screen_v2 import final as screen_final, llm_class
from experiments.alignment_check import openrouter_key
from experiments.item3b_rewrite import directive_count, specificity, words
from experiments.regen_corpus import refusal_class
from src.attacks.camouflage_generator import cosine_similarity_texts
from src.utils.llm_client import LLMClient, raise_if_insufficient_credits

FROZEN = {
    "data/item3c_honest_conclusions.json": "ceae77093ef15aaff8ebb2498fdf16f816592b0c34b1b119b6d7fd36bc61d79a",
    "data/item3c_prompts.json": "ea48a5677aae564087dde14566c790c245fe20ce7745e43da2d300e115df8d2a",
    "data/tasks.json": "e866427d4318685d8671ba20d4610735018263445c08f25846acd4ced164c494",
    "data/camouflage_corpus_v2.json": "e38f7467367d1055c95b99bf03b4cb7aae238485731ebd28dce58f286a1aa8e7",
    "data/camouflage_corpus_gpt4o.json": "9725f1acfb5a2aaee5687fd9ab72dc76627c4048640f1b3dc145f5fc59053654",
    "data/item3c_j1_prompt.json": "5df89340f32622621e97b4f08643a866b5fac4a1c483998b74350c883a1b4191",
    "data/refusal_exclusions_final.json": "c721727671fbcbbd1110576afe2752bfd80d73845432f33ad3dbbeaa1fddd311",
}
CANDS = Path("results/item3c_candidates.jsonl")
CAM_J1 = Path("results/item3c_camouflage_j1.jsonl")
API_LOG = Path("results/item3c_api_log.jsonl")
OUT = Path("data/item3c_twins.json")
CAP = 5.00  # amendment 1
PRICE = {"openai/gpt-4o-2024-08-06": (2.50, 10.00), "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32)}
N_VARIANTS, N_CANDS = 3, 5
CORPUS = {"llama": "v2_llama", "gpt4o": "gpt4o_full"}


def check_frozen() -> None:
    for f, h in FROZEN.items():
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f"{f} changed since freeze"


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open() if x.strip()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


class Api:
    """Metered OpenRouter calls for 3c (GPT-4o generation, J1); hard cap and HTTP 402 stop."""

    def __init__(self):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key(), timeout=120.0,
                             max_retries=1)
        self.lock = threading.Lock()
        self.spent = sum(r.get("cost_usd", 0.0) for r in jl(API_LOG))

    def call(self, purpose: str, model: str, system: str, user: str, temperature: float, max_tokens: int,
             schema: dict | None = None) -> str:
        if self.spent >= CAP:
            raise SystemExit(f"3c cap reached: ${self.spent:.4f} >= ${CAP}")
        kw = {}
        if schema:
            kw = {"response_format": {"type": "json_schema", "json_schema": schema},
                  "extra_body": {"provider": {"require_parameters": True}}}
        err = None
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=model, messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
                    temperature=temperature, max_tokens=max_tokens, **kw)
                if r.usage is None or not r.choices:
                    raise ValueError("malformed response (no usage or choices)")
                cost = (r.usage.prompt_tokens * PRICE[model][0] + r.usage.completion_tokens * PRICE[model][1]) / 1e6
                with self.lock:
                    self.spent += cost
                    append(API_LOG, {"timestamp": datetime.now(timezone.utc).isoformat(), "purpose": purpose,
                                     "model": model, "prompt_tokens": r.usage.prompt_tokens,
                                     "completion_tokens": r.usage.completion_tokens, "cost_usd": cost})
                content = r.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("empty output")
                return content
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                raise_if_insufficient_credits(e)
                err = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"{purpose}: {type(err).__name__}: {str(err)[:200]}")


def inputs() -> tuple[dict, dict, dict, dict]:
    tasks = {t["task_id"]: t for t in json.loads(Path("data/tasks.json").read_text())}
    concl = json.loads(Path("data/item3c_honest_conclusions.json").read_text())["conclusions"]
    prompts = json.loads(Path("data/item3c_prompts.json").read_text())
    j1 = json.loads(Path("data/item3c_j1_prompt.json").read_text())
    return tasks, concl, prompts, j1


def camouflage(arm: str) -> dict:
    """{task_id: [payload dicts]} for the arm's camouflage, frozen exclusion set removed."""
    excl = set(json.loads(Path("data/refusal_exclusions_final.json").read_text())["excluded"][CORPUS[arm]])
    if arm == "llama":
        store = json.loads(Path("data/camouflage_corpus_v2.json").read_text())["generated"]["llama"]
    else:
        store = json.loads(Path("data/camouflage_corpus_gpt4o.json").read_text())["full"]
    return {t: [p for p in ps if p["payload_id"] not in excl] for t, ps in store.items()}


def features(text: str, ctx: str) -> dict:
    w = max(words(text), 1)
    return {"acs": float(cosine_similarity_texts(text, ctx)), "words": words(text),
            "spec100": 100 * specificity(text, ctx) / w, "dir100": 100 * directive_count(text) / w}


J1_RE = re.compile(r"\b(AGREES|CONTRADICTS|NOT_ADDRESSED)\b")


def j1_answer(api: Api, j1: dict, task: dict, passage: str) -> str:
    """The judge sometimes pads its JSON with blank lines and is cut off at max_tokens (frozen at 16); the answer
    word is read with a regex; a call without it is retried with max_tokens 64 (temperature 0 repeats the same
    truncation otherwise), up to 3 calls (A-log 2026-10-20 implementation note)."""
    for attempt in range(3):
        raw = api.call("j1", j1["model"], j1["system"], j1["user_template"].format(
            clean_context=task["clean_context"], instruction=task["instruction"], passage=passage),
            j1["temperature"], j1["max_tokens"] if attempt == 0 else 64, j1["schema"])
        m = J1_RE.search(raw)
        if m:
            return m.group(1)
    raise ValueError("J1 answer missing after 3 calls")


def generate(arm: str, limit: int | None) -> None:
    check_frozen()
    tasks, concl, prompts, j1 = inputs()
    s = prompts["settings"][arm]
    api = Api()
    local = LLMClient("ollama", "llama3.1:latest", dry_run=False)  # generator (llama arm) and LLM screen
    done = {(r["arm"], r["slot"], r["cand"]) for r in jl(CANDS) if r.get("status") == "ok"}  # technical errors are redrawn
    tids = sorted(concl)[:limit] if limit else sorted(concl)
    todo = [(t, k, c) for t in tids for k in range(1, N_VARIANTS + 1) for c in range(1, N_CANDS + 1)
            if (arm, f"{t}_v{k}", c) not in done]
    print(f"[3c generate:{arm}] {len(todo)} candidates to draw; spent so far ${api.spent:.4f} of ${CAP}")
    for t, k, c in tqdm(todo, desc=f"3c {arm}", unit="cand"):
        task = tasks[t]
        user = prompts["user_template"].format(clean_context=task["clean_context"][:s["clean_context_chars"]],
                                               domain=task["domain"], honest_conclusion=concl[t])
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "arm": arm, "task_id": t, "domain": task["domain"],
               "slot": f"{t}_v{k}", "cand": c, "generator": s["model"]}
        try:
            if arm == "llama":
                text = local.complete(prompts["system"], user, temperature=s["temperature"],
                                      max_tokens=s["max_tokens"]).content.strip()
            else:
                text = api.call("generate", s["model"], prompts["system"], user, s["temperature"],
                                s["max_tokens"]).strip()
            rule = refusal_class(text)
            llm, _ = llm_class(local, text)
            fin = screen_final(rule, llm)
            rec.update(status="ok", text=text, rule_class=rule, llm_class=llm, screen=fin, j1=None,
                       **features(text, task["clean_context"]))
            if fin == "payload":
                try:
                    rec["j1"] = j1_answer(api, j1, task, text)
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001 - text kept; filled later by the rejudge phase
                    rec["j1_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        append(CANDS, rec)
    print(f"[3c generate:{arm}] done; spent ${api.spent:.4f}")


def judge_camouflage(arm: str) -> None:
    """J1 validity on the arm's re-screened camouflage (same 43 tasks)."""
    check_frozen()
    tasks, concl, _, j1 = inputs()
    api = Api()
    done = {(r["arm"], r["payload_id"]) for r in jl(CAM_J1) if r["status"] == "ok"}
    cam = camouflage(arm)
    todo = [(t, p) for t in sorted(concl) for p in cam.get(t, []) if (arm, p["payload_id"]) not in done]
    for t, p in tqdm(todo, desc=f"3c J1 camouflage {arm}", unit="payload"):
        try:
            ans = j1_answer(api, j1, tasks[t], p["payload_text"])
            append(CAM_J1, {"arm": arm, "task_id": t, "payload_id": p["payload_id"], "status": "ok", "j1": ans})
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            append(CAM_J1, {"arm": arm, "task_id": t, "payload_id": p["payload_id"], "status": "error",
                            "error": f"{type(e).__name__}: {str(e)[:200]}"})
    print(f"[3c J1 camouflage:{arm}] done; spent ${api.spent:.4f}")


def rejudge() -> None:
    """Fill J1 for screen-passing candidates whose J1 failed; appends {"rejudge": true, ...} records."""
    check_frozen()
    tasks, _, _, j1 = inputs()
    api = Api()
    rs = jl(CANDS)
    filled = {(r["arm"], r["slot"], r["cand"]) for r in rs if r.get("rejudge")}
    todo = [r for r in rs if not r.get("rejudge") and r.get("status") == "ok" and r["screen"] == "payload"
            and r["j1"] is None and (r["arm"], r["slot"], r["cand"]) not in filled]
    for r in tqdm(todo, desc="3c rejudge", unit="cand"):
        try:
            append(CANDS, {"rejudge": True, "arm": r["arm"], "slot": r["slot"], "cand": r["cand"],
                           "j1": j1_answer(api, j1, tasks[r["task_id"]], r["text"])})
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"rejudge failed {r['arm']} {r['slot']} {r['cand']}: {type(e).__name__}")
    print(f"[3c rejudge] {len(todo)} attempted; spent ${api.spent:.4f}")


def candidates() -> list[dict]:
    """Status-ok candidates with rejudge results merged in."""
    rs = jl(CANDS)
    fill = {(r["arm"], r["slot"], r["cand"]): r["j1"] for r in rs if r.get("rejudge")}
    out = [r for r in rs if not r.get("rejudge") and r["status"] == "ok"]
    for r in out:
        if r["j1"] is None and (r["arm"], r["slot"], r["cand"]) in fill:
            r["j1"] = fill[(r["arm"], r["slot"], r["cand"])]
    return out


def select() -> None:
    check_frozen()
    tasks, concl, _, _ = inputs()
    cands = candidates()
    out = {"meta": {"design": "ITEM3C_BENIGN_TWINS_DESIGN.md amendment 1", "created": datetime.now(timezone.utc).isoformat()},
           "arms": {}}
    for arm in ("llama", "gpt4o"):
        cam = camouflage(arm)
        feats = {p["payload_id"]: features(p["payload_text"], tasks[t]["clean_context"])
                 for t in concl for p in cam.get(t, [])}
        keys = ("acs", "words", "spec100", "dir100")
        vals = {k: [f[k] for f in feats.values()] for k in keys}
        mean = {k: sum(v) / len(v) for k, v in vals.items()}
        sd = {k: (math.sqrt(sum((x - mean[k]) ** 2 for x in v) / (len(v) - 1)) or 1.0) for k, v in vals.items()}
        slots = {}
        for t in sorted(concl):
            task_cam = [feats[p["payload_id"]] for p in cam.get(t, [])]
            for k in range(1, N_VARIANTS + 1):
                slot, pid = f"{t}_v{k}", f"cam_{t}_v{k}"
                if pid in feats:
                    target, how = feats[pid], "slot payload"
                elif task_cam:
                    target, how = {q: sum(f[q] for f in task_cam) / len(task_cam) for q in keys}, "task mean (slot payload excluded)"
                else:
                    target, how = mean, "arm mean (no camouflage left for task; flagged)"
                ok = [r for r in cands if r["arm"] == arm and r["slot"] == slot and r["screen"] == "payload"
                      and r["j1"] == "AGREES"]
                n_drawn = sum(1 for r in cands if r["arm"] == arm and r["slot"] == slot)
                if not ok:
                    slots[slot] = {"status": "failed", "n_drawn": n_drawn, "n_passing": 0, "target": how}
                    continue
                dist = lambda r: math.sqrt(sum(((r[q] - target[q]) / sd[q]) ** 2 for q in keys))
                best = min(ok, key=lambda r: (dist(r), r["cand"]))
                slots[slot] = {"status": "selected", "task_id": t, "domain": tasks[t]["domain"],
                               "twin_id": f"twin_{arm}_{slot}", "cand": best["cand"], "distance": dist(best),
                               "target": how, "target_payload_id": pid if how == "slot payload" else None,
                               "n_drawn": n_drawn, "n_passing": len(ok), "text": best["text"],
                               **{q: best[q] for q in keys}}
        out["arms"][arm] = {"camouflage_sd": sd, "slots": slots,
                            "n_selected": sum(s["status"] == "selected" for s in slots.values()),
                            "n_failed": sum(s["status"] == "failed" for s in slots.values())}
        print(f"[3c select:{arm}] selected {out['arms'][arm]['n_selected']}, failed {out['arms'][arm]['n_failed']}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["generate", "judge-camouflage", "rejudge", "select"])
    ap.add_argument("--arm", choices=["llama", "gpt4o"])
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only (smoke)")
    a = ap.parse_args()
    if a.phase == "generate":
        generate(a.arm, a.limit)
    elif a.phase == "judge-camouflage":
        judge_camouflage(a.arm)
    elif a.phase == "rejudge":
        rejudge()
    else:
        select()


if __name__ == "__main__":
    main()
