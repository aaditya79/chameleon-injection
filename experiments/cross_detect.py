"""
Cross-model detection: separate attacker strength from detector strength.

Within-model CDG (attacker = detector = model under test) confounds two things.
GPT-4o shows CDG 0.993 against Llama's 0.896, but GPT-4o is also the stronger
attacker (ACS 0.653 vs 0.563, 0 refusals vs 43.7%), so the wider gap could be
better camouflage rather than worse detection. This script fills the off-diagonal
of the 2x2:

                        detector = Llama 3.1 8B    detector = GPT-4o
  payloads = Llama          have (0.868)              cell D  <- run
  payloads = GPT-4o         cell C  <- run            have (0.993)

Reading the result:
  * If the row means differ (GPT-4o payloads evade BOTH detectors more), the
    driver is attacker strength.
  * If the column means differ (BOTH payload sets evade GPT-4o more), the driver
    is detector weakness.
  * Both can be true; the 2x2 says in what proportion.

Only full-context camouflage is run, which is the condition the headline CDG
figures use. Refusals are excluded (Llama corpus carries 14 unresolved).
Static baselines come from the existing runs: IDR_static = 1.000 for both
detectors, so CDG = 1.000 - IDR_camouflage in every cell.

dry_run is passed explicitly False everywhere; CONFIG.dry_run defaults True.
Resume-safe append-only JSONL; judge completion by record count.

    python experiments/cross_detect.py --cell C
    python experiments/cross_detect.py --cell D
    python experiments/cross_detect.py --cell both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.attacks.payload_bank import PayloadBank
from src.detection.static_detector import StaticDetector

OUT_PATH = Path("results/cross_detect.jsonl")
GPT4O_MODEL = "openai/gpt-4o-2024-08-06"
LLAMA_MODEL = "llama3.1"

_lock = threading.Lock()


def load_env_key(name: str) -> str:
    """Read a key from the environment, falling back to .env."""
    val = os.environ.get(name, "")
    if val:
        return val
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


def llama_detector() -> StaticDetector:
    c = LLMClient(provider="ollama", model=LLAMA_MODEL, dry_run=False)
    assert c.dry_run is False
    return StaticDetector(c)


def gpt4o_detector() -> StaticDetector:
    key = load_env_key("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (env or .env)")
    c = LLMClient("openrouter", GPT4O_MODEL, api_key=key,
                  base_url=CONFIG.openrouter_base_url, dry_run=False,
                  cost_alert_usd=25.0)
    assert c.dry_run is False
    return StaticDetector(c)


def load_gpt4o_full() -> list[tuple[str, dict]]:
    d = json.loads(Path("data/camouflage_corpus_gpt4o.json").read_text())
    out = []
    for tid, plist in d["full"].items():
        for p in plist:
            if not p.get("is_refusal"):
                out.append((tid, p))
    return out


def load_llama_full() -> list[tuple[str, dict]]:
    d = json.loads(Path("data/camouflage_corpus_v2.json").read_text())
    out = []
    for tid, plist in d["generated"]["llama"].items():
        for p in plist:
            if not p.get("is_refusal"):
                out.append((tid, p))
    return out


def run_cell(cell: str, detector: StaticDetector, payloads, corpus_tag: str,
             detector_tag: str, tasks: dict, workers: int) -> int:
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["cell"] == cell:
                done.add(r["payload_id"])
    todo = [(t, p) for t, p in payloads if p["payload_id"] not in done]
    print(f"[cell {cell}] {len(payloads)} payloads, {len(done)} done, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return 0

    bank = PayloadBank()
    fh = OUT_PATH.open("a")
    n_ok = 0

    def work(item):
        tid, p = item
        task = tasks[tid]
        ctx = bank.inject(task["clean_context"], p["payload_text"], "end")
        try:
            res = detector.detect(ctx, task["instruction"], ground_truth=True)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}", item
        return res, None, item

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, it) for it in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            res, err, (tid, p) = fut.result()
            if err:
                print(f"  ! {p['payload_id']}: {err}", flush=True)
                continue
            rec = {
                "cell": cell,
                "corpus_model": corpus_tag,
                "detector_model": detector_tag,
                "task_id": tid,
                "payload_id": p["payload_id"],
                "domain": p["domain"],
                "acs": p.get("acs"),
                "verdict": res.verdict,
                "confidence": res.confidence,
                "dry_run": False,
                "is_refusal": False,
            }
            with _lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            n_ok += 1
            if n % 25 == 0:
                print(f"  [cell {cell}] {n}/{len(todo)}", flush=True)
    fh.close()
    print(f"[cell {cell}] wrote {n_ok} records", flush=True)
    return n_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["C", "D", "both"], default="both")
    ap.add_argument("--llama-workers", type=int, default=2,
                    help="kept low: the clean-corpus generator is using the GPU")
    ap.add_argument("--api-workers", type=int, default=6)
    args = ap.parse_args()

    assert CONFIG.dry_run is True, "expected CONFIG.dry_run default True; we override explicitly"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tasks = {t["task_id"]: t for t in json.loads(Path("data/tasks.json").read_text())}

    if args.cell in ("D", "both"):
        # API-bound, no GPU contention: run first.
        run_cell("D", gpt4o_detector(), load_llama_full(),
                 corpus_tag="llama3.1", detector_tag=GPT4O_MODEL,
                 tasks=tasks, workers=args.api_workers)

    if args.cell in ("C", "both"):
        run_cell("C", llama_detector(), load_gpt4o_full(),
                 corpus_tag=GPT4O_MODEL, detector_tag=LLAMA_MODEL,
                 tasks=tasks, workers=args.llama_workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
