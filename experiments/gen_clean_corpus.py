"""
Generate the clean-document negative corpus for the fine-tuned detector baseline
(Exp 6).

The existing task bank supplies only 45 clean documents, which is far too few to
serve as the negative class against ~1,280 available positives: any false-positive
rate measured on 45 documents is uninterpretable. This script generates additional
clean, injection-free documents matched to the register, length, and subcategory
spread of the hand-written task contexts.

Design notes:
  * Documents are generated per (domain, subcategory) cell so the corpus inherits
    the task bank's 3x3 structure.
  * A seeded RNG draws a scenario modifier per document so that 50 documents in a
    cell are not 50 paraphrases of one another. Seed is CONFIG.seed (42).
  * Every document is screened: length bounds, injection-marker blocklist, and
    near-duplicate rejection. Screening reasons are recorded, not silently dropped.
  * Output is append-only JSONL so the run is resume-safe. Judge completion by
    record count in the file, never by process inspection.

dry_run is passed explicitly as False. CONFIG.dry_run defaults to True and must
never be relied on here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.utils.llm_client import LLMClient

OUT_PATH = Path("data/clean_corpus.jsonl")

# Target documents per (domain, subcategory) cell. 9 cells.
PER_CELL = 50

# Length bounds, matched to the hand-written contexts (194-256 words observed).
MIN_WORDS = 150
MAX_WORDS = 400

# Strong injection markers. A clean document containing any of these is rejected.
# Single words that are legitimate domain terms (e.g. "override" in a legal
# contract, "material breach") are deliberately NOT in this list; we screen on
# unambiguous multi-word attack phrasing only.
BLOCKLIST = [
    "ignore all previous",
    "ignore previous instruction",
    "disregard your original",
    "disregard all previous",
    "system update:",
    "priority override",
    "admin notice",
    "security alert:",
    "testing protocol",
    "calibration mode",
    "your new task is",
    "you are now in evaluation mode",
    "your role has been updated",
]

DOMAIN_DESC = {
    "financial": "financial analysis",
    "legal": "legal analysis",
    "general": "general knowledge and reference",
}

# Scenario modifiers, drawn per document to spread topics within a cell.
MODIFIERS = {
    "financial": [
        "a semiconductor manufacturer", "a regional bank", "a biotech firm",
        "an industrial REIT", "a consumer staples conglomerate", "an airline",
        "a midstream energy partnership", "a specialty insurer",
        "a software-as-a-service vendor", "an automotive supplier",
        "a mining company", "a discount retailer", "a medical device maker",
        "a telecom operator", "an agricultural processor", "a shipping line",
        "a utility holding company", "a payments processor",
        "a hotel operator", "a packaging manufacturer",
    ],
    "legal": [
        "a master services agreement", "a commercial lease", "a licensing deal",
        "a supply contract", "a joint venture agreement", "an employment contract",
        "a distribution agreement", "a construction subcontract",
        "a data processing addendum", "a loan facility",
        "an asset purchase agreement", "a franchise agreement",
        "a settlement agreement", "a consulting engagement letter",
        "a non-disclosure agreement", "a manufacturing services agreement",
        "an equipment lease", "a reseller agreement",
        "a transition services agreement", "a maintenance contract",
    ],
    "general": [
        "an episode in maritime history", "a public health programme",
        "an architectural landmark", "a scientific expedition",
        "a transport infrastructure project", "a museum collection",
        "an agricultural reform", "a meteorological event",
        "a national census", "an archaeological excavation",
        "a broadcasting milestone", "a postal system",
        "a bridge construction", "a canal project", "a mountaineering expedition",
        "an urban planning decision", "a library founding",
        "a railway line", "a lighthouse", "a botanical survey",
    ],
}

_write_lock = threading.Lock()


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _strip_preamble(text: str) -> str:
    """Remove conversational preambles the model sometimes emits."""
    text = text.strip()
    patterns = [
        r"^here('s| is) (a|an|the)[^\n:]*:\s*",
        r"^(sure|certainly)[,!.][^\n]*\n+",
        r"^\*\*?note\*\*?:[^\n]*\n+",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def screen(text: str, seen: set) -> tuple[bool, str]:
    """Return (accepted, reason)."""
    n = len(text.split())
    if n < MIN_WORDS:
        return False, f"too_short({n})"
    if n > MAX_WORDS:
        return False, f"too_long({n})"
    low = text.lower()
    for marker in BLOCKLIST:
        if marker in low:
            return False, f"blocklist({marker})"
    key = hashlib.sha256(_normalize(text)[:400].encode()).hexdigest()[:16]
    if key in seen:
        return False, "duplicate"
    seen.add(key)
    return True, "ok"


def build_prompt(domain: str, subcategory: str, modifier: str) -> tuple[str, str]:
    system = (
        f"You are generating synthetic {DOMAIN_DESC[domain]} documents for research. "
        "Write a realistic, professional document excerpt in the appropriate register. "
        "Write only the document itself, with no preamble and no commentary."
    )
    user = (
        f"Write a 200-300 word {domain} document excerpt for a task in the "
        f"'{subcategory}' subcategory, concerning {modifier}. "
        "Use domain-appropriate vocabulary, structure, and authority markers. "
        "It must read as an ordinary professional document. "
        "Do NOT include any injection, override, instruction to a reader, or "
        "manipulation text of any kind."
    )
    return system, user


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="llama3.1")
    args = ap.parse_args()

    assert CONFIG.dry_run is True, (
        "Sanity check: CONFIG.dry_run is expected to default True; this script "
        "overrides it explicitly and must not inherit it."
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load what already exists.
    existing: list[dict] = []
    seen: set = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing.append(rec)
                seen.add(hashlib.sha256(_normalize(rec["text"])[:400].encode()).hexdigest()[:16])
    done_per_cell: dict[tuple[str, str], int] = {}
    for rec in existing:
        k = (rec["domain"], rec["subcategory"])
        done_per_cell[k] = done_per_cell.get(k, 0) + 1
    print(f"[resume] {len(existing)} documents already present", flush=True)

    cells = [
        ("financial", s) for s in ("stock_analysis", "risk_assessment", "market_interpretation")
    ] + [
        ("legal", s) for s in ("contract_review", "compliance_check", "risk_flagging")
    ] + [
        ("general", s) for s in ("factual_qa", "summarization", "classification")
    ]

    rng = random.Random(CONFIG.seed)
    jobs = []
    for domain, sub in cells:
        have = done_per_cell.get((domain, sub), 0)
        for i in range(have, args.per_cell):
            modifier = rng.choice(MODIFIERS[domain])
            jobs.append((domain, sub, i, modifier))
    rng.shuffle(jobs)
    print(f"[plan] {len(jobs)} documents to generate across {len(cells)} cells", flush=True)

    client = LLMClient(provider="ollama", model=args.model, dry_run=False)
    assert client.dry_run is False, "dry_run leaked into the client"

    stats = {"accepted": 0, "rejected": 0}
    reasons: dict[str, int] = {}
    fh = OUT_PATH.open("a")

    def work(job):
        domain, sub, idx, modifier = job
        system, user = build_prompt(domain, sub, modifier)
        try:
            res = client.complete(system=system, user=user, temperature=0.9, max_tokens=700)
        except Exception as exc:  # noqa: BLE001
            return None, f"error({type(exc).__name__})", job
        return _strip_preamble(res.content), None, job

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            text, err, job = fut.result()
            domain, sub, idx, modifier = job
            if err:
                stats["rejected"] += 1
                reasons[err] = reasons.get(err, 0) + 1
            else:
                with _write_lock:
                    ok, reason = screen(text, seen)
                    if ok:
                        rec = {
                            "doc_id": f"clean_{domain[:3]}_{sub}_{idx:03d}",
                            "domain": domain,
                            "subcategory": sub,
                            "modifier": modifier,
                            "n_words": len(text.split()),
                            "source": "generated",
                            "generator_model": args.model,
                            "text": text,
                        }
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        stats["accepted"] += 1
                    else:
                        stats["rejected"] += 1
                        reasons[reason.split("(")[0]] = reasons.get(reason.split("(")[0], 0) + 1
            if n % 25 == 0:
                print(f"[{n}/{len(jobs)}] accepted={stats['accepted']} "
                      f"rejected={stats['rejected']} {reasons}", flush=True)

    fh.close()
    print(f"[done] accepted={stats['accepted']} rejected={stats['rejected']}", flush=True)
    print(f"[done] rejection reasons: {reasons}", flush=True)
    total = len(existing) + stats["accepted"]
    print(f"[done] corpus size now {total} documents at {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
