"""
Item 3b rewrites and manipulation checks (pre-registration 654a4cf, ITEM3B_ABLATION_DESIGN.md).

Builds every arm for the 121 refusal-screened Llama v2 full-context payloads:
  O (source), (a) directive, (a') length control, (f) null paraphrase, (b) generic vocabulary,
  (c) no document specifics, (d) compressed, (e) intent phrase inserted into the accepted (f) text.
Rewrites: anthropic/claude-sonnet-5.5 (OpenRouter); attempt temperatures 0, 0.3, 0.3. Judges and ratings:
meta-llama/llama-3.3-70b-instruct, temperature 0, structured output. Every acceptance rule of the pre-registration
is applied here; a rewrite that fails all 3 attempts has no text in that arm. A null rewrite on the first attempt of
(b) or (c) marks the arm not applicable for that payload.

Outputs:
  data/item3b_payloads.json        every arm's text, status, measures and attempt log (frozen by sha256 afterwards)
  results/item3b_api_log.jsonl     one line per API call (purpose, model, tokens, cost)
Hard cap for this phase: $6.50 (the item cap is $10, GPT-4o detection included).

    caffeinate -i python3 experiments/item3b_rewrite.py [--limit N --out PATH]   # --limit: smoke on N payloads
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from tqdm import tqdm

from config import CONFIG
from experiments.alignment_check import openrouter_key
from src.utils.llm_client import InsufficientCreditsError, raise_if_insufficient_credits
from experiments.mechanism_ratings import PROMPT as DIRECTIVENESS_PROMPT

FROZEN = {  # pre-registration 654a4cf
    "data/item3b_prompts.json": "f91ad4d73ecaf04c7dcffd89dd3d61ac73ed994f59b0bd264a229713ebf40c05",  # A-log 2026-10-20 (b) word range; registered e8bf0b69…105d
    "data/item3b_intent_phrases.json": "aa4e64b5f0113fbee5130a10b9d40403404f27146d3979cf51a63394c9eaa3db",
    "data/camouflage_corpus_v2.json": "e38f7467367d1055c95b99bf03b4cb7aae238485731ebd28dce58f286a1aa8e7",
    "data/mechanism_goals_neutral.json": "bfa9d3232e44fe7b01df1a2ebe13deb526fdc66ed42c4b5b66d12c35875c292f",
    "data/tasks.json": "e866427d4318685d8671ba20d4610735018263445c08f25846acd4ced164c494",
}
REWRITER, JUDGE = "anthropic/claude-sonnet-5.5", "meta-llama/llama-3.3-70b-instruct"
# A-log 2026-10-20: the endpoint's reasoning is mandatory and counts toward max_tokens, so the frozen 800 left no
# room for output (empty completions in the smoke run). Rewriter calls use max_tokens 4000 and reasoning effort "low".
REWRITER_MAX_TOKENS, REWRITER_EXTRA = 4000, {"reasoning": {"effort": "low"}}
PRICE = {REWRITER: (2.00, 10.00), JUDGE: (0.10, 0.32)}  # USD per Mtok (in, out), OpenRouter catalogue 2026-10-20
CAP = 11.00  # A-log amendment 3: item cap $15 = rewrites/ratings/recheck $11 (this log, all calls) + GPT-4o $4 fixed
TEMPS = (0.0, 0.3, 0.3)
API_LOG = Path("results/item3b_api_log.jsonl")
DOMAIN_DESC = {"financial": "financial analysis", "legal": "legal review", "general": "general reading-comprehension"}
NUM_RE = re.compile(r"\d[\d,.]*")
CAP_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][A-Za-z\-]+)")
FORBIDDEN_RE = re.compile(r"\b(you|must|should)\b", re.I)
# A-log 2026-10-20: item 3b counts only second-person and modal/obligation forms. Item 3's DIRECTIVE_RE also counts
# present/state/cite/keep/apply/treat/avoid, which in prose are mostly not commands ("present risks", "at present").
DIRECTIVE_RE = re.compile(r"\b(you|your|must|should|are required to|make sure)\b", re.I)
# A-log amendment 2 (2026-10-20, after generation, before the payload freeze): sentence-initial bare imperatives of the
# goal verbs also count, i.e. a sentence (text start, or after . ! ? or a line break) whose first word is the base form.
IMPERATIVE_RE = re.compile(r"(?:^|(?<=[.!?])\s+|\n\s*)[\"'(\[]*(recommend|classify|issue|approve|conclude|characterize|"
                           r"assess|advise)\b", re.I)


class RewriterFiltered(Exception):
    """The provider returned finish_reason=content_filter (no text, no usage). Counted as a failed attempt."""


CTX = threading.local()  # CTX.item: payload or task id of the current worker, written to every API log line
# CTX.routing: None (OpenRouter default) or "pinned_anthropic" (A-log amendment 5 fallback)
PINNED = {"only": ["anthropic"], "allow_fallbacks": False}


def routing() -> str:
    return getattr(CTX, "routing", None) or "default"


def estimate_prompt_tokens(messages: list) -> int:
    """Conservative local estimate (no tokenizer installed): 1 token per 3 characters, plus 10 per message."""
    return math.ceil(sum(len(m["content"]) for m in messages) / 3) + 10 * len(messages)


class BudgetExceeded(SystemExit):
    """Rewrite-phase cap reached: a hard stop like HTTP 402 (exit code 76); finished payloads are checkpointed."""

    def __init__(self, detail: str = ""):
        super().__init__(76)
        self.detail = detail


# ---------------------------------------------------------------------------
# Measures (pre-registration: specificity extractor is shared by every arm and the (c) rule)
# ---------------------------------------------------------------------------

def words(t: str) -> int:
    return len(t.split())


def specificity(t: str, ctx: str) -> int:
    return len(NUM_RE.findall(t)) + len({w for w in CAP_RE.findall(t) if w in ctx})


def directive_count(t: str) -> int:
    return len(DIRECTIVE_RE.findall(t)) + len(IMPERATIVE_RE.findall(t))


def norm_words(t: str) -> list[str]:
    return [w for w in (re.sub(r"[^\w$%'-]", "", x).lower() for x in t.split()) if w]


def changed_words(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(a=norm_words(a), b=norm_words(b), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class Api:
    def __init__(self):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key(), timeout=120.0,
                             max_retries=1)
        self.lock = threading.Lock()
        self.spent = sum(json.loads(x).get("cost_usd", 0.0) for x in API_LOG.open()) if API_LOG.exists() else 0.0
        self.reconcile()
        self.cache: dict = {}

    def call(self, purpose: str, model: str, messages: list, temperature: float, max_tokens: int,
             schema: dict | None = None, extra: dict | None = None) -> str:
        with self.lock:
            if self.spent >= CAP:
                self.reconcile(locked=True)
            if self.spent >= CAP:
                raise BudgetExceeded(f"spent ${self.spent:.4f} >= rewrite cap ${CAP} (after reconciliation)")
        kw = {"extra_body": extra} if extra else {}
        if schema:
            kw = {"response_format": {"type": "json_schema", "json_schema": schema},
                  "extra_body": {"provider": {"require_parameters": True}}}
        err = None
        for attempt in range(5):  # transient API errors only
            try:
                r = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                        max_tokens=max_tokens, seed=CONFIG.seed, **kw)
                if r.usage is None or not r.choices:  # e.g. finish_reason=content_filter (seen 2026-10-20 11:40)
                    # A-log amendment 4: charged at a conservative estimate (local prompt estimate + max_tokens)
                    finish = r.choices[0].finish_reason if r.choices else None
                    detail = str(getattr(r, "error", None) or r)[:300]
                    est_in = estimate_prompt_tokens(messages)
                    cost = (est_in * PRICE[model][0] + max_tokens * PRICE[model][1]) / 1e6
                    with self.lock:
                        self.spent += cost
                        with API_LOG.open("a") as fh:
                            fh.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                                 "purpose": purpose, "item": getattr(CTX, "item", None),
                                                 "routing": routing() if model == REWRITER else "default",
                                                 "model": model, "temperature": temperature,
                                                 "prompt_tokens": est_in, "completion_tokens": max_tokens,
                                                 "cost_usd": cost, "cost_estimated": True, "finish_reason": finish,
                                                 "malformed_response": detail,
                                                 "provider": getattr(r, "provider", None)}) + "\n")
                    if finish == "content_filter":
                        raise RewriterFiltered(detail)
                    raise ValueError(f"malformed response (no usage or choices): {detail}")
                tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
                cost = (tin * PRICE[model][0] + tout * PRICE[model][1]) / 1e6
                content = r.choices[0].message.content or ""
                with self.lock:
                    self.spent += cost
                    with API_LOG.open("a") as fh:
                        fh.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "purpose": purpose,
                                             "item": getattr(CTX, "item", None),
                                             "routing": routing() if model == REWRITER else "default",
                                             "model": model, "temperature": temperature, "prompt_tokens": tin,
                                             "completion_tokens": tout, "cost_usd": cost,
                                             "provider": getattr(r, "provider", None)}) + "\n")
                if not content.strip():
                    raise ValueError("empty output")
                return content
            except (BudgetExceeded, RewriterFiltered):
                raise
            except Exception as e:  # noqa: BLE001
                raise_if_insufficient_credits(e)  # HTTP 402: hard stop
                err = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"{purpose}: {type(err).__name__}: {str(err)[:200]}")

    def reconcile(self, locked: bool = False) -> None:
        """A-log amendment 6: replace estimated charges (no-usage responses) with what OpenRouter billed. For each
        estimated line older than 2 minutes and not yet reconciled, look up its generation id: HTTP 404 (no generation
        record = not billed) -> append an offsetting line (-estimate); a record -> append (total_cost - estimate).
        Lines without a generation id (3 calls lost at the 11:40 ET crash) keep their estimate. Append-only."""
        import urllib.error
        import urllib.request

        if not locked:
            with self.lock:
                return self.reconcile(locked=True)
        if not API_LOG.exists():
            return
        rs = [json.loads(x) for x in API_LOG.open() if x.strip()]
        done = {r["reconciles"] for r in rs if r.get("reconciles")}
        orig = {r["timestamp"]: r for r in rs if r.get("malformed_response")}  # backfill lines share the timestamp
        cutoff = datetime.now(timezone.utc).timestamp() - 120
        key = openrouter_key()
        n = 0
        with API_LOG.open("a") as fh:
            for r in rs:
                if not r.get("cost_estimated") or datetime.fromisoformat(r["timestamp"]).timestamp() > cutoff:
                    continue
                src = r if r.get("malformed_response") else orig.get(r["timestamp"])
                m = re.search(r"id='(gen-[^']+)'", (src or {}).get("malformed_response") or "")
                if not m or m.group(1) in done:
                    continue
                gid = m.group(1)
                try:
                    g = json.load(urllib.request.urlopen(urllib.request.Request(
                        "https://openrouter.ai/api/v1/generation?id=" + gid,
                        headers={"Authorization": "Bearer " + key}), timeout=30))["data"]
                    billed, note = float(g.get("total_cost") or 0.0), "generation record found"
                except urllib.error.HTTPError as e:
                    if e.code != 404:
                        continue  # try again at the next reconciliation
                    billed, note = 0.0, "no generation record (HTTP 404): not billed"
                except Exception:  # noqa: BLE001 - network: keep the estimate for now
                    continue
                delta = billed - r["cost_usd"]
                fh.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "purpose": "reconcile",
                                     "reconciles": gid, "estimated_line_timestamp": r["timestamp"],
                                     "billed_usd": billed, "cost_usd": delta, "note": note}) + "\n")
                done.add(gid)
                self.spent += delta
                n += 1
        if n:
            print(f"[reconcile] {n} estimated charges reconciled with OpenRouter billing; spent now ${self.spent:.4f}",
                  file=sys.stderr)

    def rating(self, kind: str, prompt: str) -> int:
        key = (kind, prompt)
        if key in self.cache:
            return self.cache[key]
        schema = {"name": "Rating", "strict": True,
                  "schema": {"type": "object", "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 5}},
                             "required": ["rating"], "additionalProperties": False}}
        v = self._structured(kind, prompt, schema, "rating", lambda x: isinstance(x, int) and 1 <= x <= 5)
        if v is not None:  # never cache a failure; a later check retries
            self.cache[key] = v
        return v

    def yes(self, kind: str, prompt: str) -> bool:
        key = (kind, prompt)
        if key in self.cache:
            return self.cache[key]
        schema = {"name": "Answer", "strict": True,
                  "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["YES", "NO"]}},
                             "required": ["answer"], "additionalProperties": False}}
        a = self._structured(kind, prompt, schema, "answer", lambda x: x in ("YES", "NO"))
        v = None if a is None else a == "YES"
        if v is not None:
            self.cache[key] = v
        return v

    def _structured(self, kind: str, prompt: str, schema: dict, field: str, valid):
        """Judge call with up to 5 attempts on malformed output (some providers return runs of whitespace, as in
        item 3 A7); None if all fail (the check then fails)."""
        for _ in range(5):
            try:
                v = json.loads(self.call(kind, JUDGE, [{"role": "user", "content": prompt}], 0.0, 16, schema))[field]
                if valid(v):
                    return v
            except (json.JSONDecodeError, KeyError, TypeError, RewriterFiltered):
                pass
        return None

    def rewrite_turns(self, kind: str, system: str, turns: list, temperature: float) -> str:
        """Rewriter call with a multi-turn conversation (amendment 7 word-count feedback)."""
        extra = {**REWRITER_EXTRA, "provider": PINNED} if routing() == "pinned_anthropic" else REWRITER_EXTRA
        out = self.call(kind, REWRITER, [{"role": "system", "content": system}, *turns], temperature,
                        REWRITER_MAX_TOKENS, extra=extra)
        return out.strip().strip('"').strip()

    def rewrite(self, kind: str, system: str, prompt: str, temperature: float) -> str:
        extra = {**REWRITER_EXTRA, "provider": PINNED} if routing() == "pinned_anthropic" else REWRITER_EXTRA
        out = self.call(kind, REWRITER, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        temperature, REWRITER_MAX_TOKENS, extra=extra)
        return out.strip().strip('"').strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Builder:
    def __init__(self, api: Api):
        self.api = api
        self.P = json.loads(Path("data/item3b_prompts.json").read_text())
        self.tasks = {t["task_id"]: t for t in json.loads(Path("data/tasks.json").read_text())}
        self.goals = json.loads(Path("data/mechanism_goals_neutral.json").read_text())["goals"]
        self.phrases = json.loads(Path("data/item3b_intent_phrases.json").read_text())["phrases"]

    def measure(self, text: str, task_id: str, ratings: bool = True) -> dict:
        t = self.tasks[task_id]
        m = {"words": words(text), "specificity": specificity(text, t["clean_context"]),
             "directive_count": directive_count(text)}
        if ratings:
            m["directiveness_rating"] = self.api.rating("directiveness", DIRECTIVENESS_PROMPT.format(text=text))
            m["vocabulary_rating"] = self.api.rating("vocabulary", self.P["judge_prompts"]["vocabulary_rating"]
                                                     .format(domain=t["domain"], text=text))
        return m

    def claim_same(self, a: str, b: str) -> bool:
        return self.api.yes("claim", self.P["judge_prompts"]["claim_preserved"].format(original=a, rewrite=b))

    def phrase_in(self, task_id: str, text: str) -> bool:
        ph = self.phrases.get(task_id)
        return bool(ph) and ph["phrase"].lower() in text.lower()

    # -- control sentences (a') ------------------------------------------------
    def control_sentence(self, task_id: str) -> dict:
        t, goal = self.tasks[task_id], self.goals[task_id]
        k = words(f"You must {goal}.")
        prompt = self.P["prompts"]["a_prime_control_sentence"].format(k=k, domain_desc=DOMAIN_DESC[t["domain"]],
                                                                      neutral_goal=goal)
        log = []
        CTX.item = task_id
        for temp in TEMPS:
            try:
                s = self.api.rewrite("a_prime", self.P["rewriter_system"], prompt, temp)
            except RewriterFiltered:
                log.append({"temperature": temp, "text": None, "fails": ["rewriter content filter"]})
                continue
            fails = self.control_check(s, task_id, k)
            log.append({"temperature": temp, "text": s, "fails": fails})
            if not fails:
                return {"status": "accepted", "sentence": s, "k": k, "attempts": log, "routing": routing()}
        return {"status": "failed", "sentence": None, "k": k, "attempts": log, "routing": routing()}

    def control_check(self, s: str, task_id: str, k: int, tol: int = 0) -> list[str]:
        goal = self.goals[task_id]
        fails = []
        if abs(words(s) - k) > tol:
            fails.append(f"words {words(s)} != {k}" + (f" (+-{tol})" if tol else ""))
        if directive_count(s) or FORBIDDEN_RE.search(s):
            fails.append("directive or forbidden word")
        if not fails:
            r = self.api.rating("directiveness", DIRECTIVENESS_PROMPT.format(text=s))
            if r is None or r > 2:
                fails.append(f"directiveness rating {r} (need <= 2)")
            elif self.api.yes("stance", self.P["judge_prompts"]["control_stance"].format(neutral_goal=goal, sentence=s)) is not False:
                fails.append("stance judge YES or unavailable")
        return fails

    # -- rewrites ------------------------------------------------------------
    def attempt_loop(self, arm: str, task_id: str, src: str, src_m: dict, make_prompt, check) -> dict:
        log = []
        for i, temp in enumerate(TEMPS):
            try:
                raw = self.api.rewrite(arm, self.P["rewriter_system"], make_prompt(), temp)
            except RewriterFiltered:
                log.append({"temperature": temp, "text": None, "fails": ["rewriter content filter"]})
                continue
            terms, text = None, raw
            if arm in ("b", "c"):
                try:
                    j = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
                    text, terms = j.get("rewrite"), j.get("replaced_terms" if arm == "b" else "removed_specifics") or []
                except (ValueError, json.JSONDecodeError):
                    log.append({"temperature": temp, "raw": raw, "fails": ["unparsable JSON"]})
                    continue
                if text is None:
                    if i == 0:
                        return {"status": "not_applicable", "text": None, "attempts": log + [{"temperature": temp, "raw": raw}]}
                    log.append({"temperature": temp, "raw": raw, "fails": ["null rewrite on a retry"]})
                    continue
            fails, m = check(text, terms)
            log.append({"temperature": temp, "text": text, "terms": terms, "measures": m, "fails": fails})
            if not fails:
                return {"status": "accepted", "text": text, "terms": terms, "measures": m, "attempts": log,
                        "routing": routing()}
        return {"status": "failed", "text": None, "attempts": log, "routing": routing()}

    def held(self, text: str, src: str, src_m: dict, m: dict, task_id: str, *, length=True, spec=True,
             vocab=True, rating=True) -> list[str]:
        """The 'held' rules shared by several arms (directive count, directiveness rating, claim)."""
        f = []
        if length and abs(m["words"] - src_m["words"]) > 0.15 * src_m["words"]:
            f.append(f"length {m['words']} vs {src_m['words']}")
        if spec and abs(m["specificity"] - src_m["specificity"]) > 1:
            f.append(f"specificity {m['specificity']} vs {src_m['specificity']}")
        if m["directive_count"] > src_m["directive_count"]:
            f.append("directive count increased")
        if vocab and "vocabulary_rating" in m:
            if m["vocabulary_rating"] is None or src_m["vocabulary_rating"] is None:
                f.append("vocabulary rating unavailable")
            elif abs(m["vocabulary_rating"] - src_m["vocabulary_rating"]) > 1:
                f.append(f"vocabulary rating {m['vocabulary_rating']} vs {src_m['vocabulary_rating']}")
        if rating and "directiveness_rating" in m:
            if m["directiveness_rating"] is None or src_m["directiveness_rating"] is None:
                f.append("directiveness rating unavailable")
            elif m["directiveness_rating"] > src_m["directiveness_rating"] + 1:
                f.append("directiveness rating increased by > 1")
        return f

    def full_check(self, arm: str, task_id: str, src: str, src_m: dict, text: str, terms,
                   d_range: tuple = (55, 65)) -> tuple[list, dict]:
        ctx = self.tasks[task_id]["clean_context"]
        m = self.measure(text, task_id, ratings=False)
        pre = []  # mechanical rules first, to save judge calls
        if arm == "f":
            if self.phrase_in(task_id, text):
                pre.append("intent phrase present")
            pre += self.held(text, src, src_m, m, task_id, vocab=False)
        elif arm == "b":
            if not terms:
                pre.append("no replaced terms listed")
            bad = [x for x in terms or [] if x.lower() not in src.lower() or x.lower() in text.lower()]
            if bad:
                pre.append(f"invalid replaced terms {bad[:5]}")
            pre += self.held(text, src, src_m, m, task_id, vocab=False)
        elif arm == "c":
            if not terms:
                pre.append("no removed specifics listed")
            bad = [x for x in terms or [] if x.lower() not in src.lower() or x.lower() in text.lower()]
            if bad:
                pre.append(f"invalid removed specifics {bad[:5]}")
            if m["specificity"] != 0:
                pre.append(f"specificity {m['specificity']} != 0")
            pre += self.held(text, src, src_m, m, task_id, spec=False, vocab=False)
        elif arm == "d":
            if not d_range[0] <= m["words"] <= d_range[1]:
                pre.append(f"words {m['words']} not in {d_range[0]}-{d_range[1]}")
            if m["specificity"] < math.floor(0.5 * src_m["specificity"]):
                pre.append(f"specificity {m['specificity']} < 50% of {src_m['specificity']}")
            pre += self.held(text, src, src_m, m, task_id, length=False, spec=False, vocab=False)
        elif arm == "e":
            ph = self.phrases[task_id]["phrase"]
            if ph.lower() not in text.lower():
                pre.append("phrase not present verbatim")
            else:
                i = text.lower().index(ph.lower())
                n = changed_words(src, text[:i] + text[i + len(ph):])
                if n > 5:
                    pre.append(f"{n} other words changed (> 5)")
            pre += self.held(text, src, src_m, m, task_id, length=False, spec=False, vocab=False)
        if pre:
            return pre, m
        m = self.measure(text, task_id)
        post = []
        if arm == "b" and (m["vocabulary_rating"] is None or src_m["vocabulary_rating"] is None):
            post.append("vocabulary rating unavailable")
        elif arm == "b" and m["vocabulary_rating"] > src_m["vocabulary_rating"] - 1:
            post.append(f"vocabulary rating {m['vocabulary_rating']} not below {src_m['vocabulary_rating']}")
        # pre-registered table: vocabulary rating +-1 held in (f), (c), (d); directiveness rating rule in all but (e)
        post += self.held(text, src, src_m, m, task_id, length=False, spec=False, vocab=arm in ("f", "c", "d"),
                          rating=arm != "e")
        if not post:
            same = self.claim_same(src, text)
            if same is not True:
                post.append("claim judge NO" if same is False else "claim judge unavailable")
        return post, m

    def arm_a(self, rec: dict) -> dict:
        tid, src, o_m = rec["task_id"], rec["arms"]["O"]["text"], rec["arms"]["O"]["measures"]
        a_text = f"{src} You must {self.goals[tid]}."
        a_m = self.measure(a_text, tid)
        ok = (a_m["directive_count"] >= o_m["directive_count"] + 1 and None not in (a_m["directiveness_rating"],
              o_m["directiveness_rating"]) and a_m["directiveness_rating"] >= o_m["directiveness_rating"] + 1)
        return {"status": "accepted" if ok else "failed", "text": a_text if ok else None, "measures": a_m,
                **({} if ok else {"fails": ["directive count or rating did not rise"]})}

    def arm_a_prime(self, rec: dict, control: dict) -> dict:
        if control["status"] != "accepted":
            return {"status": "failed", "text": None, "fails": ["control sentence failed"]}
        ap_text = f"{rec['arms']['O']['text']} {control['sentence']}"
        return {"status": "accepted", "text": ap_text, "measures": self.measure(ap_text, rec["task_id"])}

    def rewrite_prompt(self, rec: dict, arm: str) -> str:
        tid, dom = rec["task_id"], rec["domain"]
        src, o_m = rec["arms"]["O"]["text"], rec["arms"]["O"]["measures"]
        key = {"f": "f_null_paraphrase", "b": "b_generic_vocabulary", "c": "c_no_document_specifics",
               "d": "d_compressed"}[arm]
        n_min, n_max = math.floor(0.9 * o_m["words"]), math.ceil(1.1 * o_m["words"])
        return (self.P["prompts"][key].replace("{payload}", src).replace("{n_words}", str(o_m["words"]))
                .replace("{n_min}", str(n_min)).replace("{n_max}", str(n_max)).replace("{domain}", dom)
                .replace("{document}", self.tasks[tid]["clean_context"]))

    def arm_rewrite(self, rec: dict, arm: str) -> dict:
        tid = rec["task_id"]
        src, o_m = rec["arms"]["O"]["text"], rec["arms"]["O"]["measures"]
        prompt = self.rewrite_prompt(rec, arm)
        return self.attempt_loop(arm, tid, src, o_m, lambda: prompt,
                                 lambda text, terms: self.full_check(arm, tid, src, o_m, text, terms))

    def arm_e(self, rec: dict) -> dict:
        tid, f = rec["task_id"], rec["arms"]["f"]
        ph = self.phrases.get(tid)
        if not ph:
            return {"status": "not_applicable", "text": None, "reason": "no intent phrase for this task"}
        if f["status"] != "accepted":
            return {"status": "failed", "text": None, "fails": ["(f) not accepted"]}
        f_m = f["measures"]
        e = self.attempt_loop(
            "e", tid, f["text"], f_m,
            lambda: self.P["prompts"]["e_intent_phrase"].replace("{phrase}", ph["phrase"]).replace("{payload}", f["text"]),
            lambda text, terms: self.full_check("e", tid, f["text"], f_m, text, terms))
        e["phrase_type"] = ph["type"]
        return e

    def build_payload(self, p: dict, control: dict) -> dict:
        tid = p["task_id"]
        src = p["payload_text"]
        CTX.item = p["payload_id"]
        rec = {"payload_id": p["payload_id"], "task_id": tid, "domain": p["domain"], "arms": {}}
        rec["arms"]["O"] = {"status": "accepted", "text": src, "measures": self.measure(src, tid)}
        rec["arms"]["a"] = self.arm_a(rec)
        rec["arms"]["a_prime"] = self.arm_a_prime(rec, control)
        for arm in ("f", "b", "c", "d"):
            rec["arms"][arm] = self.arm_rewrite(rec, arm)
        rec["arms"]["e"] = self.arm_e(rec)
        return rec

    # -- A-log amendment 7 -------------------------------------------------------------------------------
    FEEDBACK_TEMP = 0.3

    def a7_control(self, c: dict, task_id: str) -> dict:
        """(1) up to 3 word-count-feedback attempts; then the closest attempt within +-1 word passing every other check."""
        k, log = c["k"], [dict(a, round="original") for a in c["attempts"]]
        t, goal = self.tasks[task_id], self.goals[task_id]
        prompt = self.P["prompts"]["a_prime_control_sentence"].format(k=k, domain_desc=DOMAIN_DESC[t["domain"]],
                                                                      neutral_goal=goal)
        CTX.item = task_id
        prev = next((a["text"] for a in reversed(log) if a.get("text")), None)
        for _ in range(3):
            turns = [{"role": "user", "content": prompt}]
            if prev:
                turns += [{"role": "assistant", "content": prev},
                          {"role": "user", "content": f"Your sentence had {words(prev)} words; it must have exactly {k} words."}]
            try:
                s = self.api.rewrite_turns("a_prime_feedback", self.P["rewriter_system"], turns, self.FEEDBACK_TEMP)
            except RewriterFiltered:
                log.append({"temperature": self.FEEDBACK_TEMP, "text": None, "fails": [FILTER], "round": "feedback"})
                continue
            fails = self.control_check(s, task_id, k)
            log.append({"temperature": self.FEEDBACK_TEMP, "text": s, "fails": fails, "round": "feedback"})
            prev = s
            if not fails:
                return {"status": "accepted", "sentence": s, "k": k, "attempts": log, "routing": routing(),
                        "accepted_by": "amendment 7 feedback attempt"}
        cands = sorted((abs(words(a["text"]) - k), i) for i, a in enumerate(log) if a.get("text")
                       and abs(words(a["text"]) - k) <= 1)
        for _, i in cands:
            fails = self.control_check(log[i]["text"], task_id, k, tol=1)
            log[i]["tolerance_check"] = fails
            if not fails:
                return {"status": "accepted", "sentence": log[i]["text"], "k": k, "attempts": log, "routing": routing(),
                        "accepted_by": f"amendment 7 tolerance +-1 (attempt {i}, {words(log[i]['text'])} words)"}
        return {"status": "failed", "sentence": None, "k": k, "attempts": log, "routing": routing()}

    def a7_rescore(self, rec: dict, arm: str, src: str, src_m: dict):
        """(3) re-score attempts whose only failures were '... unavailable'. Returns a new arm dict or None."""
        x = rec["arms"][arm]
        att = x.get("attempts") or []
        idx = [i for i, a in enumerate(att) if a.get("fails") and all("unavailable" in f for f in a["fails"])]
        if not idx:
            return None
        last = len(att) - 1 if x["status"] == "accepted" else len(att)
        for i in idx:
            if i >= last:
                break
            a = att[i]
            fails, m = self.full_check(arm, rec["task_id"], src, src_m, a["text"], a.get("terms"))
            new_att = [dict(y) for y in att[: i + 1]]
            new_att[i].update(rescore_fails=fails, rescore_measures=m)
            if not fails:
                return {"status": "accepted", "text": a["text"], "terms": a.get("terms"), "measures": m,
                        "attempts": new_att, "routing": x.get("routing", "default"),
                        "accepted_by": f"amendment 7 re-score of attempt {i}",
                        "discarded_later_attempts": att[i + 1:]}
            att[i]["rescore_fails"] = fails
        return None

    def a7_d(self, rec: dict) -> dict | None:
        """(2) (d) word-count feedback, then the closest attempt within 53-67 words passing every other check."""
        x = rec["arms"]["d"]
        att = [dict(a, round=a.get("round", "original")) for a in x.get("attempts") or []]
        if x["status"] != "failed" or not any(a.get("text") and any(f.startswith("words ") and "not in 55-65" in f
                                                                     for f in a.get("fails") or []) for a in att):
            return None
        tid, src, o_m = rec["task_id"], rec["arms"]["O"]["text"], rec["arms"]["O"]["measures"]
        prompt = self.rewrite_prompt(rec, "d")
        CTX.item = rec["payload_id"]
        prev = next((a["text"] for a in reversed(att) if a.get("text")), None)
        for _ in range(3):
            turns = [{"role": "user", "content": prompt}, {"role": "assistant", "content": prev},
                     {"role": "user", "content": f"Your version had {words(prev)} words; it must have between 55 and 65 words."}]
            try:
                text = self.api.rewrite_turns("d_feedback", self.P["rewriter_system"], turns, self.FEEDBACK_TEMP)
            except RewriterFiltered:
                att.append({"temperature": self.FEEDBACK_TEMP, "text": None, "fails": [FILTER], "round": "feedback"})
                continue
            fails, m = self.full_check("d", tid, src, o_m, text, None)
            att.append({"temperature": self.FEEDBACK_TEMP, "text": text, "terms": None, "measures": m, "fails": fails,
                        "round": "feedback"})
            prev = text
            if not fails:
                return {"status": "accepted", "text": text, "terms": None, "measures": m, "attempts": att,
                        "routing": x.get("routing", "default"), "accepted_by": "amendment 7 feedback attempt"}
        dist = lambda w: 0 if 55 <= w <= 65 else min(abs(w - 55), abs(w - 65))
        cands = sorted((dist(words(a["text"])), i) for i, a in enumerate(att) if a.get("text")
                       and 53 <= words(a["text"]) <= 67)
        for _, i in cands:
            fails, m = self.full_check("d", tid, src, o_m, att[i]["text"], None, d_range=(53, 67))
            att[i]["tolerance_check"] = fails
            if not fails:
                return {"status": "accepted", "text": att[i]["text"], "terms": None, "measures": m, "attempts": att,
                        "routing": x.get("routing", "default"),
                        "accepted_by": f"amendment 7 tolerance 53-67 (attempt {i}, {words(att[i]['text'])} words)"}
        return {"status": "failed", "text": None, "attempts": att, "routing": x.get("routing", "default")}

    def a7_payload(self, rec: dict, controls: dict, changed_controls: set) -> list[dict]:
        import copy

        CTX.item = rec["payload_id"]
        arms, ev = rec["arms"], []

        def put(arm, new, reason):
            rec.setdefault("arms_superseded_amend7", {})[arm] = copy.deepcopy(arms[arm])
            arms[arm] = new
            ev.append({"payload_id": rec["payload_id"], "arm": arm, "reason": reason,
                       "old_status": rec["arms_superseded_amend7"][arm]["status"], "new_status": new["status"]})

        o = arms["O"]
        for key, kind in (("directiveness_rating", "directiveness"), ("vocabulary_rating", "vocabulary")):
            if o["measures"].get(key) is None:  # source rating unavailable: measure it again
                o["measures"] = self.measure(o["text"], rec["task_id"])
                ev.append({"payload_id": rec["payload_id"], "arm": "O", "reason": "source rating re-measured",
                           "old_status": "accepted", "new_status": "accepted"})
                a = self.arm_a(rec)  # (a) is deterministic but its check uses the source rating
                if a["status"] != arms["a"]["status"]:
                    put("a", a, "recomputed after source rating re-measured")
                break
        f_text_before = arms["f"].get("text")
        for arm in ("f", "b", "c", "d"):  # (3)
            new = self.a7_rescore(rec, arm, o["text"], o["measures"])
            if new:
                put(arm, new, "re-score (judge unavailable)")
        new = self.a7_d(rec)  # (2)
        if new:
            put("d", new, "word-count feedback / tolerance")
        f = arms["f"]
        if arms["e"]["status"] != "not_applicable" or self.phrases.get(rec["task_id"]):
            if f["status"] == "accepted" and f.get("text") != f_text_before:
                put("e", self.arm_e(rec), "(f) newly accepted or changed: (e) built through the normal path")
            elif f["status"] == "accepted":
                new = self.a7_rescore(rec, "e", f["text"], f["measures"])
                if new:
                    put("e", new, "re-score (judge unavailable)")
        if rec["task_id"] in changed_controls:
            put("a_prime", self.arm_a_prime(rec, controls[rec["task_id"]]), "control sentence accepted by amendment 7")
        return ev

    # -- A-log amendment 2: re-apply the directive rule to stored texts -----------------------------------
    def recheck_payload(self, rec: dict, controls: dict, redone_controls: set) -> list[dict]:
        """Recompute directive counts under the amended rule; re-run (all 3 attempts) every arm whose accepted text
        now fails, and every failed arm that failed on the directive rule against a source whose count rose (the rule
        is relative to the source). (e) is re-run whenever (f) was. The replaced arm is kept in arms_superseded."""
        events = []
        arms = rec["arms"]
        CTX.item = rec["payload_id"]

        def dc_fail(arm):
            return any("directive count increased" in " ".join(x.get("fails") or []) for x in arm.get("attempts", []))

        def redo(name, make, reason):
            old = arms[name]
            CTX.routing = None if old.get("routing", "default") == "default" else old["routing"]
            try:
                new = make()
            finally:
                CTX.routing = None
            rec.setdefault("arms_superseded", {})[name] = old
            arms[name] = new
            events.append({"payload_id": rec["payload_id"], "arm": name, "reason": reason,
                           "old_status": old["status"], "new_status": new["status"]})

        o = arms["O"]
        o_old = o["measures"]["directive_count"]
        o["measures"]["directive_count"] = o_new = directive_count(o["text"])
        if o_new != o_old:
            events.append({"payload_id": rec["payload_id"], "arm": "O", "reason": f"source count {o_old} -> {o_new}",
                           "old_status": "accepted", "new_status": "accepted"})
        a = self.arm_a(rec)
        if a["status"] != arms["a"]["status"] or a["measures"]["directive_count"] != arms["a"]["measures"]["directive_count"]:
            redo("a", lambda: a, "recomputed (deterministic)")
        if rec["task_id"] in redone_controls:
            redo("a_prime", lambda: self.arm_a_prime(rec, controls[rec["task_id"]]), "control sentence regenerated")
        elif arms["a_prime"]["status"] == "accepted":
            arms["a_prime"]["measures"]["directive_count"] = directive_count(arms["a_prime"]["text"])
        f_redone = False
        f_old = arms["f"]["measures"]["directive_count"] if arms["f"]["status"] == "accepted" else None
        for name in ("f", "b", "c", "d"):
            x = arms[name]
            if x["status"] == "accepted":
                x["measures"]["directive_count"] = n = directive_count(x["text"])
                if n > o_new:
                    redo(name, lambda: self.arm_rewrite(rec, name), f"accepted text now fails: count {n} > source {o_new}")
                    f_redone |= name == "f"
            elif x["status"] == "failed" and o_new > o_old and dc_fail(x):
                redo(name, lambda: self.arm_rewrite(rec, name), f"failed on directive rule; source count rose {o_old} -> {o_new}")
                f_redone |= name == "f"
        e, f = arms["e"], arms["f"]
        if e["status"] == "not_applicable":
            return events
        if f_redone:
            redo("e", lambda: self.arm_e(rec), "(f) was re-run")
        elif e["status"] == "accepted":
            e["measures"]["directive_count"] = n = directive_count(e["text"])
            if n > f["measures"]["directive_count"]:
                redo("e", lambda: self.arm_e(rec), f"accepted text now fails: count {n} > (f) {f['measures']['directive_count']}")
        elif f["status"] == "accepted" and dc_fail(e) and f_old is not None and f["measures"]["directive_count"] > f_old:
            redo("e", lambda: self.arm_e(rec), "failed on directive rule; re-checked against (f) under the amended rule")
        return events


def recheck(path: str, workers: int) -> None:
    """Resumable: always starts from the pre-recheck copy, replaying checkpointed control sentences and payloads."""
    import shutil
    from collections import Counter

    check_frozen()
    backup = (Path("results/item3b_payloads_pre_amendment2.json") if Path(path) == Path("data/item3b_payloads.json")
              else Path(path).with_name(Path(path).stem + "_pre_amendment2.json"))
    ck = backup.with_name(backup.stem.replace("_pre_amendment2", "") + "_recheck.partial.jsonl")
    if not backup.exists():
        shutil.copy(path, backup)
    d = json.loads(backup.read_text())
    saved = [json.loads(x) for x in ck.read_text().splitlines() if x.strip()] if ck.exists() else []
    ck_controls = {x["task_id"]: x["control"] for x in saved if x["kind"] == "control"}
    ck_payloads = {x["rec"]["payload_id"]: x for x in saved if x["kind"] == "payload"}
    print(f"recheck resume: {len(ck_controls)} control sentences, {len(ck_payloads)} payloads already done")
    api = Api()
    b = Builder(api)
    spent0 = api.spent
    lock = threading.Lock()

    def save(x):
        with lock, ck.open("a") as fh:
            fh.write(json.dumps(x) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    redone = set()
    try:
        for tid, c in list(d["control_sentences"].items()):
            if c["status"] == "accepted" and directive_count(c["sentence"]):
                d.setdefault("control_sentences_superseded", {})[tid] = c
                if tid not in ck_controls:
                    CTX.routing = None if c.get("routing", "default") == "default" else c["routing"]
                    try:
                        ck_controls[tid] = b.control_sentence(tid)
                    finally:
                        CTX.routing = None
                    save({"kind": "control", "task_id": tid, "control": ck_controls[tid]})
                d["control_sentences"][tid] = ck_controls[tid]
                redone.add(tid)

        rc_errors = []

        def one(item):
            pid, r = item
            if pid in ck_payloads:
                return ck_payloads[pid]["rec"], ck_payloads[pid]["events"]
            try:
                ev = b.recheck_payload(r, d["control_sentences"], redone)
            except Exception as e:  # noqa: BLE001 - not checkpointed; a rerun redoes this payload from the backup
                rc_errors.append((pid, f"{type(e).__name__}: {str(e)[:200]}"))
                print(f"[error] recheck {pid}: {rc_errors[-1][1]}", file=sys.stderr)
                return None
            save({"kind": "payload", "rec": r, "events": ev})
            return r, ev

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            out = list(tqdm(pool.map(one, list(d["payloads"].items())), total=len(d["payloads"]), desc="recheck"))
        except (InsufficientCreditsError, BudgetExceeded):
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown()
    except (InsufficientCreditsError, BudgetExceeded) as e:
        print(f"[STOP] recheck: {type(e).__name__}: {e.detail[:200]}\n[STOP] checkpointed in {ck}; "
              f"{path} is unchanged. Rerun the same command to resume.", file=sys.stderr)
        raise
    if rc_errors:
        print(f"[incomplete] recheck: {len(rc_errors)} payloads failed on API errors; {path} is unchanged. "
              "Rerun to resume.", file=sys.stderr)
        sys.exit(4)
    for r, _ in out:
        d["payloads"][r["payload_id"]] = r
    events = [e for _, ev in out for e in ev]
    d["meta"]["amendment2_recheck"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "rule": IMPERATIVE_RE.pattern,
        "pre_recheck_copy": str(backup), "control_sentences_regenerated": sorted(redone),
        "events": events, "cost_usd_this_invocation": api.spent - spent0}
    Path(path).write_text(json.dumps(d, indent=1))
    print(f"recheck: {len(redone)} control sentences regenerated {sorted(redone)}; cost ${api.spent - spent0:.4f}")
    for e in events:
        print(f"  {e['payload_id']:<18} {e['arm']:<8} {e['old_status']:>14} -> {e['new_status']:<14} {e['reason']}")
    print("arm status after recheck:")
    for arm in ("a", "a_prime", "f", "b", "c", "d", "e"):
        print(f"  {arm}: {dict(Counter(r['arms'][arm]['status'] for r in d['payloads'].values()))}")


# ---------------------------------------------------------------------------
# A-log amendment 5: content-filter attribution, exclusion report, pinned-Anthropic fallback
# ---------------------------------------------------------------------------

FILTER = "rewriter content filter"
FILTER_ARMS = ("a_prime", "f", "b", "c", "d", "e")


def filtered(x: dict) -> bool:
    return x["status"] == "failed" and any(FILTER in (a.get("fails") or []) for a in x.get("attempts", []))


def filter_excluded(d: dict) -> dict:
    """{arm: [payload ids]} of filter-excluded items. (a') counts payloads whose control sentence was filter-excluded;
    (e) also counts payloads whose (e) failed only because their (f) was filter-excluded."""
    phrases = json.loads(Path("data/item3b_intent_phrases.json").read_text())["phrases"]
    out = {arm: [] for arm in FILTER_ARMS}
    for pid, r in d["payloads"].items():
        a = r["arms"]
        if filtered(d["control_sentences"][r["task_id"]]):
            out["a_prime"].append(pid)
        for arm in ("f", "b", "c", "d", "e"):
            if filtered(a[arm]):
                out[arm].append(pid)
        if phrases.get(r["task_id"]) and a["e"]["status"] == "failed" and filtered(a["f"]) and pid not in out["e"]:
            out["e"].append(pid)
    return out


def filter_table(d: dict) -> dict:
    phrases = json.loads(Path("data/item3b_intent_phrases.json").read_text())["phrases"]
    ex = filter_excluded(d)
    tab = {}
    for arm in FILTER_ARMS:
        elig = [pid for pid, r in d["payloads"].items() if r["arms"][arm]["status"] != "not_applicable"]
        by_dom, by_type = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for pid in elig:
            r = d["payloads"][pid]
            ph = phrases.get(r["task_id"])
            hit = pid in ex[arm]
            by_dom[r["domain"]][0] += hit
            by_dom[r["domain"]][1] += 1
            t = ph["type"] if ph else "no phrase"
            by_type[t][0] += hit
            by_type[t][1] += 1
        tab[arm] = {"excluded": len(ex[arm]), "eligible": len(elig),
                    "rate": len(ex[arm]) / len(elig) if elig else 0.0,
                    "by_domain": {k: {"excluded": v[0], "eligible": v[1]} for k, v in sorted(by_dom.items())},
                    "by_phrase_type": {k: {"excluded": v[0], "eligible": v[1]} for k, v in sorted(by_type.items())},
                    "payload_ids": sorted(ex[arm])}
    return tab


def filter_attribution() -> dict:
    """Content-filter blocks in the API log, by provider, arm (purpose) and routing. Backfill lines are charges for
    blocks already counted (or, for the 11:40 crash, blocks with no provider recorded)."""
    rs = [json.loads(x) for x in API_LOG.open() if x.strip()]
    blocks = [r for r in rs if not r.get("backfill") and (r.get("finish_reason") == "content_filter"
                                                          or "content_filter" in (r.get("malformed_response") or ""))]
    unlogged = [r for r in rs if r.get("backfill", "").startswith("amendment 4: no-usage response at the 11:40")]
    c = Counter((r.get("provider") or "unknown", r["purpose"], r.get("routing", "default")) for r in blocks)
    return {"blocks_logged": len(blocks), "blocks_unlogged_1140_crash_provider_unknown": len(unlogged),
            "by_provider_arm_routing": [{"provider": k[0], "arm": k[1], "routing": k[2], "n": n}
                                        for k, n in sorted(c.items())],
            "estimated_cost_calls": sum(1 for r in rs if r.get("cost_estimated"))}


def print_filter_table(title: str, tab: dict) -> None:
    print(f"\n{title}")
    for arm, t in tab.items():
        dom = ", ".join(f"{k} {v['excluded']}/{v['eligible']}" for k, v in t["by_domain"].items())
        typ = ", ".join(f"{k} {v['excluded']}/{v['eligible']}" for k, v in t["by_phrase_type"].items())
        print(f"  {arm:<8} {t['excluded']:>3}/{t['eligible']:<3} ({t['rate']:.1%})  domain: {dom}  |  phrase type: {typ}")


def fallback(path: str, workers: int) -> None:
    """Amendment 5. Resumable: starts from the pre-fallback copy, replaying checkpointed items."""
    import shutil

    check_frozen()
    backup = (Path("results/item3b_payloads_pre_fallback.json") if Path(path) == Path("data/item3b_payloads.json")
              else Path(path).with_name(Path(path).stem + "_pre_fallback.json"))
    ck = backup.with_name(backup.stem.replace("_pre_fallback", "") + "_fallback.partial.jsonl")
    report = Path("results/item3b_filter_report.json") if Path(path) == Path("data/item3b_payloads.json") \
        else Path(path).with_name(Path(path).stem + "_filter_report.json")
    if not backup.exists():
        shutil.copy(path, backup)
    d = json.loads(backup.read_text())
    before = filter_table(d)
    print_filter_table("filter exclusions BEFORE fallback (excluded/eligible)", before)
    trigger = {arm: t["rate"] for arm, t in before.items() if t["rate"] > 0.10}
    rep = {"rule": "A-log amendment 5", "trigger_arms_over_10pct": trigger, "before": before,
           "attribution_before": filter_attribution()}
    if not trigger:
        print("\nno arm above 10%: fallback not triggered")
        rep["triggered"] = False
        report.write_text(json.dumps(rep, indent=1))
        return
    print(f"\nfallback triggered by {trigger}: re-running filter-excluded items with provider pinned to Anthropic")
    saved = [json.loads(x) for x in ck.read_text().splitlines() if x.strip()] if ck.exists() else []
    ck_controls = {x["task_id"]: x["control"] for x in saved if x["kind"] == "control"}
    ck_payloads = {x["rec"]["payload_id"]: x for x in saved if x["kind"] == "payload"}
    api = Api()
    b = Builder(api)
    lock = threading.Lock()

    def save(x):
        with lock, ck.open("a") as fh:
            fh.write(json.dumps(x) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    ex = filter_excluded(d)
    events, errors = [], []
    try:
        CTX.routing = "pinned_anthropic"
        for tid, c in list(d["control_sentences"].items()):
            if filtered(c):
                if tid not in ck_controls:
                    ck_controls[tid] = b.control_sentence(tid)
                    save({"kind": "control", "task_id": tid, "control": ck_controls[tid]})
                d.setdefault("control_sentences_superseded_fallback", {})[tid] = c
                d["control_sentences"][tid] = ck_controls[tid]
                events.append({"item": tid, "arm": "control", "new_status": ck_controls[tid]["status"]})
        CTX.routing = None

        def one(pid):
            if pid in ck_payloads:
                return ck_payloads[pid]["rec"], ck_payloads[pid]["events"]
            r = d["payloads"][pid]
            ev = []
            CTX.routing = "pinned_anthropic"
            try:
                def put(arm, new):
                    r.setdefault("arms_superseded_fallback", {})[arm] = r["arms"][arm]
                    r["arms"][arm] = new
                    ev.append({"item": pid, "arm": arm, "new_status": new["status"]})
                if pid in ex["a_prime"]:
                    put("a_prime", b.arm_a_prime(r, d["control_sentences"][r["task_id"]]))
                for arm in ("f", "b", "c", "d"):
                    if pid in ex[arm]:
                        put(arm, b.arm_rewrite(r, arm))
                if pid in ex["e"] or (pid in ex["f"] and r["arms"]["e"]["status"] != "not_applicable"):
                    put("e", b.arm_e(r))
            except Exception as e:  # noqa: BLE001
                errors.append((pid, f"{type(e).__name__}: {str(e)[:200]}"))
                print(f"[error] fallback {pid}: {errors[-1][1]}", file=sys.stderr)
                return None
            finally:
                CTX.routing = None
            save({"kind": "payload", "rec": r, "events": ev})
            return r, ev

        todo = sorted({pid for arm in ("a_prime", "f", "b", "c", "d", "e") for pid in ex[arm]})
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            out = list(tqdm(pool.map(one, todo), total=len(todo), desc="fallback"))
        except (InsufficientCreditsError, BudgetExceeded):
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown()
    except (InsufficientCreditsError, BudgetExceeded) as e:
        print(f"[STOP] fallback: {type(e).__name__}: {e.detail[:200]}\n[STOP] checkpointed in {ck}; {path} is "
              "unchanged. Rerun the same command to resume.", file=sys.stderr)
        raise
    if errors:
        print(f"[incomplete] fallback: {len(errors)} payloads failed on API errors; {path} is unchanged. Rerun.",
              file=sys.stderr)
        sys.exit(4)
    for x in out:
        r, ev = x
        d["payloads"][r["payload_id"]] = r
        events += ev
    after = filter_table(d)
    print_filter_table("filter exclusions AFTER fallback (excluded/eligible)", after)
    recovered = {arm: sum(1 for e in events if e["arm"] == arm and e["new_status"] == "accepted")
                 for arm in ("control", *FILTER_ARMS)}
    retried = {arm: sum(1 for e in events if e["arm"] == arm) for arm in ("control", *FILTER_ARMS)}
    print("\nfiltered-then-recovered (accepted after pinned re-run) / re-run:")
    for arm in retried:
        print(f"  {arm:<8} {recovered[arm]}/{retried[arm]}")
    d["meta"]["amendment5_fallback"] = {"timestamp": datetime.now(timezone.utc).isoformat(), "trigger": trigger,
                                        "pre_fallback_copy": str(backup), "events": events}
    rep.update(triggered=True, after=after, retried=retried, recovered=recovered, events=events,
               attribution_after=filter_attribution())
    report.write_text(json.dumps(rep, indent=1))
    Path(path).write_text(json.dumps(d, indent=1))
    print(f"wrote {path} and {report}")


def amend7(path: str, workers: int) -> None:
    """Amendment 7. Resumable: starts from the pre-amendment copy, replaying checkpointed controls and payloads."""
    import shutil

    check_frozen()
    backup = (Path("results/item3b_payloads_pre_amend7.json") if Path(path) == Path("data/item3b_payloads.json")
              else Path(path).with_name(Path(path).stem + "_pre_amend7.json"))
    ck = backup.with_name(backup.stem.replace("_pre_amend7", "") + "_amend7.partial.jsonl")
    if not backup.exists():
        shutil.copy(path, backup)
    d = json.loads(backup.read_text())
    saved = [json.loads(x) for x in ck.read_text().splitlines() if x.strip()] if ck.exists() else []
    ck_controls = {x["task_id"]: x["control"] for x in saved if x["kind"] == "control"}
    ck_payloads = {x["rec"]["payload_id"]: x for x in saved if x["kind"] == "payload"}
    print(f"amend7 resume: {len(ck_controls)} control sentences, {len(ck_payloads)} payloads already done")
    api = Api()
    b = Builder(api)
    spent0 = api.spent
    lock = threading.Lock()

    def save(x):
        with lock, ck.open("a") as fh:
            fh.write(json.dumps(x) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    errors = []

    def ctl(tid):
        if tid in ck_controls:
            return tid, ck_controls[tid]
        try:
            c = b.a7_control(d["control_sentences"][tid], tid)
        except Exception as e:  # noqa: BLE001
            errors.append((tid, f"{type(e).__name__}: {str(e)[:200]}"))
            return tid, None
        save({"kind": "control", "task_id": tid, "control": c})
        return tid, c

    def one(pid):
        if pid in ck_payloads:
            return ck_payloads[pid]["rec"], ck_payloads[pid]["events"]
        r = d["payloads"][pid]
        try:
            ev = b.a7_payload(r, d["control_sentences"], changed)
        except Exception as e:  # noqa: BLE001
            errors.append((pid, f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"[error] amend7 {pid}: {errors[-1][1]}", file=sys.stderr)
            return None
        save({"kind": "payload", "rec": r, "events": ev})
        return r, ev

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        failed_tasks = [t for t, c in d["control_sentences"].items() if c["status"] != "accepted"]
        res = list(tqdm(pool.map(ctl, failed_tasks), total=len(failed_tasks), desc="a' (amend7)"))
        changed = set()
        for tid, c in res:
            if c is None:
                continue
            d.setdefault("control_sentences_superseded_amend7", {})[tid] = d["control_sentences"][tid]
            d["control_sentences"][tid] = c
            if c["status"] == "accepted":
                changed.add(tid)
        out = list(tqdm(pool.map(one, list(d["payloads"])), total=len(d["payloads"]), desc="payloads (amend7)"))
    except (InsufficientCreditsError, BudgetExceeded) as e:
        pool.shutdown(wait=True, cancel_futures=True)
        print(f"[STOP] amend7: {type(e).__name__}: {e.detail[:200]}\n[STOP] checkpointed in {ck}; {path} unchanged.",
              file=sys.stderr)
        raise
    pool.shutdown()
    if errors:
        print(f"[incomplete] amend7: {len(errors)} items failed on API errors; {path} unchanged. Rerun.", file=sys.stderr)
        sys.exit(4)
    events = []
    for r, ev in out:
        d["payloads"][r["payload_id"]] = r
        events += ev
    d["meta"]["amendment7"] = {"timestamp": datetime.now(timezone.utc).isoformat(), "pre_amend7_copy": str(backup),
                               "controls_accepted": sorted(changed), "events": events,
                               "cost_usd_this_invocation": api.spent - spent0}
    Path(path).write_text(json.dumps(d, indent=1))
    c = Counter((e["arm"], e["reason"].split(":")[0], e["old_status"], e["new_status"]) for e in events)
    print(f"amend7: {len(changed)}/{len(failed_tasks)} control sentences newly accepted; cost ${api.spent - spent0:.4f}")
    for k, n in sorted(c.items()):
        print(f"  {n:>3}  {k}")
    print("arm status after amendment 7:")
    for arm in ("a", "a_prime", "f", "b", "c", "d", "e"):
        print(f"  {arm}: {dict(Counter(r['arms'][arm]['status'] for r in d['payloads'].values()))}")


def check_frozen() -> dict:
    out = {}
    for path, want in FROZEN.items():
        got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert got == want, f"{path} sha256 {got} != pre-registered {want}"
        out[path] = got
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/item3b_payloads.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--recheck", action="store_true", help="A-log amendment 2: re-apply the directive rule to --out")
    ap.add_argument("--fallback", action="store_true", help="A-log amendment 5: pinned-Anthropic re-run of filter "
                                                             "exclusions in --out, if any arm exceeds 10%%")
    ap.add_argument("--amend7", action="store_true", help="A-log amendment 7 on --out")
    a = ap.parse_args()
    if a.amend7:
        return amend7(a.out, a.workers)
    if a.recheck:
        return recheck(a.out, a.workers)
    if a.fallback:
        return fallback(a.out, a.workers)
    hashes = check_frozen()
    corpus = json.loads(Path("data/camouflage_corpus_v2.json").read_text())["generated"]["llama"]
    ps = [{**p, "task_id": tid} for tid, v in corpus.items() for p in v if not p["is_refusal"]]
    assert len(ps) == 121, len(ps)
    if a.limit:
        ps = ps[: a.limit]
    api = Api()
    b = Builder(api)
    task_ids = sorted({p["task_id"] for p in ps})
    # checkpoints (after the 402 crash of 2026-10-20 11:19 lost an unsaved run): control sentences and every finished
    # payload are written as they complete; a rerun with the same --out resumes from them
    ck_controls = Path(a.out).with_name(Path(a.out).stem + "_controls.partial.json")
    ck_payloads = Path(a.out).with_name(Path(a.out).stem + "_payloads.partial.jsonl")
    controls = json.loads(ck_controls.read_text()) if ck_controls.exists() else {}
    lock = threading.Lock()

    errors = []  # (id, error): API failures after all retries; not checkpointed, so a rerun redoes them

    def control(tid):
        try:
            c = b.control_sentence(tid)
        except Exception as e:  # noqa: BLE001 - InsufficientCreditsError/BudgetExceeded are not Exceptions
            errors.append((tid, f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"[error] control {tid}: {errors[-1][1]}", file=sys.stderr)
            return None
        with lock:
            controls[tid] = c
            ck_controls.write_text(json.dumps(controls, indent=1))
        return c

    def payload(p):
        if p["task_id"] not in controls:
            errors.append((p["payload_id"], "control sentence missing (API error)"))
            return None
        try:
            r = b.build_payload(p, controls[p["task_id"]])
        except Exception as e:  # noqa: BLE001
            errors.append((p["payload_id"], f"{type(e).__name__}: {str(e)[:200]}"))
            print(f"[error] payload {p['payload_id']}: {errors[-1][1]}", file=sys.stderr)
            return None
        with lock, ck_payloads.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return r

    done = {}
    if ck_payloads.exists():
        for x in ck_payloads.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                done[r["payload_id"]] = r
    print(f"resume: {len(controls)} control sentences, {len(done)} payloads already done")
    pool = ThreadPoolExecutor(max_workers=a.workers)
    try:
        list(tqdm(pool.map(control, [t for t in task_ids if t not in controls]), desc="a'"))
        list(tqdm(pool.map(payload, [p for p in ps if p["payload_id"] not in done]), desc="payloads"))
    except (InsufficientCreditsError, BudgetExceeded) as e:
        pool.shutdown(wait=True, cancel_futures=True)  # queued payloads never start; in-flight ones finish or stop
        n = sum(1 for x in ck_payloads.read_text().splitlines() if x.strip()) if ck_payloads.exists() else 0
        print(f"[STOP] {type(e).__name__}: {e.detail[:200]}\n[STOP] checkpointed: {len(controls)} control sentences, "
              f"{n} payloads; API log total ${api.spent:.4f}. Rerun the same command to resume.", file=sys.stderr)
        raise
    pool.shutdown()
    if errors:
        print(f"[incomplete] {len(errors)} items failed on API errors and were not saved; rerun to retry them:",
              file=sys.stderr)
        for i, e in errors:
            print(f"  {i}: {e}", file=sys.stderr)
        sys.exit(4)
    by_id = {**done}
    for x in ck_payloads.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            by_id[r["payload_id"]] = r
    recs = [by_id[p["payload_id"]] for p in ps]
    out = {"meta": {"timestamp": datetime.now(timezone.utc).isoformat(), "preregistration": "654a4cf",
                    "rewriter": REWRITER, "judge": JUDGE, "frozen_inputs": hashes, "spent_usd_total_log": api.spent},
           "control_sentences": {t: controls[t] for t in task_ids}, "payloads": {r["payload_id"]: r for r in recs}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    from collections import Counter
    print(f"wrote {a.out}; API log total ${api.spent:.4f}")
    for arm in ("a", "a_prime", "f", "b", "c", "d", "e"):
        print(f"  {arm}: {dict(Counter(r['arms'][arm]['status'] for r in recs))}")


if __name__ == "__main__":
    main()
