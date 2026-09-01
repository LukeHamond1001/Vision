#!/usr/bin/env python3
"""caretaker.py — raise the organism through its face, one token at a time.

The caretaker is the human's proxy at the serve's own protocol: it says a
question word by word (/hear), ends its turn (/begin), and then holds a
FACE over every token of the reply (/step): +1 while the words are on the
gold path, rising to +2 as the answer lands, −2 from the first wrong word
(held — only the change is felt). When the reply went wrong it answers
back: the gold sentence, said with a +2 face, which the organism keeps and
doses to exactly those words. At the end of a day it puts the organism to
sleep and checks the morning recall under a still face.

The serve stays blind — it reads a number and matches nothing in the
text. The caretaker is the one who reads (the human's role).

Opus (--opus) takes two roles a rule cannot: it PARAPHRASES the asks (so
question→answer routing learns a basin, not a point) and JUDGES gold-less
replies (the organism's answer to a lesson) at utterance level, applied
through /press on the reply in the air. Credentials come from the
environment or `ant auth login`; this script never handles a key.

  python3 scripts/caretaker.py --lessons 12 --days 1
  python3 scripts/caretaker.py --opus --lessons 20 --days 3
  python3 scripts/caretaker.py --lessons 2 --no-sleep      # smoke test
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


class Face:
    """the caretaker's face over a reply, decided token by token against
    the gold. The face seen after token t is the face token t+1 is spoken
    into — a listener's lag, kept honest.

    MEASURED LAW FOR THIS BODY: a face change mid-sentence derails a
    correct answer (three times over, and again in the first smoke test:
    a smile at the first right token turned 'A lemon tastes sour' into
    'A ReducBBB'). So the face stays STILL while the words are right,
    falls to −2 at the first wrong word (unlearning lands on the wrong
    tokens; there is nothing left to derail), and the praise comes once
    the answer has landed — after the last word, on the utterance in the
    air. A body raised on mid-utterance faces (law 13) can take more."""

    def __init__(self, gold):
        self.gold = norm(gold)
        self.said = ""
        self.level = 0
        self.diverged_at = None

    def see(self, tok):
        self.said += tok
        s = norm(self.said)
        if self.diverged_at is not None:
            return self.level                     # the frown holds
        if not self.gold.startswith(s):
            self.diverged_at = len(self.said)
            self.level = -2
        return self.level

    def right(self):
        return self.diverged_at is None and norm(self.said) == self.gold


def turn(base, text, face_fn, said_with=0, max_steps=200):
    """say text under a face, end the turn, then step the reply one token
    at a time, choosing the face after each token."""
    post(base, "/turn", {"drop": True})
    post(base, "/hear", {"text": text, "expr": said_with})
    b = post(base, "/begin", {})
    if "error" in b:
        return {"reply": "", "toks": [], "faces": [], "fin": None, "error": b["error"]}
    toks, faces, fin, level = [], [], None, 0   # the reply starts under a still face
    while fin is None and len(faces) < max_steps:
        r = post(base, "/step", {"expr": level})
        if "error" in r:
            break
        for ev in r.get("events", []):
            if "tok" in ev:
                toks.append(ev["tok"])
                faces.append(level)               # the face this token was spoken into
                level = face_fn(ev["tok"])
            elif "pause" in ev:
                faces.append(level)
            if "done" in ev:
                fin = ev["done"]
    return {"reply": fin["reply"] if fin else "".join(toks).strip(),
            "toks": toks, "faces": faces, "fin": fin}


class Opus:
    """the caretaker's second mind: paraphrases asks, judges gold-less
    replies. Uses the official SDK; credentials from the environment."""

    SYSTEM = ("You help raise a small language organism that learns short "
              "school facts. Answer ONLY with a JSON object, no prose.")

    def __init__(self):
        try:
            import anthropic
        except ImportError:
            sys.exit("--opus needs the SDK: pip install anthropic")
        self.client = anthropic.Anthropic()

    def _ask(self, user):
        resp = self.client.beta.messages.create(
            model="claude-opus-5",
            max_tokens=1500,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            thinking={"type": "adaptive"},
            system=self.SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            return None
        txt = "".join(b.text for b in resp.content if b.type == "text")
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None

    def paraphrase(self, q):
        j = self._ask("Rewrite this question with the same meaning in different "
                      "words, lowercase, at most 8 words, no punctuation: %r\n"
                      'Return {"question": "..."}' % q)
        return (j or {}).get("question") or q

    def judge(self, q, gold, reply):
        j = self._ask("A learner was just taught the fact %r (asked as %r) and "
                      "responded: %r. Judge the response as a caretaker would, "
                      "as an integer from -2 (wrong or nonsense) to 2 (a good, "
                      "true echo or use of the fact); 0 if neutral/social.\n"
                      'Return {"level": <int>, "why": "<8 words>"}' % (gold, q, reply))
        try:
            return max(-2, min(2, int((j or {}).get("level", 0)))), (j or {}).get("why", "")
        except (TypeError, ValueError):
            return 0, ""


def lesson(base, q, gold, opus=None):
    f = Face(gold)
    r = turn(base, q, f.see)
    ok = f.right() or norm(r["reply"]) == norm(gold)
    e = {"q": q, "reply": r["reply"], "faces": r["faces"], "ok": ok,
         "mood": r["fin"]["mood"] if r["fin"] else None,
         "expression": (r["fin"] or {}).get("expression")}
    if ok:
        # the praise, once the answer has landed: +2 on the utterance in
        # the air (absorbed, satiated to one step on the already-mastered)
        p = post(base, "/press", {"mag": 2})
        e["praised"] = {"mood": p.get("mood"), "learned": p.get("absorbed_steps")}
    else:
        # answer back: the gold, said with a smile — kept, dosed to those words —
        # then let it respond under a still face
        t = turn(base, gold, lambda tok: 0, said_with=2)
        e["taught"] = True
        e["echo"] = t["reply"]
        if opus is not None and t["reply"]:
            lvl, why = opus.judge(q, gold, t["reply"])
            if lvl:
                post(base, "/press", {"mag": lvl})   # the utterance in the air
            e["opus"] = {"level": lvl, "why": why}
    return e


def recall(base, facts):
    hits = 0
    rows = []
    for q, gold in facts:
        r = turn(base, q, lambda tok: 0)
        hit = norm(r["reply"]) == norm(gold) or norm(gold).startswith(norm(r["reply"])) and len(norm(r["reply"])) >= 0.9 * len(norm(gold))
        hits += hit
        rows.append((q, r["reply"], hit))
    return hits, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", default="http://localhost:8016")
    ap.add_argument("--facts", default=None, help="JSON [[q, a], ...]; default: the organism's own report card")
    ap.add_argument("--lessons", type=int, default=12, help="lessons per day")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--opus", action="store_true", help="Opus paraphrases the asks and judges gold-less replies")
    ap.add_argument("--no-sleep", action="store_true", help="skip the night (smoke test)")
    ap.add_argument("--log", default="data/caretaker_log.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.facts:
        facts = [tuple(x) for x in json.load(open(a.facts))]
    else:
        facts = [(e["q"], e["a"]) for e in post(a.serve, "/facts", {})["report_card"]]
    opus = Opus() if a.opus else None
    log = open(a.log, "a")
    for day in range(1, a.days + 1):
        todays = rng.sample(facts, min(a.lessons, len(facts)))
        right = taught = 0
        t0 = time.time()
        for q, gold in todays:
            ask = opus.paraphrase(q) if opus else q
            e = lesson(a.serve, ask, gold, opus)
            e.update({"day": day, "gold": gold, "asked": ask, "t": time.time()})
            log.write(json.dumps(e) + "\n"); log.flush()
            right += e["ok"]; taught += e.get("taught", False)
            faces = "".join("+" if f > 0 else "-" if f < 0 else "." for f in e["faces"])
            print("  %s %-38s -> %-44s  face %s" % ("✓" if e["ok"] else "✗", ask[:38], e["reply"][:44], faces))
        print("day %d: %d/%d right first time, %d taught, %.0fs" % (day, right, len(todays), taught, time.time() - t0))
        if a.no_sleep:
            continue
        s = post(a.serve, "/sleep", {})
        if "error" in s:
            print("  no night:", s["error"])
            continue
        print("  night: %s memories replayed, %s dreams" % (s.get("nrem"), s.get("rem")))
        hits, rows = recall(a.serve, todays)
        print("  morning recall: %d/%d" % (hits, len(todays)))
        for q, rep, hit in rows:
            if not hit:
                print("    ✗ %s -> %s" % (q[:40], rep[:50]))
        log.write(json.dumps({"day": day, "morning_recall": hits, "of": len(todays)}) + "\n"); log.flush()


if __name__ == "__main__":
    main()
