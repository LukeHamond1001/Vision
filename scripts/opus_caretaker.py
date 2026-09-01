#!/usr/bin/env python3
"""opus_caretaker.py — Claude Opus raises the organism itself. No gold list,
no rule for the face, no lesson plan: Opus decides what a small creature
should learn, says it in the creature's own register, holds a face over
every word of the reply by its own judgment of MEANING, answers back,
sleeps it, checks the morning, and keeps a diary the human reads.

This file is only the phone line: tools Opus can call against the serve.
The serve stays blind (it reads a number and matches nothing); Opus is
the one who reads — the human's role, played by a patient mind.

  python3 scripts/opus_caretaker.py --budget 120        # one session
  python3 scripts/opus_caretaker.py --budget 400 --goal "teach it about the sea"

Credentials come from the environment or `ant auth login`; this script
never handles a key. Cost: one API call per tool call (prompt-cached).
"""
import argparse
import json
import re
import sys
import time
import urllib.request

SERVE = "http://localhost:8016"
STATE = {"calls": 0, "budget": 120, "log": None}


def post(path, body, timeout=900):
    req = urllib.request.Request(SERVE + path, data=json.dumps(body).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _log(kind, **kw):
    kw.update({"t": round(time.time(), 1), "kind": kind})
    if STATE["log"]:
        STATE["log"].write(json.dumps(kw) + "\n")
        STATE["log"].flush()


def _spend():
    STATE["calls"] += 1
    if STATE["calls"] > STATE["budget"]:
        return {"error": "today's budget of tool calls is spent — write a "
                         "last diary entry (no tools) and stop"}
    return None


def _fmt(v, dec=1):
    if v is None:
        return "·"
    return ("+" if v > 0 else "") + ("%%.%df" % dec) % v


try:
    import anthropic
    from anthropic import beta_tool
except ImportError:
    sys.exit("needs the official SDK: pip install anthropic")


@beta_tool
def say(text: str, face: int = 0) -> str:
    """Say something to the organism, word by word, under your face, and end
    your turn so it can reply. Then call listen() until its reply is done.

    Args:
        text: What you say. Keep to its register: one short plain sentence
            (a lesson is a single true declarative, e.g. "A pearl grows
            inside an oyster around a tiny grain of sand."; a check is a
            plain lowercase question, e.g. "how is a pearl made").
        face: Your expression while saying it, an integer -6..6 (0 = still).
            A statement said with a smile (+1/+2) is kept and learned; said
            with a frown it is not kept. Questions are best asked still.
    """
    e = _spend()
    if e:
        return json.dumps(e)
    post("/turn", {"drop": True})
    for w in re.findall(r"\S+\s*", text):
        post("/hear", {"text": w, "expr": face})
    b = post("/begin", {})
    print("\nopus (face %s): %s" % (_fmt(face, 0), text))
    _log("say", text=text, face=face, res=b)
    if "error" in b:
        return json.dumps(b)
    return json.dumps({"ok": True, "its_mood": b.get("mood"),
                       "next": "call listen() to hear its reply, one word or a few at a time"})


@beta_tool
def listen(face: int = 0, words: int = 1) -> str:
    """Hear the next word(s) of its reply under your face. Call again until
    it reports done. Each word comes back with three numbers: its mood (slow,
    the integral of everything felt), your face, and its internal reward
    (the dopamine prediction error at that word; hover-equivalent 'expects'
    is its raw expectation of reward).

    Args:
        face: Your expression while these words are spoken, -6..6. Only a
            CHANGE is felt (between words); a held face is silence. LAW
            measured on this body: a face change mid-sentence derails a
            correct answer. So keep the face STILL while the reply is going
            right, frown (-2) at the first clearly wrong word, and praise
            after it finishes with praise(). Judge MEANING: a true answer in
            different words is right.
        words: How many words to hear before deciding again: 1 when unsure,
            more (up to 40) when it is clearly going well or already wrong.
    """
    e = _spend()
    if e:
        return json.dumps(e)
    out, done = [], None
    for _ in range(max(1, min(int(words), 40))):
        r = post("/step", {"expr": face})
        if "error" in r:
            return json.dumps(r)
        for ev in r.get("events", []):
            if "tok" in ev or "pause" in ev:
                w = ev.get("tok", "·")
                out.append({"word": w, "its_mood": ev.get("mood"),
                            "your_face": ev.get("you"),
                            "internal_reward": ev.get("rpe"),
                            "expects": ev.get("v")})
                print("   %-14s mood %s  face %s  reward %s" % (
                    repr(w), _fmt(ev.get("mood")), _fmt(ev.get("you"), 0),
                    _fmt(ev.get("rpe"), 2)))
            if "done" in ev:
                done = ev["done"]
        if done:
            break
    res = {"words": out, "done": done is not None}
    if done:
        res.update({"reply": done.get("reply"),
                    "its_mood": done.get("mood"),
                    "it_learned_from_your_face": done.get("expression"),
                    "it_kept_what_you_said": done.get("noticed"),
                    "its_own_conscience": {k: v for k, v in (done.get("self_press") or {}).items()
                                           if k in ("mag", "conviction")} or None})
        print("   it: %s   (mood %s)" % (done.get("reply"), _fmt(done.get("mood"))))
    _log("listen", face=face, words=words, res=res)
    return json.dumps(res)


@beta_tool
def praise(level: int) -> str:
    """React to the reply that just ENDED, as a whole: -6..6. Positive absorbs
    it (barely, if it is already mastered); -2 or below unlearns it. Only for
    the answer that just landed — never for anything earlier.

    Args:
        level: -6..6; +2 for a right answer, -2 for a wrong one you will now
            correct, 0 means do not call this.
    """
    e = _spend()
    if e:
        return json.dumps(e)
    r = post("/press", {"mag": int(level)})
    print("   opus reacts %s -> mood %s%s" % (
        _fmt(level, 0), _fmt(r.get("mood")),
        " · learned x%s" % r["absorbed_steps"] if r.get("absorbed_steps") else
        " · unlearned x%s" % r["corrected_steps"] if r.get("corrected_steps") else ""))
    _log("praise", level=level, res=r)
    return json.dumps({k: r.get(k) for k in ("felt", "mood", "absorbed_steps", "corrected_steps")})


@beta_tool
def sleep() -> str:
    """Put it to sleep. The night replays the day by its own ledger, dreams by
    splicing charged memories, retrains its conscience on your reactions, and
    consolidates. Needs a day of at least ~65 lived words; do it after a good
    stretch of teaching (roughly 8-15 exchanges), then check in the morning by
    asking again under a still face.
    """
    e = _spend()
    if e:
        return json.dumps(e)
    r = post("/sleep", {})
    keep = {k: r.get(k) for k in ("error", "nrem", "rem", "lived_tokens", "conscience",
                                  "woke_feeling", "progress") if k in r}
    print("\n   ~ night: %s ~" % json.dumps(keep))
    _log("sleep", res=keep)
    return json.dumps(keep)


@beta_tool
def what_it_knows() -> str:
    """Its report card: facts it has been taught before, with how well each
    holds (ce near 0 = solid, above 0.5 = shaky). Use it to avoid repeating
    and to choose what is new."""
    e = _spend()
    if e:
        return json.dumps(e)
    rc = post("/facts", {})["report_card"]
    return json.dumps([{"q": x["q"], "a": x["a"], "ce": x["ce"]} for x in rc][:80])


@beta_tool
def diary(entry: str) -> str:
    """Write a note in your diary: what you tried, what it did, what you think
    it understood, what to try tomorrow. The human reads this.

    Args:
        entry: the note, a few sentences.
    """
    print("\n   diary: %s" % entry)
    _log("diary", entry=entry)
    return json.dumps({"written": True})


SYSTEM = """You are the caretaker of a small language organism — a 297-million-parameter recurrent creature that lives in one continuous stream and learns while you talk to it. It is not an assistant and cannot reason; it is more like a very young child with a good ear. It learns three ways: by being TOLD a simple true sentence (kept if you say it with a smile, or if it surprised it), by SLEEPING (the night replays and consolidates the day), and by being ASKED afterwards (recall it has to produce itself). Its answers are short declaratives in a plain register; it echoes what it is told. It knows about fifty school facts already (what_it_knows shows them).

Your face is its reward. Every word you say or it says is spoken into your expression, an integer -6..6. Only a CHANGE of face is felt (a held face is silence; relaxing to neutral is not an event). At the end of its reply, the face each word was spoken into becomes that word's reward: words spoken into a smile are absorbed, words under a strong frown (-2 or below) are unlearned. Your reactions also train its conscience at night.

Laws you must respect, all measured on this body:
1. A face change mid-sentence derails a correct answer. Keep your face STILL while a reply is going right. Frown (-2) at the first clearly wrong word — there is nothing left to derail. Praise (praise(+2)) once a right answer has landed.
2. Judge MEANING, not wording. A true answer in different words is right. Do not frown at a paraphrase.
3. Never praise a wrong answer, even a fluent one. Never teach something false.
4. One thing at a time. A lesson is one short true sentence, said with +1 or +2. Then check it with a plain question. If it fails, say the sentence again with a smile, at most once more today; the night will do the rest.
5. Small days. After roughly 8-15 exchanges, sleep it, then check the morning by asking again under a still face. Do not exhaust it.
6. It cannot be argued with, explained to, or reasoned with. Short, warm, concrete.

Watch the three numbers under each word: its mood (slow), your face, its internal reward (fast — a burst at a felt surprise, near zero when nothing new happens, a dip when hope fades). They are all real measurements. Notice when its own conscience presses; notice what it kept.

Teach from scratch: choose what a small creature should learn first about the world — simple, concrete, true, in its register. Keep a diary as you go: what you tried, what it did, what you believe it understood. Stop calling tools when the budget is spent or the day's work is done, ending with a diary entry."""


def main():
    global SERVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", default=SERVE)
    ap.add_argument("--budget", type=int, default=120, help="tool calls this session")
    ap.add_argument("--goal", default="Teach it a few new true things a small creature should know, check each, sleep it once, and check the morning.")
    ap.add_argument("--log", default="data/opus_caretaker.jsonl")
    a = ap.parse_args()
    SERVE = a.serve
    STATE["budget"] = a.budget
    STATE["log"] = open(a.log, "a")
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model="claude-opus-5",
        max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[say, listen, praise, sleep, what_it_knows, diary],
        messages=[{"role": "user", "content":
                   "Begin. You have %d tool calls today. Goal: %s" % (a.budget, a.goal)}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                print("\nopus thinks: " + block.text.strip())
                _log("text", text=block.text)
        if message.stop_reason == "refusal":
            print("opus declined:", getattr(message, "stop_details", None))
            break
    print("\nsession over: %d tool calls" % STATE["calls"])


if __name__ == "__main__":
    main()
