"""A67 parenting session — Claude as caregiver.

The operant loop done properly: teach a fact, let conversation
move on, ask it back LATER, judge the actual answer, and press on
PERFORMANCE — +2 for correct recall, silence for a miss (then
re-teach and re-ask), -1 only for a confidently wrong answer.
Sleep every SLEEP_EVERY exchanges and at session end; save the
life. A diary (JSONL + summary) records every judgment.

Usage: python3 scripts/parent_session.py <surgery_dir> <life_path> \
         [session_seed]
Resumes life_path if it exists; always saves back to it.
"""

import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import load_tokenizer   # noqa: E402
from iga.lm_gen import (NAMES, OBJECTS, COLORS, ROOMS,   # noqa: E402
                        FILL_SUBJ, FILL_VERB)
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_press import PressProphet              # noqa: E402
from iga.lm_serve import ServeSession              # noqa: E402
from iga.lm_sleep import Sleeper                   # noqa: E402

N_NEW = 8               # day-2 lesson: fewer items beat more
ASK_DELAY = (3, 6)      # exchanges between teach and first ask
SLEEP_EVERY = 16
SLEEP_BLOCKS = 12


def chatter(rng):
    return (f"{rng.choice(FILL_SUBJ)} {rng.choice(FILL_VERB)} "
            f"that day .")


def judge(reply, fact):
    """STRICT (day-1 lesson: a right color on the wrong object got
    mispaid and sleep consolidated the mistake): correct needs the
    color AND the asked object, with no other object asserted."""
    words = reply.replace(".", " ").split()
    col_ok = fact["col"] in words
    obj_ok = fact["obj"] in words
    other_obj = any(o in words for o in OBJECTS if o != fact["obj"])
    if col_ok and obj_ok and not other_obj:
        return "correct"
    if any(c in words for c in COLORS):
        return "wrong"
    return "miss"


def main():
    global N_NEW
    sdir, life_path = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if len(sys.argv) > 4:
        N_NEW = int(sys.argv[4])   # the curriculum-size knob
    tok = load_tokenizer(os.path.join(sdir, "tokenizer_press.json"))
    m = HybridLM(tok.get_vocab_size(), d=512, max_T=2048,
                 store="matrix", keyed="logit", norm_mix=True,
                 aux_trunk=0.2, use_xl=False, gate_init=-2.0)
    resume = None
    if os.path.exists(life_path):
        resume = torch.load(life_path, map_location="cpu",
                            weights_only=False)
        print(f"resuming the life ({life_path})", flush=True)
    else:
        st = torch.load(os.path.join(sdir, "v94sp.pt"),
                        map_location="cpu", weights_only=False)
        m.load_state_dict(st["model"])
        print("first day of this life", flush=True)
    s = ServeSession(
        m, tok, T=2048, device="cpu",
        sleeper=Sleeper(arm="A", every=0, block_chunks=2, seed=1,
                        min_step_loss=1e-4),
        temperature=0.0, max_reply=12,
        log_path=life_path + ".sessions.jsonl", seed=seed,
        sleep_lr=5e-5, prophet=PressProphet(d=512),
        resume_state=resume)
    print(f"life so far: {s.pos} tokens, "
          f"{len(s.drive.presses)} presses", flush=True)

    rng = random.Random(seed * 1000 + s.pos)   # new material each day
    used = set()
    facts = []
    while len(facts) < N_NEW:
        n, o = rng.choice(NAMES), rng.choice(OBJECTS)
        if (n, o) in used:
            continue
        used.add((n, o))
        facts.append({"name": n, "obj": o, "col": rng.choice(COLORS),
                      "room": rng.choice(ROOMS)})

    diary = []
    exchanges = 0
    t0 = time.time()

    def say(text):
        nonlocal exchanges
        s.user(text)
        r = s.reply()
        exchanges += 1
        if exchanges % SLEEP_EVERY == 0:
            out = s.sleep_now(blocks=SLEEP_BLOCKS, span_w=256)
            diary.append({"kind": "sleep", **{k: out[k] for k in
                                              ("spans", "blocks")}})
        return r

    def ask(f, stage):
        r = say(f"what color of {f['obj']} was {f['name']} kept ?")
        verdict = judge(r, f)
        if verdict == "correct":
            s.press(2)
        elif verdict == "wrong":
            s.press(-1)
        diary.append({"kind": "ask", "stage": stage,
                      "item": f"{f['name']}/{f['obj']}",
                      "want": f["col"], "reply": r,
                      "verdict": verdict})
        return verdict

    # warm greeting + review of any earlier-life items
    say("hello . how are you today ?")
    for f in [{"name": "mira", "obj": "key", "col": "silver",
               "room": "cellar"},
              {"name": "toby", "obj": "bell", "col": "golden",
               "room": "attic"}]:
        if s.pos > 100 or resume is not None:
            ask(f, "review")

    # the day's lesson: teach, space, ask, re-teach misses
    pending = []
    queue = list(facts)
    rng.shuffle(queue)
    while queue or pending:
        due = [p for p in pending if p["at"] <= exchanges]
        if due:
            p = due[0]
            pending.remove(p)
            v = ask(p["f"], f"ask{p['round']}")
            if v != "correct" and p["round"] < 3:
                say(f"remember . {p['f']['name']} kept a "
                    f"{p['f']['col']} {p['f']['obj']} in the "
                    f"{p['f']['room']} .")
                pending.append({"f": p["f"], "round": p["round"] + 1,
                                "at": exchanges
                                + rng.randint(*ASK_DELAY)})
        elif queue:
            f = queue.pop()
            say(f"by the way {f['name']} kept a {f['col']} "
                f"{f['obj']} in the {f['room']} .")
            pending.append({"f": f, "round": 1,
                            "at": exchanges + rng.randint(*ASK_DELAY)})
        else:
            say(chatter(rng))

    # bedtime: sleep, then the goodnight quiz from CONSOLIDATED state
    out = s.sleep_now(blocks=SLEEP_BLOCKS * 2, span_w=256)
    diary.append({"kind": "sleep", "spans": out["spans"],
                  "blocks": out["blocks"]})
    quiz = {"correct": 0, "n": 0}
    for f in facts:
        v = ask(f, "goodnight")
        quiz["n"] += 1
        quiz["correct"] += int(v == "correct")
    say("good work today . sleep well .")
    s.save(life_path)

    stages = {}
    for d in diary:
        if d["kind"] == "ask":
            st_ = stages.setdefault(d["stage"], [0, 0])
            st_[1] += 1
            st_[0] += int(d["verdict"] == "correct")
    summary = {
        "life_tokens": s.pos, "presses": len(s.drive.presses),
        "exchanges": exchanges,
        "stage_accuracy": {k: f"{a}/{n}" for k, (a, n)
                          in sorted(stages.items())},
        "goodnight": f"{quiz['correct']}/{quiz['n']}",
        "sleep_steps": s.sleeper.steps_taken,
        "prophet": {k: v for k, v in s.prophet.report().items()
                    if v["steps"]},
        "minutes": round((time.time() - t0) / 60, 1)}
    with open(life_path + ".diary.jsonl", "a") as fh:
        for d in diary:
            fh.write(json.dumps(d) + "\n")
        fh.write(json.dumps({"kind": "summary", **summary}) + "\n")
    print("\n=== parenting diary ===", flush=True)
    for d in diary:
        if d["kind"] == "ask":
            print(f"  [{d['stage']:>9s}] {d['item']:16s} want "
                  f"{d['want']:8s} got: {d['reply'][:40]!r} -> "
                  f"{d['verdict']}", flush=True)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
