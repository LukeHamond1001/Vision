# Vision — the one-token organism

A 297M-parameter language creature that lives in **one continuous
stream** and **learns while you talk to it** — on a single MacBook.
No context window tricks, no retrieval, no serve-side scripting:
every word on screen is sampled from the model, and every decision
about what to learn, when to sleep, and what to feel proud of comes
from the model's own measured signals.

```
you: how does a submarine work?
it : I do not know that yet, but you could teach me.
you: a submarine dives by filling its tanks with water and rises
     by pushing the water back out.
you: so how does a submarine work?             (next morning)
it : A submarine dives by filling its tanks with water and rises
     by pushing the water back out.
```

## The body — `iga/lm_scan.py`

| Organ | What it is |
|---|---|
| **Trunk** | 13-layer selective-scan recurrence, d=1024 — O(1) memory per token |
| **Council** | six timescale bands (clocks 1 → 32k tokens): syntax up to the shape of the day |
| **PFC** | routing and silent thought tokens |
| **Hippocampus** | episodic store: surprise-gated writes, decode-free logit-space reads, carries across days with nightly decay |
| **Plan / dreamer** | imagined futures scored on foresight; REM splices two memories at a scene cut |
| **BG · DA** | reward as experience: graded press tokens felt in-stream through real value/dopamine circuitry |
| **Goal organ** | wake-surviving pursuit slots read in identity space (lesion-verified) |

## The life — `scripts/organism.py`

**Day** — it measures its own surprise every turn and keeps what
spikes (budgeted); reward presses are felt as native tokens
(<+1><+2><−1><−2>) and expressed as graded plasticity; when its answer to something
it learned satisfies its own conscience, it presses its own button —
felt only, never self-teaching. **Night** — its own progress ledger
picks the replay (mastery graduates, stuck fades, fresh always
settles, nothing drills more than two nights running); dreams pair
the most emotionally charged memories; the conscience retrains on
the human's real presses; mastered facts enter an expanding
retention schedule (1/3/7/14/30 nights). **Always** — mood, built
from felt presses and its own measured surprise, feeds back into
how it thinks. The life autosaves every night and survives process
death.

Every threshold, budget, schedule, and reflex is a plain number in
the code (the disclosed genome), shown on screen as it acts. The
serve never authors a word and never parses the human's text for
meaning.

## Run it

```bash
git clone https://github.com/LukeHamond1001/one-token-organism.git && cd Vision
pip install -e .          # torch, tokenizers, numpy
# weights (the living body + tokenizer, ~1.2G):
#   https://huggingface.co/Luke1001/one-token-organism
# put organism_life.pt and ship_tok.json in data/, then:
python3 scripts/organism.py data/organism_life.pt data/ship_tok.json \
    --dev mps --temp 0.05 --save data/organism_life.pt
# open http://localhost:8016 — talk to it. The number beside the text
# box is your expression (−6…6), felt with what you say. Its reply
# comes one token per Enter under your current number; the face each
# word was spoken into is that word's reward. Sleep it at night.
```

First reply after launch takes a few minutes (the model compiles);
after that it answers in seconds. Works on Apple Silicon (`--dev
mps`), CUDA (`--dev cuda`), or CPU (slow). What has been measured
about it: [RESULTS.md](RESULTS.md).

## Tools

`scripts/mini_school.py` — balanced interleaved consolidation over
everything it knows (the periodic deep sleep) ·
`scripts/stutter_repair.py` — targeted unlikelihood on a degenerate
token loop, with a golds-abort gate · `scripts/knowledge_school.py`
— bulk knowledge raising · `scripts/goal_gym.py` — frozen-body
training of the goal organ · `scripts/critic_train.py` — conscience
seeding · `scripts/scan_chat.py` — terminal REPL · `pod_*.sh` /
`launch_pod.sh` — GPU gestation infrastructure for the next scale ·
`iga/lm_train.py` + `iga/lm_data_life.py` — the gestation method
itself (day/night-structured lives with rewards in-stream, the diet
the body was born on and the recipe the next body scales).

## The next body

[GESTATION.md](GESTATION.md) — the v17 pretraining method: the
organism's food is lives, not documents. `scripts/author_lives.py`
writes the childhoods (every rule measured on this organism by its
teachers); `iga/lm_data_life` turns them into lane shards;
`iga/lm_train` gestates 32+ childhoods at once through one body,
against a matched transformer control on identical food.

## Honest limits

At 297M, question→answer routing shares narrow capacity: heavy
drilling or repeated corrections can collapse it (the school pass
restores balance), abstract inference does not emerge, and social
statement replies are thin. The conscience is young — it learns the
human's taste a few presses per night. These are measured size
limits, not serve tricks; the development history lives in the git
log.
