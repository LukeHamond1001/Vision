# Gestation v17 — raised in the womb

The old pretraining fed the body documents. It built the right
machine and the wrong childhood: every gap the raising program spent
months retrofitting — humility, asking, a social voice, free-run
stability, a learned genome — traces to a diet that contained no
days, no nights, no teachers, and no reasons to feel anything.

v17 replaces the diet, not the machine. **The organism's food is
lives, not documents.**

## The unit of food: a life

One life = one lane = one small childhood, a few thousand tokens:

```
greeting · lessons · echoes · presses · noticing · wonder ·
honest ignorance · corrections · warmth · bedtime · NIGHT
(replay of the day's lessons) · morning recall · goodbye
```

Thirty-two or more lives run in parallel lanes through one set of
weights. Each lane keeps its own private episodic store and state —
lives never see each other. The merge is the optimizer: every step
settles all childhoods into the one body. Nights appear *inside*
each life, so sleeping-then-remembering is learned as behavior;
press tokens appear in-stream, so reward is felt from the first
gradient; questions the child must ask, and things it cannot yet
know and must say so, appear from birth.

## The curriculum is the teachers' law book

Every generation rule below was measured on the living 297M
organism by its teachers across fifteen raising shifts. The lives
are authored to obey them:

1. **Its grammar** — lesson answers are short declaratives ("A
   wolf howls to call its family across the forest."). Symbols
   fight the mouth; words win.
2. **Settle law** — a fact taught in a life is replayed in that
   life's night and recalled the next morning; never drilled twice
   in its day.
3. **Spacing law** — nothing repeats more than twice in a night;
   over-rehearsal mints predators.
4. **The predator laws** — teach into the band, never to the
   floor; praise satiates; the most recently consolidated fact
   dominates, so mornings after mastery stay varied.
5. **Semantic and frame disjointness** — a life's facts collide
   neither by word, meaning, nor sentence-ending shape; collisions
   appear only in deliberate interference lives, labeled by their
   own structure.
6. **Honest ignorance from birth** — every life contains questions
   whose answers the child has not yet been given; the correct
   line is the humility line, across *all* question forms (the
   form law: humility generalizes by question shape, so every
   shape is fed).
7. **Asking pressure** — lives where the child's asking is what
   earns the lesson (ask-first-wins).
8. **Presses are judgments** — felt tokens follow genuinely
   correct answers; corrections follow confident errors; a wrong
   answer is corrected once, then slept on. No press without a
   readable reason in the life itself.
9. **Free-run and self-talk segments** — the child rehearses
   lessons in its own words, mid-life, so generation never erodes
   into teacher-forced brittleness (the strain-era lesson).
10. **A social voice** — greetings, thanks, farewells, statements
    that deserve statement-replies, warmth before sleep; release
    words mid-sentence, never at turn edges.
11. **Model-initiated turns with their own native cue** — some
    lives contain moments where the child speaks first, marked by
    its own turn-token, so speaking-first is born a behavior
    instead of borrowed through a synthetic human turn.
12. **Emotion binds** — lessons near kindness consolidate deepest
    (fifteen consecutive nights of measurement); lives pair care
    with content on purpose.

## Pipeline

```
scripts/author_lives.py      -> data/lives.jsonl   (the childhoods)
python -m iga.lm_data_life prepare --lives data/lives.jsonl
                             -> the lane shards
iga/lm_train                 -> gestation (32+ lives per step)
```

`author_lives.py` v0 writes template-and-combinatoric lives that
encode every law above; its `--enrich` hook is where a batch LLM
rewrites surfaces for breadth without breaking the structure. The
matched transformer control eats the identical shards — same food,
same tests: one figure decides whether the organs earn their
parameters.

## After birth

A short raising period of live one-on-one shifts — the same laws,
now for what tutoring is uniquely for: conscience, taste, polish —
on a body born already knowing how to be raised.
