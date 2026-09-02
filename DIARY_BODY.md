# The diary body

Two writers, one page, one symbol at a time. The user's idea (2026-09-02):
like the diary in the Chamber of Secrets, both hands write into the same
stream on a shared clock; there are no turns, no end-of-utterance marks,
no breath, no hush. Silence is a symbol the body must choose, and your
face on its noise and on its quiet is how it learns when to be still.

## What it is

- **Alphabet:** printable ASCII plus newline, 96 symbols, and the
  organism's specials (`data/tok_char.json`, 107 ids; `<pad>` is silence).
- **One stream, two hands.** Each tick has two positions: your symbol (or
  silence) as speaker 0, then its symbol (or silence) as speaker 1. A
  fixed small speaker vector joins each embedding, so a blank body tells
  the two hands apart from birth. Simultaneity lives at the tick; the
  trunk still reads one sequence. Two symbols are never blended into one
  position (that is noise, not a conversation).
- **The ear writes, the mouth does not,** now inside the model by the
  speaker channel: only speaker 0's symbols become memory values.
- **Three hands in the bag.** The first GPU run showed the newborn mouth
  babbling between your letters as you typed, and its junk letters became
  part of the keys your sentences were stored under. So the mouth has two
  hands: a letter it writes from memory (speaker 2) joins the thought and
  advances the echo; a letter it writes from noise (speaker 1) is in the
  stream but not in the bag.
- **Memory over letters:** the same content-keyed store with a running
  bag (decay 0.92 per symbol) that fades in silence (x0.6 per silent
  tick), so a pause separates one thought from the next without a mark;
  kernel sharpness 4.5. Measured on a random 2-layer body, letter by
  letter through the two-hand stream: "The s" -> "un is hot and bright.",
  "Cold w" -> "ater freezes into hard ice.", repeated cues intact, its own
  echoed letters never poison the page. An untaught cue babbles, because
  nothing stops it but your face.
- **Faces per tick,** as in the word body; credit by eligibility traces;
  doses become rolling instead of per-utterance.
- **Same nights, same cortisol** (a cost per emitted symbol), same
  doctrine, same instruments (its face, mood, reward, stress, uncertainty,
  memory votes).

## The body

`data/organism_diary_0p5b.pt`: conceived from nothing, d 1024 x 29
layers, untied head, content keys 40/0.92, kernel 4.5, three hands
(speakers 3), silence decay 0.6:

```bash
python3 scripts/conceive.py data/organism_diary_0p5b.pt data/tok_char.json \
    --d 1024 --n-layers 29 --content-keys --kc-w 40 --kc-decay 0.92 --kernel 4.5 \
    --speakers 3 --sil-decay 0.6
```

## What to expect

Letters are five times slower than words and a blank trunk learns
letters before words, so the trunk lags even further behind than in the
word body. The memory carries the diary behavior from the first day:
what you write, it can write back from a few letters. The first thing
your face has to teach is silence; until then it fills every quiet tick
with noise.

## Status

- model: speaker channel, ear-writes by speaker, running bag with
  silence decay — built and measured on CPU (2026-09-02).
- serve: `scripts/diary.py` — the tick loop (two forwards per tick: your
  symbol as the ear, its symbol as the mouth), the page (your letters
  black, its letters brown; arrow keys move your face; Enter is a new
  line), rolling doses by the word body's thresholds, the same nights,
  save and reset. Smoked on a tiny CPU body (2026-09-02): ticks, both
  hands on the page, faces felt, a night, a morning.
- first serve: the word body is parked after seven days; the diary body
  runs on port 8018:

```bash
python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps --port 8018 \
    --temp 0.05 --store-read-beta 1.0 --store-boost 16 --store-boost-min 0.15 \
    --live-lr 1e-5 --store-decay 0.9 --save data/organism_diary_0p5b.pt --diary-period 0.5
```

The live rate is ten times the word body's on purpose: the first thing
frowns must teach a newborn mouth is silence, which is the cheapest
attractor a trunk can fall into, so here the collapse the word body had
to avoid is the lesson.

Reflexes it keeps from the word body: press marks banned (silence is
allowed), a sure memory speaks with one voice while the trunk is unsure
(6 logits), stress favours silence (0.5 logits per unit), the face lesson
every tick. Reflexes it drops: the breath, the hush, the end-is-an-end
rule, the bag reset (silence fades the bag instead).
