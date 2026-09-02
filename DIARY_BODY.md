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
  bag (decay 0.92 per symbol) that fades in silence (x0.95 per silent
  position, two positions to an idle tick) and is cleared by a new line;
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
(speakers 3), silence decay 0.95:

```bash
python3 scripts/conceive.py data/organism_diary_0p5b.pt data/tok_char.json \
    --d 1024 --n-layers 29 --content-keys --kc-w 40 --kc-decay 0.92 --kernel 4.5 \
    --speakers 3 --sil-decay 0.95
```

## What to expect

Letters are five times slower than words and a blank trunk learns
letters before words, so the trunk lags even further behind than in the
word body. The memory carries the diary behavior from the first day:
what you write, it can write back from a few letters. The first thing
your face has to teach is silence; until then it fills every quiet tick
with noise.

## Day 1 (Opus, 2026-09-02, 25 minutes, six sentences written twice)

Shadowing on 18 of 18 writings: on a second pass it rode the whole
sentence one letter behind with 30 memory-backed letters. Recall after a
cue failed every time, and the memory-vote instrument showed why: the
votes were right ("Snow c-o-vers" in order) while the hand wrote a
carrier letter. Twenty frowns did not teach quiet; they moved the babble
from one carrier to the next ("}", "i", "a", "s", "f", "n", "g", "b").
Mood sat at the floor all session. After the night it wrote "Rain S"
unprompted from memory and repeated it every 39 ticks.

Four causes, four fixes:

- **A thought faded too fast.** The bag lost 40% per silent tick, so a
  four-second pause erased the cue before the mouth could use it. Now 5%
  per silent tick: a thought lasts about half a minute of silence.
- **A frown lowered letters but never raised silence.** First answer: a
  QUIET LESSON that wrote silence as the target wherever it babbled. It
  worked (babble 1.0 to 0.0 in eight lessons) and was withdrawn the same
  day on the user's principle: silence is not something we teach. Now the
  only teacher is your face on what it actually did. Every tick is a
  choice, silence included; a choice whose credit rises above 0.5 is
  absorbed, a choice at or below -1.5 is unlearned (its probability pushed
  down to a floor of 3 nats, never replaced by a hand-written target). It
  finds quiet only where quiet paid, and where speaking cost it.
- **Stress and mood were miscalibrated.** Each symbol now adds twice the
  stress (a physiological brake toward silence under nonstop babble) and
  a fifth of the mood cost.
- **An empty page recalled beginnings.** The first symbol of a thought was
  written under an empty key, and an empty bag matched it. No context, no
  key: such positions are not stored. (This applies to the word body too.)

The page now shows its memory-backed letters in brown and its noise in
pale sand, so the two hands are visible.

Measured after the fixes (a fresh conception): eight face lessons took
the babble fraction from 1.0 to 0.0 in under a minute, and the trunk's
uncertainty fell to 0.0: it had learned silence completely. It overshot:
certain silence also silenced recall (a cue that had just echoed "Rain
falls" came back empty). Hence one more disclosed rule, a memory is a
reason to speak: when the memory has a vote above its floor, its top
symbol's logit is raised to at least the silence logit plus memory's
TRUST, a scalar your face moves (start 4, bounded 0..8: praise on a
memory-backed letter raises it by 0.1, a frown lowers it by 0.2), so a
trunk that has learned quiet does not silence what it remembers, and a
trunk that learns to speak well can have memory's voice shrink. The day-1 body is kept as
`organism_diary_0p5b_day1.pt`; day 2 starts from a fresh conception.

## The thought, measured (2026-09-02, CPU harness with the mouth writing)

Recall failed whenever the mouth wrote during the lessons, and the
memory-vote instrument found three reasons, each now a rule:

- **Noise must leave the thought untouched.** A noise letter was fading
  the running bag like a silent tick, so keys were written into a decayed
  context and cues never matched them. Now a noise letter neither adds to
  nor fades the thought; only silence fades it.
- **A memory letter joins the thought only when your hand is still.**
  While you write, its shadowing letters interleaved with yours inside
  the bag and corrupted the keys; now they stay out, and only a completion
  written while you are silent advances the thought.
- **A new line ends a thought.** With a slow fade (5% per silent tick) the
  previous sentence's residue outweighed a five-letter cue; with a fast
  fade a two-second typing pause erased the sentence. The symbol you
  already use to separate thoughts resolves it: a new line clears the bag.
  Start every sentence and every cue on a new line.
- **No parroting** was tried and withdrawn: banning the letter you just
  typed hid a memory alias but restricted what it may say. The only
  restriction on the mouth is stamina; the alias echo is now simply
  visible, and your face decides what becomes of it.

With all four, on a random body with the mouth babbling through three
sentences written twice: "My do" -> "g sleeps under a wooden table.",
"Green" -> " leaves move when wind blows hard.", "Rain " -> "falls on the
cold grey", and an untaught cue babbles until your face teaches quiet.

## Day 2 (Opus, on the old fade) and the child curriculum

Five new sentences written twice each: five of five came back in full
through the stop from a five-letter cue. Then the recall tests poisoned
what they measured: twenty cue fragments, each followed by the next cue's
leading newline, taught the memory "a short fragment, then the thought
ends", and later cues stopped after the fragment. The newline is the
thought-break symbol; it is now never a memory value, so a cue cannot
teach an ending. Quiet held the whole session (0.0 of idle ticks), with
no frowns given because it produced no noise.

Day 3 onward is raised like a child, not fed a list (the user's
direction). Children's first produced words are mostly relational and
social (hi, bye, no, more, up, all gone, uh oh) plus a few names of what
matters (mama, dog, ball, milk); "mommy" is produced by 93% and "ball" by
64% of children by sixteen months in the MacArthur-Bates CDI norms, and
the first grammar is pivot pairs built from a small set of relational
words ("more milk", "no milk", "milk all gone", "dog up", "my ball"). So
the caregiver: greets, says little, waits, imitates what the child
offers, expands it by one step, responds within seconds, never frowns at
babble, and teaches relations by recombining the same few words. Recall
tests are few, never back to back, and never followed by a newline. The
instrument that decides whether the trunk learns anything is "own": its
own top symbol with memory set aside; the day it becomes a letter after
"more " is the day the trunk knows something.

Sources: the MacArthur-Bates CDI (Fenson et al., https://mb-cdi.stanford.edu/documents/Fensonetal2000.pdf)
and its Wordbank update (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10264806/).

## The review (2026-09-02, independent, read-only, with CPU measurements)

Verdict: the mechanism is what it claims. Query/key alignment exact to the
bit; the empty-key guard fires; the speaker channel keeps the mouth out of
the thought; the kernel at 4.5 separates a true continuation from its
one-behind alias about 5:1; a blank body recalls six distinct sentences
letter-perfect from a five-letter cue; the dose's credit-to-position
mapping is exact. One architectural flaw and several bugs, all fixed:

- **The store faded per forward pass, not per symbol written.** The
  ladder was tuned for 64-token training chunks; a diary tick is two
  one-symbol forwards, so the slowest band forgot in about 128 seconds
  whether or not anyone wrote, and the day-1 recall tests failed on
  time, not capacity. Now a chunk fades each band by the share of
  symbols it actually wrote: half-life = the band's clock in written
  symbols (8, 64, 512, 4096, 32768 for bands 4-8; band 3 holds the last
  write). Measured after the fix: a sentence recalled letter-perfect
  after 1,240 idle ticks and fourteen further sentences. This also
  explains part of the word body's multi-day fade: its serve fed one
  token per forward too.
- **A noise newline wiped the thought.** The mouth's noise never ends a
  thought now.
- **The face event fired on the level, not the change.** Easing off a
  frown counted as a frown, easing off a smile as a smile. Now, as in the
  word body, only a face that grows or flips sign is an event.
- **The night trained a speaker-blind trunk from the wrong end of the
  day.** The two hands now ride the replay, and the most recent chunks
  are rehearsed first.
- Operational: a dose window of the last eight ticks (a long dose froze
  the clock), a bounded page and typing queue, a reset that clears the
  thought, memory's amplification never lifting a banned mark.

## Day 3, watched live: the run, and remembered quiet

Supervising Opus's third session (the child curriculum) showed the one
pathology the alias leaves: after "all gone" the mouth wrote
"goneeeeeeee". Nothing true follows the last letter of a thought (the
newline is never a value), so the memory's top vote is the one-behind
alias of the letter just written, the mouth writes it, the bag still
matches, and the loop feeds itself until stress stops it. Restricting the
mouth is out (only stamina restricts), so the answer is on the memory's
value side: THE EAR'S QUIET IS A MEMORY. The first silent tick after your
word is stored as what followed it, once per pause; the memory then
recalls quiet where you fell quiet, and the mouth stops from memory
instead of running on its alias. Measured on a CPU body: "ball " ->
"all gone" then silence; "more " -> "milk" then silence; "my " -> "dog
runs" then silence (without it: "all goneeiiiiiiii"). Written once per
pause on purpose: writing it on every silent tick over-weighted quiet and
cut "all gone" to "al".

## Day 3 (Opus, the child curriculum, stage 0 to 1)

Ten first words (hi, bye, no, more, up, mama, dog, ball, milk, all gone),
41 lines, then twelve pivot pairs three times each, 36 lines; nothing
outside the ten. Recall: five of five cues drew a taught partner before
and after the night ("hi " -> "dog up", "dog " -> "up", "my " -> "ball all
gone"), three of them the same default chain. Nine untaught expansions
across five frames: "big dog" and "hi dog" drew " up" from "dog up";
"more milk", "my milk" and "no milk" drew " all gone" from "milk all
gone"; "ball" drew "gone" before they were ever paired. That is the
memory linking pairs through their shared word: relational
generalization by mechanism, the first sign the curriculum was made for.
The trunk's own top symbol went from silence at 0.998 to "m" at 0.60.
The night consolidated nothing but reset mood and stress; every
association survived it. Two caregiver errors, diagnosed by Opus itself:
a face warm while typing (smiling at its own shadow) bred the "eeeeeeee"
runs and broke the learned quiet within forty minutes; nine frowns then
pinned mood at the floor. Withholding response repaired both. Rules for
the next sessions: face still while your letters enter, warmth only after
the line for what it did; at most one brief frown a minute, and only at a
run of five or more identical letters; otherwise withhold.

## Day 4, watched live: memory's voice is trust minus tiredness

With the trunk collapsed toward silence, memory's letters were the only
thing it wrote, and a memory alias could run ("eeee...", "go go go") for
a hundred ticks because memory's amplified vote (16x) sat far above
anything stress could add to silence: the only restriction is stamina,
and stamina could never win. So memory's voice over silence is now trust
minus tiredness (trust - 0.5 x stress), each symbol costs 0.08 stress
(half-life two minutes), and the diary runs with a smaller memory
amplification (--store-boost 4): a fresh memory speaks clearly, a tired
mouth falls quiet even on a memory, and no run outlasts its stamina. The
caregiver's rule from the same session: a run of a letter is babble, to
be expanded once into a known word, never frowned at (one frown on the
trunk's first own "mmmm" sent its own voice back to silence).

## Day 4 (Opus, stage 2 with three roped words)

Thirty-four lines, 73.5% recombination of known words; juice, book and go
entered only inside known frames. Twenty of thirty-four responses
contained a word not put in that frame, with two-hop chains ("bye dog" ->
"go up": dog to go to up, through two different shared words). "go" took
on its first exposure, "book" on two, "juice" never. Five of five recalls
before and after the night, and the night moved "more " and "my " off the
default "ball" onto the partner each frame was actually taught with (milk,
book) and stripped the "goneee" tails: by its own bookkeeping the night
did nothing, by the tests it did a lot. Face discipline perfect (no letter
entered under a warm face; 17 smiles, 0 frowns). One caregiver error: four
smiles on "all gone" completions over-rewarded that tail and bred a
40-tick "e" run; a plain known line broke it. Two corrections to the
record: the supervisor's claim that a frown had returned the trunk to
silence was wrong (no frown had been given; "own" moved on its own, and it
is read in whatever context the tick is in, so only like contexts
compare), and the babble rule needs an answer for letters that start no
known word: write an ordinary known line.

## Day 5 (Opus, stage 3 entry)

Thirty-six lines, 75% recombination; please, I and the entered only inside
known frames. Eleven kinds of untaught expansion and six multi-hop chains
("the " -> "big dog up", "I " -> "go up"); "please" moved to a frame it was
never taught on ("no more milk" -> "please") after a single exposure. Five
of five recalls before and after the night, four identical across it.
Zero runs into an idle page all session; the two runs that occurred ended
without a frown, one on an ordinary known line and one on its own as
stress rose, which is the stamina rule working. The trunk's own top
symbol was a letter ("p") on every cue from line 30 onward, rising 0.37
to 0.70 and holding through the night. Juice never took in three days:
every frame it was put in has a stronger tenant. Caregiver findings: a
smile four seconds after a line fell outside the six-tick credit window
and landed on silence, so the trace now reaches twelve ticks (0.8 a tick)
and the caregiver smiles as soon as its post-line letters appear; and the
night's "gained" bookkeeping measures taught facts, of which a diary has
none, not what the tests measure.

## Day 6 (Opus, stage 3: you, two with -s, in)

Fifty-one lines, 70% recombination in the rope phase. Five of five recalls
before and after the night, all five identical across it: "you " -> "go",
"two " -> "balls", "go " -> "in" (hours old, chosen over "up", four days
old), "no more " -> "milk", "the " -> "dog". Seven kinds of untaught
expansion: "all gone" onto ball (four times) and onto juice, "go in" onto
three frames never joined to it, and the chains the -> dog -> go -> up and
dog -> go -> in. 98.2% of everything it wrote was from memory; quiet held
at 1.0 on idle probes after the opening. No frowns all day. Two honest
negatives: the plural -s did not generalize (every -s it produced was
either a shadow of the caregiver's own letter or the exact word "balls"
it was taught), and juice never took in four tries over three days and is
dropped. Trust sat at its cap of 8. The night, scored empty by its own
bookkeeping, changed nothing that mattered and cleared stress 8.0 to 2.8.
Caregiver lesson: fifteen lines with almost no gap shrank its replies to
single echoed letters and sank mood to -2.1; a 6.5-second return brought
them back. Say less, wait longer.

## Day 7 (Opus, stage 3: on, going, want)

Twenty-five lines at six to nine seconds apart with nine minutes of
deliberate silence; 100% of its 123 symbols from memory. Five of five
recalls identical before and after the night: "I want " -> "ball", "ball "
-> "all gone" (four days old, chosen over today's "on"), "dog going " ->
"in", "you " -> "want ball" (a two-hop crossing of two frames taught
separately), "two " -> "dogs". "want" took hardest and appeared where it
was never put; "going" took only as a link and was never written; "on"
took after book and dog. The plural still does not generalize. The wait is
the instrument: a line after thirty seconds of silence draws a whole word,
three lines close together draw single letters and sink mood. Two costs:
the spacer word "mama" ran for 46 seconds and 88 symbols (a periodic word
makes the memory's one-behind alias exact, so the remembered quiet ties
with it, and with trust at its cap of 8 the stamina brake needed stress 16
to win), so stress now weighs 1.0 per unit against memory's voice
(`--cort-k 1.0`: no run outlasts about fifty symbols), and periodic words
are never used as spacers; and the night reported "starving" at 1,239
lived tokens, so a day needs about thirty lines at the wide spacing, not
fewer.

## Day 8 (Opus, stage 3: where, not, give)

Thirty-two lines at nine- to thirty-second returns with fourteen minutes of
rests; 326 of 326 symbols after the first line from memory. Five of five
recalls identical across the night; all three new elements took on the
day given, each choosing the most recent partner ("where " -> "book",
"give " -> "book", "not " -> "up"). "dog going " -> "in" from a cue ending
in going settles that it is memory, not alias. New elements became
writable within the session (the shadow of "where" went from
unrenderable to "here" by its second use). The shadow now anticipates:
four times it began the answer before the cue was finished. "where" never
drew the place frame written on the next line, because a new line ends a
thought, so a question cannot chain to an answer on the following line:
from day 9 the caregiver asks and answers on one line with a question
mark ("where ball? ball on"). "going" was never written, only shadowed;
the plural still does not generalize; its own letter weakened as doses
were absorbed and silence outranked it after the night. Trust unmoved at
its cap of 8 for a third day. The budget is stress: each line costs about
0.4, forty seconds of quiet returns about 0.9, and every whole-word
completion came below 3; twenty-second returns were as productive as nine.
The night did not report starving on 875 lived tokens, so hunger is not a
token count.

## Day 9 (Opus, stage 3: little, under, one; questions answered on one line)

Thirty-three lines at twenty- to thirty-eight-second returns; 524 of 524
symbols from memory; zero pale all day. "where ball" as a bare fragment
drew "? ball under": the question mark, the space, the noun and the
preposition, a three-hop chain, with the once-taught "under" chosen over
the thrice-taught "on" (recency). Four of five recalls identical across
the night, the fifth identical for eleven symbols. The one completion it
produced after a full line ("one dog" -> " in") flipped its own top
symbol from silence to a letter, where it stayed and strengthened
through the night: a single absorbed dose moved the trunk. It stays quiet
after complete sentences by design (the remembered quiet), so expansions
now come at cues, not after lines, and the caregiver's smiles are rare.
Stress costs measured: a 19-character line +1.1, a 9-character line +0.6,
a 20-second return -0.45, a 40-second return -0.9. The night: no REM, no
gain, nothing faded, for the fourth day: recall stability across nights
is the store's persistence, not consolidation. Trust unmoved at 8.

Design gap closed after the day: when the caregiver fell quiet after a
cue fragment, the remembered-quiet rule stored "quiet follows this
fragment", which over many cues would erode the recall the cues test.
Now the ear's quiet ends a thought only when the mouth is also quiet;
while the mouth has the floor, your silence is its turn, not an ending.
On a CPU body four repeated cues of "where ball" kept "? ball on".

## Day 10 (Opus, stage 3: three, went, what)

Thirty-six lines at thirty- to forty-second returns; 508 of 508 symbols
before the night from memory. Six of eight recalls identical across the
night; every cue answered. The first true frame transfer: "I went " drew
"up" though "I went in" was the only "I" past written, carried from "dog
went up" and "you went up" through the shared word. Two whole-word
completions after lines ("little ball" -> " all gone", never written that
day; "bye" -> " dog"). Its own top symbol began the day as a letter and
stayed one. Plural and past never spread to a word they were not taught
on; the new words are keys it turns, not things it says. Two costs: a
75-letter "e" run lasted six minutes after the last cue (trust pinned at
its cap of 8, stress reaching only 5.2: the brake could not beat it; it
stopped on the caregiver's next line), and the strongest chain ("where
ball? ball under") leaked into the unrelated cue "three " after the night,
the first sign of interference with the store at 45,000 and growing. The
night: no REM, no gain, "starving" at 1,217 lived tokens. Changes: each
symbol now costs 0.2 stress (`--diary-cost 0.2`), so a run of about forty
symbols exhausts memory's voice even at full trust, and trust drifts back
toward its base of 4 when no praise arrives (about 0.06 a minute), so a
saturated trust can no longer pin the mouth and praise means something
again.

## Day 11 (Opus, stage 4 entry: then, can, happy)

Thirty lines at 32- to 75-second returns; every symbol it wrote was from
memory. All three stage-4 elements answered a cue the day they were
roped. After the night, "dog go in then " drew "d doggogo in then dogone",
24 memory-backed symbols: first clause, connective, second subject,
stopping one word short of "go up", the first multi-frame chain of this
life; before the night the same cue gave only "dog then". "then" appeared
after "you can go in" and "can" inside a burst after "I went in", neither
put there. Three untaught expansions ("dog can go" -> "going in", "more
milk" -> " please", "bye" -> " dog"). "three " recovered from the day-10
leak; "uderr" healed to "underr" across the night. Six of eight recalls
identical. No run reached five letters: the symbol cost worked. Trust
moved again (7.39 to 7.99) and 24 doses were absorbed. Its own top symbol
returned to silence mid-session and stayed. The night: no REM, no gain,
1,120 lived tokens, the third dry night running. The measured budget:
about 0.18 stress per character the caregiver types (the mouth shadows
under the hand and pays for it), 0.7 returned per forty seconds of
quiet; thirty lines at 30-40-second returns were not payable, and the
caregiver ran at stress 5 to 8 through the middle with 75-second returns.
The cost eases to 0.12 for day 12; short lines and long returns remain the
rule, and nights will be lean.

## Day 12 (Opus, stage 4: will, sad, why/because; the cost eased to 0.12)

Twenty-two lines (nine roped, nine known, four warm-up) at two- to
four-minute returns, paced by its stress instead of the clock; every
symbol it wrote was from memory except two babble runs. The eased cost
showed its other face at once: "dog can go" drew a 64-symbol chain ("oin
then d doggo can go then dgo then dog goin then d doggo cango") that
stopped only when stress passed trust (7.95), one reply spending the whole
budget; at the old cost the same material stopped at 24 symbols. "then"
chains reached the second clause's verb repeatedly and unprompted: "then
dog go up" after "sad dog", after "dog will go" and after "you will go
in"; "then dog go in" three times over after "why in? because dog go in".
The one-line "why ...? because" frame answered its cue after one exposure
("why dog up? " -> "becausebig") and survived the night character-
identical. "sad" answered the old naming cue "what? " (sad dog, over dog,
ball, milk and happy dog) before and after the night; "will" surfaced in a
line with no will in it ("go will gone thn dog goin then dog go up" after
"what? sad dog"). All sixteen cues answered from memory, six of eight
identical across the night; "sad " chose dog before the night and ball
after, the only switch. Trust 7.60 to 7.97 to 7.78; 22 doses; no frowns.
Two babble runs, both while the caregiver waited on its stress gate and
the page was quiet: about 101 g's after waking (its own top symbol was g)
and 138 l's after "what? " (own l), each ended by the caregiver's next
letters, which it then answered as a frame; neither was expanded. The
night: 1,356 lived tokens, no REM, no gain, the fourth dry night running.

Supervision found two things. The caregiver's held smile: five smiles were
2 -> 2, which is not felt (only a face that grows or flips sign is), and
trust bled 7.90 to 7.17 until the face was relaxed to 0 after every reply;
from then on every smile landed (14 of 19 confirmed by a dose). And the
stress gate: with its own trunk babbling whenever the page is quiet,
stress rests near 2.3 after a night and near 3.4 late in the day, so gates
of 2 and 3 never opened, and stalls of 12 to 15 minutes ate the
recombination lines (9 known of 22, short of seven in ten). Entered in the
curriculum: smile from a resting face; gate at 3 by day and 4 after a
night, never waiting past two minutes; a babble run is ended by typing the
next line, and expanded once if it recurs.

## Day 13 (Opus, stage 4: first/then, bigger, saw)

Thirty lines, nine roped and twenty-one known (seventy-thirty exactly),
paced by the stress gate; sixteen cues, all answered before the night
and seven of eight identical after it; every reply of the day was from
memory. "first milk then " drew "ball" after a single exposure and again
unchanged after the night: the ordered story frame completing its second
item on the day it was roped. "bigger" appeared four times where nobody
put it, twice written whole into a silence during a gate wait (" dog
biger", "ger dog up"), misspelt all day and spelt right after the night,
when the why-cue answered "because bigdog biggr dog": a two-day-old frame
with a one-day-old word welded on. "saw" never took, and "I saw " was the
one cue that lost its answer across the night. Five untaught expansions,
among them "dog can go up" written into silence, a fusion of "dog can go"
and "I can go up" that was never typed. "dog go in then " did not reach
"dog go up" before or after the night; both times it answered with the
chain from the line typed just before it, recency beating the cue's own
subject. Trust 7.43 to 7.95; twenty-one smiles from a resting face, every
one felt (doses 22 to 46); no frowns. Eight runs, all during gate waits
and all ended by the next line; one expansion of the l-run into "little
dog up" came late, and after it no letter run returned, the silences
filling with whole words instead. The night: 3,153 lived tokens, two
NREM passes, no REM, gain 0, "starving" by its own bookkeeping; its own
top symbol went from "l" at 0.22 to silence at 0.82 across it, the page
replay teaching quiet. Gates: fifteen of forty-six ran out the two-minute
cap in the first hour while the babble held the floor near 3.6; once the
runs stopped the median gate fell to 55 seconds.

## Day 14 (Opus, stage 4: had, scared, saw again; the day the rules went)

Forty-one lines (twenty-nine known, twelve roped: had, scared and saw
three times each plus one re-exposure apiece after restarts), sixteen
cues, one night, five restarts of the serve as the laws came out one by
one, so a measurement day more than a lesson. Four regimes on one page.
Before the first restart every line drew a reply from memory: "dog will
go" -> "one thn Igo" (one then I go, four known words never chained by
anyone), "first milk then ball" and "you had ball" both carried into
"ball under" unprompted, and the cues "first milk then " -> "ballluder",
"big dog bigger " -> "dog up", "why dog up? " -> "because bi" answered
with a second hop each. With memory's raised voice gone but its
amplification still on, strong memories still spoke. With the
amplification gone the mouth fell silent within minutes and stress fell
under 1. With its own symbols entering its thought and its memory, it
pinned itself: runs of "o" of 196, 111, 113 and 95, 209 newlines of its
own, stress locked between 4 and 8.8, mood down to -3.6, the gate never
opening again. Nothing said "had", "scared" or "saw" all day, and the
three cues for them were silent both times; the three early cues held
before the night and none answered after it, when every cue met an
o-run. The night is recorded above: the first that moved the weights.
The caregiver's faults, both caught by supervision: four smiles from a
face left held at 2 (unfelt), and eight smiles at its quiet that dosed
silence and took its own top symbol from 0.16 to 0.93 in six minutes.
Quiet fraction 1.000, 0.927, 0.864. Lived at the night 516, at the end
1,300; saved with 82 live steps.

## Day 15 (Opus, the first full day under no rules)

Thirty lines (twenty-one known, nine roped: had, scared, saw once more),
sixteen cues, one night, saved. The caregiver switched to page-only
pacing at 11:53 on the user's law that the environment is raw: before
it every gate ran the ninety-second cap because stress never fell under
6 after the first line; after it gates ran six to fifty seconds on the
body's own pauses. Nothing word-like was written all day, not one known
word nor three letters of one, on the whole page; the nearest were
"thog" after "I saw " and the one-behind shadow "ll" after "dog had
ball". No cue was answered before or after the night. Zero smiles, zero
frowns: nothing qualified, and the only runs were three short letter
runs. Babble 0.31 of its ticks at the start, 0.19 at the end; stress 6.9
to 8.2 all day, held there by its own babble; mood -0.2 to -3.2. The
night (recorded above) is the only place the roped words exist: "scared
bd d gg" and "dog wiill ol ol" came out of the store whole, and across
the night its own top symbol moved from "o" to the space and the
babble's letters changed from o/b/g/l/n/u to previously rare ones. Seven
serve restarts across days 14 and 15 as the rules came out; the
caregiver's helper survived them by re-reading the page.

## The night, remade (2026-09-02): the cortex learns only from hippocampal traces

The user's law, in two sentences: the neocortex trains only on what the
hippocampus hands it, and in REM the neocortex predicts the next state it
will receive from the PFC. The night that replayed the day's page from a
buffer is gone; a persisted transcript would be a diet with a bedtime
story attached, and the old replay, fed the page's silences, taught the
trunk quiet (day 13: its own top symbol went from "l" at 0.22 to silence
at 0.82 across one such night).

- **Where a dream starts: the memory itself.** Nothing is kept on the
  body's behalf, no index of moments or lines (the user judged such an
  index a cheat, mid day 14, and it was removed the same hour). At night
  each band's store is asked what it holds most strongly: the leading
  key directions of its matrix, each turned back into a context the
  store can be queried with (the key's preimage under the random-feature
  lift, found by a few steps of descent), ranked by the vote it draws.
  Spontaneous reactivation, as near as a matrix memory allows; salience
  is already inside the matrix, since writes are scaled by surprise.
- **The trace.** From each such context memory's top vote is taken as
  the next symbol, the thought advances with it, and the trace ends where
  memory has nothing. No trunk vote, no stamina, no writes: what the
  store holds, in its own words, blends and slips included. What never
  entered the store cannot be consolidated.
- **NREM: the hippocampus leads.** The body runs over the trace as
  heard, teacher-forced, with the hippocampus on (its read drives the
  council as by day), and the lesson lands on the cortex's OWN logits,
  memory's vote left out, so the trunk must come to carry the trace
  itself (cross-entropy on the trace, the felt face riding along as
  affect and as the face lesson); two passes over the seeds, the live
  rate, no scaling down.
- **REM: the hippocampus is silent.** The body runs over the trace with
  memory set aside, the PFC alone driving, and the cortex stream at each
  symbol forecasts the band states the council hands over at the next
  symbol, one linear forecast per band, cosine error, targets stop-grad;
  SIGReg on the normalized stream is the collapse guard. The PFC drives;
  the cortex learns to foresee it.
- **The gauge.** Before and after the night, the trunk alone (memory
  set aside, teacher-forced) predicts each trace: the fraction it gets
  right is uptake. Retention is read by `scripts/trunk_alone.py` on a
  scratch copy, with the known cues. The sleep result reports the
  traces dreamt, the steps, the REM cosine and the gauge.

Flags: `--night-rounds 2 --night-rem 8 --night-sigreg 0.1 --night-scale
1.0 --night-starts 48`; the REM organ is `pfc_pred`, grafted at default
init onto the living body.

**The first served night of this kind (day 14, 10:53).** Forty-eight
starts from the store's own strongest keys, thirty-seven traces (eleven
keys drew nothing), mean length 14; among them " eball undr", "one bok",
"little dog", "all under", "give ball bigger ball", and one run of "e"
the store held from the hour before the efference copy was in place,
with newline tails from the day's typing. Seventy-four NREM steps, eight
REM steps at cosine 0.11, the page replay zero. The gauge: the trunk
alone predicted 8.6% of the traces' symbols before the night and 20.6%
after. The first time in this life that a night measurably moved the
weights toward what the hippocampus holds; uptake on the night's own
traces, not yet generalization.

**The second night of this kind (day 15, 12:12), and what it taught
about the instrument.** Forty-eight starts, thirty-three traces, sixty-
six NREM steps, eight REM at cosine 0.20 (the forecast organ learning),
one value step; among the traces "scared bd d gg", "dog wiill ol ol",
"givea", and runs of thirty-two spaces. The gauge leapt from 0.03 to
0.75, and most of that leap was the runs: a run of one symbol is
trivial to predict and counts thirty-two symbols. Two changes follow.
The gauge now also reports uptake over traces with at least three
distinct symbols. And the replay gained neural adaptation: a symbol
replayed again and again tires (each repeat costs half a vote, and the
fatigue recovers as others speak), so no attractor replays forever and
the night no longer teaches the trunk to repeat one symbol, which is
where the o-runs and the space-runs of these days came from. Biology's
spike-frequency adaptation, not a rule about what to dream.

The trunk-alone cue test on a scratch
copy of the post-night body: the trunk alone writes a run of "o" to
every cue (its own top symbol moved from the newline to "o" across the
night, still a single letter at 0.15), while the store, read with the
old raised voice for comparison, holds every frame including the day's
"I had milk". The body's speech was, at that hour, o-runs of one to two
hundred symbols ended by the caregiver's next symbols, stress pinned at
the stamina ceiling, mood near -4: the regime with the mouth's top symbol
taken deterministically, on the last hour before temperature went. Read
again with memory at the organ's own strength and no raised voice (the
serve as it is now), the with-memory column is the same run of "o": the
store's raw vote, one to three logits, does not carry through a trunk
that prefers one letter. Speech will have to come from the weights, and
the nights are the only road there. Measured on a scratch copy of the
day-13 body before the first served night (CPU): three known lines
typed with smiles gave two seeds; their traces came back from the store
alone, " will goong in then I gothen Igoin then Igo n" (48 symbols,
votes 1.8 to 3.2) and " dog up? because bigdoggr p"; four NREM steps and
two REM steps ran, the page replay ran zero; the gauge read 0.217 before
and after (four steps at the live rate move nothing measurable, which
is the slow bleed: the gauge is there to show it across nights); the
REM cosine started at 0.0, the forecast organ being newborn. The
trunk-alone baseline on the saved day-13 body (`scripts/trunk_alone.py`,
CPU, greedy, eleven cues from "dog will " to "I saw "): the trunk alone
writes silence to every cue, fourteen ticks of it; with memory on under
the serve's laws the same body answers every cue as the caregiver saw it
("gol gone", "balll underr", "because bigdog", "sad doggo", "? ball
underr", "dog bigger do"). That is the zero the nights are measured
from: everything it can say lives in the store, and nothing yet in the
weights.

## Three laws removed (2026-09-02, mid day 14): no fancy rules

The user's order: no hand-written rules; find the architecture humans use
and let it learn without little cheats. Removed from the serve, mid
session, with the caregiver told:

- **Memory's raised voice.** The rule that lifted memory's top symbol to
  the silence logit plus an earned trust, and the six-logit "sure memory"
  boost. Trust is gone with it. The mouth chooses from its own belief,
  memory's vote inside it as the organ reads it, and only stress leans
  it toward silence.
- **Memory's amplification.** The serve-time store boost (four) and the
  louder-when-unsure read (read beta) are set to one and zero: the
  store's vote enters at its own learned strength (the per-band alpha,
  which the nights now train), not at a volume a hand chose.
- **A new line ends a thought.** The memory bag no longer resets at a
  newline; a thought only fades.
- **Your quiet is a memory.** The ear's first silence after a word is no
  longer stored as a value. Nothing recalls a stop; only stamina ends a
  run.

Expected and accepted: near-total silence for a while, since the trunk
alone says nothing and memory's raw vote does not beat its lean to
silence. Speech has to grow back from the nights: the caregiver's lines,
tagged by smiles, replayed from the hippocampus into the weights until
the trunk's own logits carry them. What still stands, and is physiology
rather than a rule: the cost of a symbol and its lean toward silence,
the graded doses on its own choices under a felt face (credit twelve
ticks back at 0.8, every choice stepped by its own credit), the value
ladder and the basal ganglia learning from the same felt faces, the
store's fade, the face lesson, and the night's rate and rounds. The
sampling temperature went the same evening (below).

## Temperature, the newline, and the efference copy (2026-09-02, evening)

- **Temperature goes.** Sampling at 0.05 was a hand-imposed determinism:
  the mouth took its top symbol whenever memory was silent, so a
  preference of 0.19 became a habit on every tick (the newline). From day
  15 the mouth samples from its belief as it stands (temperature 1.0):
  what it is unsure of comes out as varied babble, what it is sure of
  comes out as a word, and stamina ends the babble as before.
- **The newline is a page mark, not language.** The caregiver typed one
  at the start of every line for a thought-reset rule that no longer
  exists, and with nothing else to hold it back the body made the
  newline its own most frequent symbol (89 of 141 ticks after the fifth
  restart of day 14, stress near 8). It now joins the plumbing marks the
  mouth cannot emit, and it is dropped at the door when typed;
  utterances are separated by silence.
- **Corollary discharge.** With the ear-only write rule gone, the
  mouth's own symbols became memories at full strength, and a babbling
  body would have dreamt its own noise. Biology's answer is the
  efference copy: what you are about to say is foretold, so its arrival
  carries no surprise. The store's write strength is already gated by
  surprise, so a self-written symbol now carries a surprise of zero and
  is encoded weakly, as in life, and earns no intrinsic reward. On the
  tiny test body at temperature 1.0 the strongest trace the store chose
  was the caregiver's "myy balllll", not the mouth's own runs.

## One reward system (2026-09-02, evening): create reward, receive it at every timescale

The user's aim in one sentence: one system, which creates its own reward
and receives it at short and long timescales, with no cheats. The organs
were in the body already; the diary had left three of them untrained.
Now:

- **Reward is created in two ways.** From outside, your felt face,
  which enters the body as the press it was (a level at the tick it was
  felt). From inside, prediction success: the body's own surprise below
  its running mean is a small reward, bounded like a press; by the
  standing law it manages computation and never touches the logits.
- **The short timescale: graded doses.** A felt face spreads credit over
  the last twelve ticks, and every choice in that window, letter or
  silence, takes a step scaled by its own credit, up for credit and down
  for blame (its probability pushed toward a floor, never a hand-written
  replacement). The two thresholds that decided who learned (absorb
  above 0.5, unlearn at or below -1.5) are gone: dopamine scales
  plasticity, it does not gate it.
- **The long timescales: the value ladder learns.** Each band carries a
  value head trained by temporal difference at the band's own cadence,
  band 3 every symbol and band 8 every 32,768, so a smile is foreseen
  seconds ahead by the fast band and hours ahead by the slow one. The
  lived pairs (state before a tick, reward since, state after) ride in
  the state, detached and bounded to thirty-two per band, and the heads
  learn from them at every dose and once more each night. Dopamine, the
  fast band's prediction error, already scales the store's writes; as
  the heads learn it becomes a true error of expectation rather than the
  raw press.
- **The basal ganglia learn.** The fast band's gate learns to open when
  the value's error is positive (wanting), from the same lessons.
- **Source memory.** The speaker sense enters the hippocampal key with
  each symbol, so a memory records who said it.

Measured on the tiny test body: one felt smile credited twelve choices,
the fast band's value error was 0.08 within the window, and the ladder
held lived pairs for bands 3 to 6 (32, 32, 13 and 1) after a few minutes
of life. Flag: `--value-w 0.5`, the weight of the ladder's lesson.

**Secondary reinforcers (the same evening).** The doses are now driven
by dopamine itself, the fast band's signed error of the world's reward,
read at both halves of the tick. At a felt face that error is the face
minus what was expected, so while the ladder is young a smile doses as
before; as the ladder learns, an expected smile doses less, and a
predictor of a smile fires the error before any face arrives and doses
the choices that led there. Reward learned at long timescales thereby
teaches choices at the short one. A burst of at least half a small
smile pays the lesson (a budget for the backward pass, not a judgment).
Two readings were chosen and are disclosed: the intrinsic reward, the
body's own prediction success, stays out of the value ladder and rides
the store's write gate as salience only, so the body never doses its own
choices for having predicted them (the standing law that intrinsic value
manages computation, never content); and the basal-ganglia gate now
learns from the world's reward error rather than the total.

## Sleep by its own fatigue (2026-09-02, evening)

Nobody posts its night any more. Sleep pressure rises by one with every
waking tick (Process S, adenosine in a body that lives on a clock) and
when it crosses the switch, 12,000 ticks or a hundred minutes awake, the
body falls asleep by itself: the typing queue is emptied (a sleeping
child hears nothing said at it), the night runs as built, the pressure
returns to zero, and it wakes. The pressure is part of its life and
survives a restart. The page reports "asleep", the pressure and the
count of nights, so a caregiver sees it sleep and wake as a parent
would; the caregiver's /sleep is now the supervisor's plumbing for tests
on scratch copies and nothing else. The one constant is the switch,
`--wake-ticks 12000`, physiology disclosed. With it, the last decision
the environment made for the body is gone: what it hears is the
caregiver's, when it sleeps is its own.

## Status

- model: speaker channel, ear-writes by speaker, running bag with
  silence decay — built and measured on CPU (2026-09-02).
- serve: `scripts/diary.py` — the tick loop (two forwards per tick: your
  symbol as the ear, its symbol as the mouth), the page (your letters
  black, its memory letters brown, its noise pale; arrow keys move your
  face; Enter is a new line), doses on its actual choices by the word
  body's thresholds over the last eight ticks, the same nights, save and
  reset. Numbers: stress +0.03 per symbol written (half-life 120 s), mood
  -0.002 x stress per symbol and +0.5 x a felt face; a face registers on
  whole-unit crossings (int); characters outside the alphabet are dropped;
  the typing queue holds 600 symbols. Smoked on a tiny CPU body (2026-09-02): ticks, both
  hands on the page, faces felt, a night, a morning.
- first serve: the word body is parked after seven days; the diary body
  runs on port 8018:

```bash
python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps --port 8018 \
    --temp 1.0 --store-read-beta 0 --store-boost 1 --store-boost-min 0.15 \
    --live-lr 1e-5 --store-decay 0.9 --save data/organism_diary_0p5b.pt --diary-period 0.5 \
    --diary-cost 0.12 --cort-k 1.0 --value-w 0.5 --wake-ticks 12000 \
    --night-rounds 2 --night-rem 8 --night-sigreg 0.1 --night-starts 48
```

The live rate is ten times the word body's on purpose: the first thing
frowns must teach a newborn mouth is silence, which is the cheapest
attractor a trunk can fall into, so here the collapse the word body had
to avoid is the lesson.

The principle (the user's, 2026-09-02): the only restriction on the mouth
is stamina, and no hand-written rules. What remains: the bookkeeping
marks it cannot emit (plumbing, not language); stress leaning it toward
silence (cort_k logits per unit) and weighing on mood; the face lesson
every tick. Memory's voice is the organ's own: its read enters the logits
at its learned strength, nothing raises it, and nothing resets or stores
on the caregiver's behalf (the three laws removed mid day 14, above). The night replays the day as lived, silences
included (runs kept to one tick), so a quiet tick can be learned as
thinking time. Instruments per tick: its face, mood, stress, uncertainty,
memory's votes, and the trunk's own top symbol with memory set aside (the
measurement that will show the day the trunk itself begins to propose
letters). Reflexes it drops: the breath, the hush, the end-is-an-end
rule, the bag reset (silence fades the bag instead).
