"""v8 (A46) — the mixed long-structure corpus: code + prose + chat.

The certified core's next diet. Code is the load-bearing slice: real
repositories carry NATURAL exact long-range dependencies (an
identifier defined thousands of tokens back gets used now) at the
density the planted instruments never had — the demand-starvation
problem (A43 curriculum law) solved by the data instead of against
it. Prose (wikipedia) carries discourse structure; chat (UltraChat)
keeps the conversational register; code_search_net (function ->
docstring) trains the DIGEST channel — short clear verdicts about
long streams — as ordinary next-token on model turns.

Format is the ladder's exactly: alternating <eot_human>/<eot_model>
turns, Instruments woven between documents (the comparable measuring
stick, standard density per the curriculum law), THANKS closing each
document, uint16 tokens.bin + events.jsonl. UltraConveyor serves the
result unchanged.

Natural identifier probes (mine_ids=True, EVAL shards only): regex
definitions (def/class/assignment), later uses at gap >= min_gap
chars, answer = the identifier's first token at the use site,
distractors = other defined identifiers in the same file. Emitted as
ordinary probe events with "nat": true so autopsies can split
natural vs planted. Training shards keep mine_ids=False — the drive
economy stays on the planted channel for v8.0 (registered).
"""

import json
import os
import random
import re

from .lm_data_ultrachat import (Instruments, TokenSink, THANKS,
                                iter_convos, load_tokenizer,
                                train_tokenizer)

DEF_RE = re.compile(
    r"^[ \t]*(?:def[ \t]+(\w+)[ \t]*\(|class[ \t]+(\w+)[ \t]*[:(]"
    r"|(\w+)[ \t]*=[ \t]*(?!=))", re.M)
_PY_KEYWORDS = {"if", "else", "elif", "for", "while", "return", "self",
                "True", "False", "None", "import", "from", "def",
                "class", "in", "not", "and", "or", "with", "as", "try",
                "except", "print", "len", "range", "str", "int", "i",
                "j", "k", "x", "y", "_"}


def iter_parquet_texts(paths, columns, filter_col=None, filter_val=None,
                       min_chars=200, max_chars=200_000):
    """Stream text from parquet shards. `columns` is a preference
    list — the first present column wins (dataset schemas drift)."""
    import pyarrow.parquet as pq
    for path in paths:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        col = next((c for c in columns if c in names), None)
        if col is None:
            raise ValueError(f"{path}: none of {columns} in {names}")
        want = [col] + ([filter_col] if filter_col in names else [])
        for batch in pf.iter_batches(batch_size=256, columns=want):
            d = batch.to_pydict()
            for i, text in enumerate(d[col]):
                if filter_col and filter_col in d and \
                        d[filter_col][i] != filter_val:
                    continue
                if text and len(text) >= min_chars:
                    yield text[:max_chars]


def doc_to_turns(text, rng, max_turn_chars=1200, max_doc_chars=60000):
    """A document becomes alternating pseudo-turns split at natural
    boundaries (blank lines / paragraphs), so the two-speaker format
    the model knows carries code and prose unchanged."""
    text = text[:max_doc_chars]
    blocks, cur = [], []
    for line in text.split("\n"):
        cur.append(line)
        if (not line.strip() and sum(len(l) for l in cur) > 200) or \
                sum(len(l) for l in cur) > max_turn_chars:
            blocks.append("\n".join(cur).strip())
            cur = []
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def mine_identifier_probes(code, tok, base_pos, min_gap_chars=1000,
                           max_probes=4, rng=None):
    """Natural exact-recall probes from real code. Returns (events,
    n_tokens_checked) with positions relative to base_pos, computed
    through the tokenizer's char offsets."""
    if not code.isascii():
        # A46 review hardening: tokenizer offsets are byte-indexed at
        # the ByteLevel while regex positions are char-indexed — on
        # non-ASCII files the two drift and a probe would silently
        # point at the WRONG token (inflating the natural read with
        # predictable junk). ASCII-only mining makes bytes == chars.
        return []
    defs = {}
    for m in DEF_RE.finditer(code):
        name = m.group(1) or m.group(2) or m.group(3)
        if name and name not in _PY_KEYWORDS and len(name) >= 3 \
                and name not in defs:
            defs[name] = m.start(m.lastindex)
    if len(defs) < 2:
        return []
    enc = tok.encode(code)
    offsets = enc.offsets
    ids = enc.ids

    def char_to_tok(cpos):
        for ti, (a, b) in enumerate(offsets):
            if a <= cpos < b or (a == cpos and a == b):
                return ti
        return None

    events = []
    names = list(defs)
    for name, dpos in defs.items():
        if len(events) >= max_probes:
            break
        uses = [m.start() for m in
                re.finditer(r"\b%s\b" % re.escape(name), code)
                if m.start() >= dpos + min_gap_chars]
        if not uses:
            continue
        upos = uses[0] if rng is None else rng.choice(uses)
        ti, di = char_to_tok(upos), char_to_tok(dpos)
        if ti is None or di is None or ti <= di:
            continue
        # belt-and-braces: the token at the probe position must
        # actually spell the start of the identifier — any residual
        # misalignment drops the probe instead of corrupting the read
        piece = tok.decode([ids[ti]]).strip()
        if not piece or not name.startswith(piece):
            continue
        distract = [n for n in names if n != name][:6]
        d_ids = []
        for dn in distract:
            e = tok.encode(dn).ids
            if e and e[0] != ids[ti]:
                d_ids.append(e[0])
        if len(d_ids) < 2:
            continue
        events.append({"pos": base_pos + ti, "kind": "probe",
                       "answer": ids[ti],
                       "gap": ti - di, "nat": True,
                       "distractors": d_ids[:4]})
    return events


def prepare_mix(out_dir, sources, budget_tokens, seed=0, vocab=32768,
                tokenizer_path=None, instrument_every=1,
                mine_ids=False, spill=4_000_000, tok_sample_docs=400,
                long_pending=8, long_boost=1, short_rate=0.6):
    """sources: list of (name, weight, doc_iterator_factory) — factory
    returns a fresh iterator of either `str` documents or
    ("digest", code, summary) tuples. Deterministic weighted
    round-robin under `seed`; stops when the token budget is met or
    every source is exhausted."""
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    print("phase: start", flush=True)
    if tokenizer_path:
        import shutil
        shutil.copy(tokenizer_path, tok_path)
        tok = load_tokenizer(tok_path)
        print(f"tokenizer: reused {tokenizer_path} "
              f"({tok.get_vocab_size()} pieces)")
    else:
        print("phase: tokenizer-sample", flush=True)
        sample = []
        for name, _, factory in sources:
            it = factory()
            for _ in range(tok_sample_docs // max(len(sources), 1)):
                try:
                    doc = next(it)
                except StopIteration:
                    break
                text = (doc[1] + "\n" + doc[2]
                        if isinstance(doc, tuple) else doc)
                sample.append(text[:20_000])
        sample.append(THANKS)
        print(f"phase: tokenizer-train ({len(sample)} texts, "
              f"{sum(len(t) for t in sample):,} chars)", flush=True)
        tok = train_tokenizer(iter(sample), tok_path, vocab)
        print(f"tokenizer: {tok.get_vocab_size()} pieces -> {tok_path}")
    assert tok.get_vocab_size() <= 65535, "uint16 stream"
    eot_h = tok.token_to_id("<eot_human>")
    eot_m = tok.token_to_id("<eot_model>")
    assert eot_h is not None and eot_m is not None

    def enc(text):
        return tok.encode(text).ids

    thanks_ids = enc(THANKS) + [eot_h]
    stream = TokenSink(os.path.join(out_dir, "tokens.bin"), spill=spill)
    events = []
    inst = Instruments(rng, long_pending=long_pending,
                       long_boost=long_boost, short_rate=short_rate)
    n_probes = n_nat = di = 0
    stats = {name: 0 for name, _, _ in sources}

    def add_instrument():
        nonlocal n_probes
        got = inst.maybe_convo(len(stream))
        if not got:
            return
        iturns, probes = got
        turn_pos = []
        for text, who in iturns:
            turn_pos.append(len(stream))
            stream.extend(enc(text))
            stream.append(eot_m if who == "model" else eot_h)
        if not probes:
            fresh = [f for f in inst.pending if f.get("plant") is None]
            for i, f in enumerate(fresh):
                off = len(enc(f"by the way {f['name']} kept a"))
                f["plant"] = turn_pos[2 * i] + off
                f["due"] = f["plant"] + f.pop("due_gap")
        for pr in probes:
            apos = turn_pos[pr["turn_idx"]] + len(enc(pr["prefix"]))
            if "plant_abs" in pr:
                plant = pr["plant_abs"]
            else:
                plant = turn_pos[pr["plant_turn"]] \
                    + len(enc(pr["plant_prefix"]))
            events.append({"pos": apos, "kind": "probe",
                           "answer": enc(" " + pr["answer"])[0],
                           "gap": max(apos - plant, 1),
                           "distractors": [enc(" " + c)[0]
                                           for c in pr["distractors"]]})
            n_probes += 1
        if probes:
            events.append({"pos": len(stream) - 1, "kind": "earned",
                           "ok": True})

    print("phase: stream", flush=True)
    iters = {name: factory() for name, _, factory in sources}
    weights = {name: w for name, w, _ in sources}
    live = set(iters)
    while len(stream) < budget_tokens and live:
        name = rng.choices([n for n in live],
                           [weights[n] for n in live])[0]
        try:
            doc = next(iters[name])
        except StopIteration:
            live.discard(name)
            continue
        if di % max(instrument_every, 1) == 0:
            add_instrument()
        if isinstance(doc, tuple):                    # digest pair
            _, code, summary = doc
            base = len(stream)
            code_ids = enc(code)
            stream.extend(code_ids)
            stream.append(eot_h)
            stream.extend(enc(summary))
            stream.append(eot_m)
            stats[name] += len(code_ids)
        else:
            turns = doc_to_turns(doc, rng)
            base = len(stream)
            if mine_ids and name == "code":
                # miner offsets assume the exact text streamed —
                # truncate FIRST, then mine, then emit as ONE human
                # turn so token positions line up exactly
                doc = doc[:60000]
                for ev in mine_identifier_probes(
                        doc, tok, base_pos=len(stream), rng=rng):
                    events.append(ev)
                    n_nat += 1
                stream.extend(enc(doc))
                stream.append(eot_h)
            else:
                for i, t in enumerate(turns):
                    stream.extend(enc(t))
                    stream.append(eot_h if i % 2 == 0 else eot_m)
            stats[name] += len(stream) - base
        stream.extend(thanks_ids)
        events.append({"pos": len(stream) - 1, "kind": "earned",
                       "ok": True})
        di += 1
        if di % 500 == 0:
            print(f"  {di} docs, {len(stream):,} tokens, "
                  f"{n_probes} planted + {n_nat} natural probes, "
                  f"mix {stats}", flush=True)
    total = stream.close()
    with open(os.path.join(out_dir, "events.jsonl"), "w") as f:
        for e in sorted(events, key=lambda e: e["pos"]):
            f.write(json.dumps(e) + "\n")
    print(f"wrote {total:,} tokens, {len(events)} events "
          f"({n_probes} planted, {n_nat} natural) mix={stats} "
          f"-> {out_dir}")
    return stats


def source_code(paths, language="Python"):
    return ("code", 0.55, lambda: iter_parquet_texts(
        paths, ["code", "content"], filter_col="language",
        filter_val=language))


def source_wiki(paths):
    return ("wiki", 0.20, lambda: iter_parquet_texts(paths, ["text"]))


def source_digest(paths):
    def it():
        for pair in iter_parquet_texts_pairs(paths):
            yield pair
    return ("digest", 0.15, it)


def iter_parquet_texts_pairs(paths):
    import pyarrow.parquet as pq
    for path in paths:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        ccol = next(c for c in ("whole_func_string", "function", "code")
                    if c in names)
        scol = next(c for c in ("func_documentation_string",
                                "docstring", "summary") if c in names)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=[ccol, scol]):
            d = batch.to_pydict()
            for code, summary in zip(d[ccol], d[scol]):
                if code and summary and len(code) > 200 and \
                        len(summary.split()) >= 4:
                    yield ("digest", code, summary.strip())


def source_chat(n_convos):
    def it():
        for turns in iter_convos(n_convos):
            yield "\n".join(turns)
    return ("chat", 0.10, it)

def iter_jsonl_texts(path, min_chars=200, max_chars=200_000):
    """v8.0 stream-extract-delete: code shards are downloaded one at
    a time, Python rows extracted to a compact jsonl, parquet deleted
    — 230 shards never coexist on disk. This iterates the result."""
    with open(path) as f:
        for line in f:
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue
            if text and len(text) >= min_chars:
                yield text[:max_chars]


def source_code_jsonl(path, weight=0.55):
    return ("code", weight, lambda: iter_jsonl_texts(path))
