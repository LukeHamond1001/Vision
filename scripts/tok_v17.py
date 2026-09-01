#!/usr/bin/env python3
"""tok_v17.py — the next body's tokenizer (law 15): the caretaker's four
presses stay where they were, and the child gets its own face —
<me+1> <me+2> <me-1> <me-2> — appended as special tokens.

  python3 scripts/tok_v17.py data/ship_tok.json data/ship_tok_v17.json
"""
import sys
from tokenizers import Tokenizer

ME = ["<me+1>", "<me+2>", "<me-1>", "<me-2>"]

def main():
    src, dst = (sys.argv + ["data/ship_tok.json", "data/ship_tok_v17.json"])[1:3]
    t = Tokenizer.from_file(src)
    v0 = t.get_vocab_size()
    t.add_special_tokens(ME)
    t.save(dst)
    print("vocab %d -> %d; %s -> %s" % (v0, t.get_vocab_size(),
                                        {s: t.token_to_id(s) for s in ME}, dst))

if __name__ == "__main__":
    main()
