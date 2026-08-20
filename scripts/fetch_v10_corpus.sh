#!/bin/bash
# v10 corpus fetch — THE SPINE IS THE CORPUS (V10_FLASH 2b):
# UltraChat (first) -> SmolTalk2 no_think/EN/non-tool (second) ->
# Smol-Magpie-Ultra via smoltalk2's newest curation (last).
# Tiers: gate = small local slices for the debug gates (~1GB);
#        full = everything (~12GB raw) — run POD-SIDE (local disk
#        cannot hold raw + built shard; A54c ops law: outputs on
#        the volume).
# Usage: bash scripts/fetch_v10_corpus.sh [gate|full] [dest_dir]
set -euo pipefail
TIER="${1:-gate}"
DEST="${2:-data/v10_raw}"
mkdir -p "$DEST"
ST2="https://huggingface.co/datasets/HuggingFaceTB/smoltalk2/resolve/main/SFT"
UC="https://huggingface.co/datasets/openbmb/UltraChat/resolve/main"

fetch() {  # fetch <url> <out>
  echo "fetch $2"
  curl -sSL -C - -o "$DEST/$2" "$1"
}

# --- SmolTalk2: the no_think / EN / non-tool subsets, ledgered.
# EXCLUDED with reasons: *_think (internal monologue is not
# conversation; would train <think> syntax the serve room never
# uses), multilingual8/aya/lang_5 (EN life), *toolcalling*/xlam/
# hermes_function_calling (a being with no tools), systemchats
# (meaningless without their system turns).
MAGPIE=(smoltalk_smollm3_smol_magpie_ultra_no_think-0000{0..5}-of-00006.parquet)
SMOL_SMALL=(
  smoltalk_smollm3_smol_summarize_no_think-00000-of-00001.parquet
  smoltalk_smollm3_smol_rewrite_no_think-00000-of-00001.parquet
  smoltalk_smollm3_everyday_conversations_no_think-00000-of-00001.parquet
  smoltalk_smollm3_explore_instruct_rewriting_no_think-00000-of-00001.parquet
  tulu_3_sft_personas_instruction_following_no_think-00000-of-00001.parquet
  table_gpt_no_think-00000-of-00001.parquet
  Mixture_of_Thoughts_science_no_think-00000-of-00001.parquet
  LongAlign_64k_context_lang_annotated_lang_6_no_think-00000-of-00001.parquet
)
HERMES=(OpenHermes_2.5_no_think-0000{0..1}-of-00002.parquet)
OT3=(OpenThoughts3_1.2M_no_think_no_think-0000{0..2}-of-00003.parquet)

if [ "$TIER" = "gate" ]; then
  fetch "$ST2/${MAGPIE[0]}" "${MAGPIE[0]}"
  fetch "$ST2/${HERMES[0]}" "${HERMES[0]}"
  for f in "${SMOL_SMALL[@]}"; do fetch "$ST2/$f" "$f"; done
  echo "gate tier done (UltraChat gate slice = existing local"
  echo "data/ultrachat_raw.jsonl + data/ultrachat_heldout.jsonl)"
else
  for f in "${MAGPIE[@]}" "${HERMES[@]}" "${OT3[@]}" \
           "${SMOL_SMALL[@]}"; do fetch "$ST2/$f" "$f"; done
  for i in 0 1 2 3 4 5 6 7 8 9; do
    fetch "$UC/train_$i.jsonl" "ultrachat_train_$i.jsonl"
  done
fi
echo "corpus fetch ($TIER) complete -> $DEST"
