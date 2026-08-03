# PR: fix(siglip): pad with the tokenizer's pad token id, not the config default

## Title

`fix(siglip): resolve pad token id from tokenizer files (SigLIP2 pads with <pad>=0, not 1)`

## What was wrong

`SiglipTextModel::forward` right-pads every sequence to `max_position_embeddings` (64)
using `SiglipTextConfig::pad_token_id`, which is read from `config.json` with a serde
default of `1` (`backends/candle/src/models/siglip.rs`, `default_text_pad_token_id`).

That default matches the transformers `SiglipTextConfig` default — but transformers
never pads with `config.pad_token_id`: padding is done by the **tokenizer**
(`padding="max_length", max_length=64`). The two agree for SigLIP **v1**, whose
sentencepiece tokenizer pads with `</s>` = 1. They do **not** agree for **SigLIP2**,
which uses the Gemma tokenizer: its pad token is `<pad>` = **0**, and id 1 is `<eos>`.
Since the `google/siglip2-*` checkpoints ship a `text_config` containing only
`model_type` and `vocab_size`, the wrong default is always used for SigLIP2.

The wrong pad token is unusually damaging for this architecture, because the
implementation (correctly) follows canonical SigLIP inference:

- **no attention mask** — every position, padding included, participates in attention
  (see `transformers/models/siglip/modeling_siglip.py`, and the SigLIP model cards,
  which always embed with `padding="max_length"` and no mask);
- **pooling takes the final position (index 63)** before the projection head
  (`SiglipTextTransformer`: `pooled_output = last_hidden_state[:, -1, :]`), and for any
  input shorter than 64 tokens that position **is a padding slot**.

So padding with `<eos>` instead of `<pad>` perturbs all 64 positions *and* swaps the
very token whose hidden state is pooled.

## Impact / evidence

Serving `m-toman/siglip2-base-patch16-224-text` (a faithful sentence-transformers
export of the `google/siglip2-base-patch16-224` text tower), `/embed` outputs reach
only **cos ≈ 0.84–0.91** against the canonical reference
(`SiglipTextModel.pooler_output`, padded to 64 with the Gemma tokenizer, unmasked).

Reproducing the bug in PyTorch — same canonical model, but with pad slots filled with
id 1 instead of 0 — matches the live TEI output to **cos ≥ 0.9999999 on every probe**
(8 probes incl. a >64-token truncated caption), confirming the pad id is the sole
divergence:

| probe | sim-bug (pad=1) vs TEI | canonical (pad=0) vs TEI |
|---|---|---|
| a dog running on grass | 1.000000 | 0.837469 |
| the quick brown fox jumps over the lazy dog | 1.000000 | 0.896580 |
| satellite image of a city at night | 1.000000 | 0.835207 |
| a red apple on a wooden table | 1.000000 | 0.854558 |
| portrait photo of a smiling elderly woman | 1.000000 | 0.893961 |
| diagram of a neural network architecture | 1.000000 | 0.907189 |
| x | 1.000000 | 0.904973 |
| 240-char caption (truncated to 64 tokens) | 1.000000 | 0.966151 |

The existing `test_siglip.rs` snapshots didn't catch this because they run
`google/siglip-base-patch16-224` (v1), where pad = `</s>` = 1 is accidentally correct.

## The fix

In `CandleBackend::new` (`backends/candle/src/lib.rs`), resolve the tokenizer's actual
pad token id from the model directory before constructing `SiglipTextModel`, and
override `config.text_config.pad_token_id` with it (warning when it differs):

1. `tokenizer.json` → top-level `padding.pad_id` — SigLIP2 checkpoints ship a
   fixed-to-64 padding strategy with `pad_id: 0` (the router strips it via
   `tokenizer.with_padding(None)`, so the model-side padding must agree with it);
2. fallback: `tokenizer_config.json` → `pad_token` (plain string or AddedToken-style
   object), mapped to its id through `tokenizer.json`'s `added_tokens`;
3. fallback: keep the `config.json` value (unchanged behavior).

For SigLIP v1 every source resolves to 1, so existing snapshots are unaffected. For
SigLIP2 the pad id resolves to 0 and `/embed` matches the transformers reference.

## How to verify

1. Rebuild and serve a SigLIP2 text checkpoint, e.g.
   `text-embeddings-router --model-id m-toman/siglip2-base-patch16-224-text`.
   Startup now logs
   `Overriding SigLIP `pad_token_id` from `config.json` (1) with the tokenizer's pad token id (0)`.
2. Run the included standalone check (needs the TEI URL; downloads
   `google/siglip2-base-patch16-224` as reference):

   ```bash
   uv run verify_tei_siglip.py --tei-url http://localhost:8080
   ```

   Before this fix every probe fails at cos ≈ 0.84–0.97; after it, all probes are
   ≥ 0.999 and the script exits 0.
3. `cargo test -p text-embeddings-backend-candle test_siglip` — v1 snapshots unchanged.

Note for users (unchanged by this PR): TEI's fast-tokenizer path does not lowercase,
while SigLIP's training text was lowercased (`do_lower_case: true` applies to the slow
tokenizer only) — clients should lowercase text before calling `/embed`.
