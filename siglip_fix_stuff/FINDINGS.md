# TEI SigLIP text-model fidelity bug — root cause

## Verdict

**Root cause found and empirically proven (cos ≈ 1.000000 reproduction): TEI pads
SigLIP2 inputs with the wrong pad token id (1 = `<eos>` instead of 0 = `<pad>`).**

## The bug

`backends/candle/src/models/siglip.rs` (branch `feat/add-siglip-text-model`,
commit `7d2c58f`):

- **Lines 43–45**: the serde default for the text config's pad token id is 1:

  ```rust
  fn default_text_pad_token_id() -> u32 {
      1
  }
  ```

- **Lines 76–77**: `pad_token_id` is read from `config.json`'s `text_config` with that
  default:

  ```rust
  #[serde(default = "default_text_pad_token_id")]
  pub pad_token_id: u32,
  ```

- **Lines 327–330** (`SiglipTextModel::forward`): every sequence is right-padded to
  `max_position_embeddings` (64) **inside the model** using that config value:

  ```rust
  // Pad up to `padded_len` with pad token </s>.
  for _ in seq_len..padded_len {
      input_ids.push(self.pad_token_id);
  }
  ```

  (The router deliberately strips tokenizer padding — `router/src/lib.rs:153`
  `tokenizer.with_padding(None);` — so this in-model padding is the only padding.)

Why the wrong id is catastrophic for SigLIP specifically:

1. **`google/siglip2-*` checkpoints ship a `text_config` containing only `model_type`
   and `vocab_size`** (verified for `google/siglip2-base-patch16-224` and the
   `m-toman/siglip2-base-patch16-224-text` export). So the serde default `1` is always
   used.
2. **The transformers default (`SiglipTextConfig.pad_token_id = 1`) is only correct for
   SigLIP v1**, whose sentencepiece tokenizer pads with `</s>` = 1. SigLIP2 uses the
   **Gemma tokenizer**, whose pad token is `<pad>` = **0** (id 1 is `<eos>`).
   transformers never uses `config.pad_token_id` at inference — the *tokenizer* does the
   padding (`padding="max_length", max_length=64`, pad id 0).
3. SigLIP runs **without an attention mask** (it attends to every position, padding
   included — the fork gets this right, `siglip.rs:336–337`) and **pools the final
   position, index 63** (`siglip.rs:344–354`) — which for any input shorter than 64
   tokens **is a padding slot**. So the wrong pad token both perturbs attention at all
   64 positions and changes the very token whose hidden state is pooled.

Net effect: TEI computes the pooler output of the text padded with `<eos>` repeated,
instead of `<pad>` repeated → cos ≈ 0.84–0.91 vs the canonical embedding.

## Why the fork's own tests didn't catch it

`backends/candle/tests/test_siglip.rs` tests `google/siglip-base-patch16-224`
(SigLIP **v1**, vocab 32000), where the v1 tokenizer's pad token `</s>` really is id 1 —
the default is accidentally correct there. The bug only manifests on SigLIP2
checkpoints (vocab 256000, Gemma tokenizer).

## Empirical proof

Canonical reference: transformers `SiglipTextModel` for `google/siglip2-base-patch16-224`,
`pooler_output`, inputs padded to 64, no attention mask. Live TEI server
(`m-toman/siglip2-base-patch16-224-text`, TEI fork 1.9.3) at `http://172.19.0.1:9923`.

Simulated bug = same reference model, but pad slots filled with id **1** instead of 0
(exactly what the Rust code does), still unmasked, pooled at index 63:

| probe | sim-bug (pad=1) vs TEI | canonical (pad=0) vs TEI |
|---|---|---|
| a dog running on grass | **1.000000** | 0.837469 |
| the quick brown fox jumps over the lazy dog | **1.000000** | 0.896580 |
| satellite image of a city at night | **1.000000** | 0.835207 |
| a red apple on a wooden table | **1.000000** | 0.854558 |
| portrait photo of a smiling elderly woman | **1.000000** | 0.893961 |
| diagram of a neural network architecture | **1.000000** | 0.907189 |
| x | **1.000000** | 0.904973 |
| 240-char caption (truncated to 64 tokens) | **1.000000** | 0.966151 |

Minimum sim-bug cosine across all probes: **0.99999988**. The single wrong pad id
explains TEI's output exactly, including the truncation path. (Masked variants and all
alternative pooling variants were ruled out earlier; e.g. pad=1 + masked attention gives
only ~0.66–0.71.)

Everything else in the port checks out against
`transformers/models/siglip/modeling_siglip.py`:

- **Activation**: config default `HiddenAct::Gelu` maps to candle `Tensor::gelu()`,
  which *is* the tanh approximation — matches `gelu_pytorch_tanh`. Not a bug.
- **Attention over padding**: unmasked, matching canonical SigLIP usage. Correct.
- **Head/pooling order**: pool position 63 of the post-final-layernorm hidden state,
  then `head` Linear — matches `SiglipTextTransformer`
  (`pooled = last_hidden[:, -1]; head(pooled)`). Correct.
- **LayerNorm eps (1e-6), position embeddings (0..63), no embedding scaling**: correct.
- **Pooling-config mapping**: the model ignores the ST `1_Pooling` config and always
  does last-position + head, which is the right behavior for this architecture.

## The fix

`fix.patch` (against `7d2c58f`) modifies `backends/candle/src/lib.rs` only: before
constructing `SiglipTextModel`, resolve the tokenizer's actual pad token id from the
model directory —

1. `tokenizer.json` → `padding.pad_id` (SigLIP2 checkpoints ship a fixed-to-64 padding
   strategy with `pad_id: 0`);
2. fallback: `tokenizer_config.json` → `pad_token` (string or AddedToken object),
   mapped to its id via `tokenizer.json` `added_tokens`;
3. fallback: keep the config value (preserves current SigLIP v1 behavior).

and override `config.text_config.pad_token_id` with it (with a `tracing::warn!` when it
differs). For SigLIP v1 both sources resolve to 1, so the existing snapshot tests are
unaffected; for SigLIP2 they resolve to 0, which by the table above yields cos 1.000
against the canonical reference.

## Separate caller-side caveat (not a TEI bug)

TEI's fast-tokenizer path does **not** lowercase, while SigLIP was trained on lowercased
text (`do_lower_case: true` in `tokenizer_config.json` applies to the slow tokenizer
only). Live check: TEI("DOG") vs TEI("dog") cosine = 0.9358 — different. Callers must
lowercase before calling `/embed`. `verify_tei_siglip.py` reports this check.
