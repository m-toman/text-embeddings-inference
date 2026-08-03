# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.47",
#   "sentencepiece",
# ]
# ///
"""Verify a TEI server's SigLIP text embeddings against the canonical transformers reference.

The canonical reference is `SiglipTextModel.pooler_output` with the checkpoint's own
tokenizer, `padding="max_length", max_length=64`, and NO attention mask (SigLIP is
trained without one: it attends to every position, padding included, and pools the
final -- padding -- position).

Exit code 0 iff every probe reaches cosine >= 0.999 between TEI and the reference.

Usage:
    uv run verify_tei_siglip.py --tei-url http://localhost:8080 \
        [--model-id google/siglip2-base-patch16-224]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

PROBES = [
    "a dog running on grass",
    "the quick brown fox jumps over the lazy dog",
    "satellite image of a city at night",
    "a red apple on a wooden table",
    "portrait photo of a smiling elderly woman",
    "diagram of a neural network architecture",
    "x",
    "an extremely long caption describing a scene with many people walking through a busy "
    "market street full of colorful stalls selling fruit vegetables spices and textiles "
    "under a bright midday sun while vendors call out to passing customers",
]

THRESHOLD = 0.999


def tei_embed(tei_url: str, texts: list[str]) -> "torch.Tensor":
    import torch

    req = urllib.request.Request(
        tei_url.rstrip("/") + "/embed",
        data=json.dumps({"inputs": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return torch.tensor(json.loads(resp.read()), dtype=torch.float32)


def reference_embed(model_id: str, texts: list[str]) -> "torch.Tensor":
    import torch
    from transformers import AutoTokenizer, SiglipTextModel

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = SiglipTextModel.from_pretrained(model_id).eval()
    enc = tokenizer(
        texts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        # No attention mask on purpose: canonical SigLIP inference is unmasked.
        out = model(input_ids=enc["input_ids"])
    return out.pooler_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tei-url", required=True, help="Base URL of the TEI server")
    parser.add_argument(
        "--model-id",
        default="google/siglip2-base-patch16-224",
        help="HF model id for the canonical reference (text tower is used)",
    )
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    print(f"TEI server : {args.tei_url}")
    print(f"Reference  : {args.model_id} (SiglipTextModel.pooler_output, "
          f"padding=max_length 64, unmasked)")
    print()

    tei = F.normalize(tei_embed(args.tei_url, PROBES), dim=-1)
    ref = F.normalize(reference_embed(args.model_id, PROBES), dim=-1)
    cosines = (tei * ref).sum(-1).tolist()

    shown = [t if len(t) <= 60 else t[:57] + "..." for t in PROBES]
    width = max(len(t) for t in shown)
    ok = True
    print(f"{'probe':<{width}}  cosine    status")
    for text, cos in zip(shown, cosines):
        passed = cos >= THRESHOLD
        ok &= passed
        print(f"{text:<{width}}  {cos:.6f}  {'PASS' if passed else 'FAIL'}")
    print()

    # Lowercase handling: SigLIP text towers are trained on lowercased text
    # (the slow tokenizer sets do_lower_case=True), but TEI's fast-tokenizer
    # pipeline does NOT lowercase. Callers must lowercase before embedding.
    upper, lower = tei_embed(args.tei_url, ["DOG", "dog"])
    upper, lower = F.normalize(upper, dim=-1), F.normalize(lower, dim=-1)
    case_cos = float((upper * lower).sum())
    differs = case_cos < 0.9999
    print(f'Lowercase check: TEI("DOG") vs TEI("dog") cosine = {case_cos:.6f} '
          f"-> {'DIFFERENT' if differs else 'identical'}")
    if differs:
        print("  NOTE: TEI does not lowercase inputs; SigLIP was trained on lowercased")
        print("  text, so callers MUST lowercase text before sending it to /embed.")
    print()

    if ok:
        print(f"ALL PROBES PASS (cosine >= {THRESHOLD})")
        return 0
    print(f"FAILURE: at least one probe below cosine {THRESHOLD}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
