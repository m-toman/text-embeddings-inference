import os
os.environ["HF_HUB_CACHE"] = "/tmp/claude-1000/-monorepo/b8019ca8-96e8-4c65-91e8-b3e64fcadd04/scratchpad/hf_cache"
import json, urllib.request
import torch
from transformers import SiglipTextModel, AutoTokenizer

MID = "google/siglip2-base-patch16-224"
model = SiglipTextModel.from_pretrained(MID).eval()
tok = AutoTokenizer.from_pretrained(MID)
print("tokenizer pad id:", tok.pad_token_id, "eos:", tok.eos_token_id)

texts = ["a dog running on grass", "the quick brown fox", "satellite image of a city at night", "DOG"]

# TEI live
req = urllib.request.Request("http://172.19.0.1:9923/embed",
    data=json.dumps({"inputs": texts}).encode(), headers={"Content-Type": "application/json"})
tei = torch.tensor(json.loads(urllib.request.urlopen(req, timeout=30).read()))

def run(pad_id, masked):
    enc = tok(texts, padding="max_length", max_length=64, return_tensors="pt", return_attention_mask=True)
    ids = enc["input_ids"].clone()
    ids[ids == tok.pad_token_id] = pad_id if pad_id != tok.pad_token_id else pad_id
    if pad_id != tok.pad_token_id:
        # replace pad positions
        pad_pos = enc["input_ids"] == tok.pad_token_id
        ids = enc["input_ids"].clone()
        ids[pad_pos] = pad_id
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=enc["attention_mask"] if masked else None)
    return out.pooler_output

def cos(a, b):
    a = torch.nn.functional.normalize(a, dim=-1); b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(-1)

for pad_id, masked, label in [(0, False, "canonical pad=0 unmasked"),
                              (1, False, "sim BUG pad=1 unmasked"),
                              (1, True,  "sim pad=1 masked")]:
    p = run(pad_id, masked)
    print(f"{label:28s} cos vs TEI:", [f"{c:.6f}" for c in cos(p, tei).tolist()])
