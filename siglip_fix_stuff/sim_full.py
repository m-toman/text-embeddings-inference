import os
os.environ["HF_HUB_CACHE"] = "/tmp/claude-1000/-monorepo/b8019ca8-96e8-4c65-91e8-b3e64fcadd04/scratchpad/hf_cache"
import json, urllib.request
import torch
import torch.nn.functional as F
from transformers import SiglipTextModel, AutoTokenizer

MID = "google/siglip2-base-patch16-224"
model = SiglipTextModel.from_pretrained(MID).eval()
tok = AutoTokenizer.from_pretrained(MID)

texts = [
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

req = urllib.request.Request("http://172.19.0.1:9923/embed",
    data=json.dumps({"inputs": texts}).encode(), headers={"Content-Type": "application/json"})
tei = F.normalize(torch.tensor(json.loads(urllib.request.urlopen(req, timeout=60).read())), dim=-1)

def sim(pad_id):
    rows = []
    for t in texts:
        ids = tok(t, truncation=True, max_length=64)["input_ids"]
        ids = ids + [pad_id] * (64 - len(ids))
        rows.append(ids)
    ids = torch.tensor(rows)
    with torch.no_grad():
        return F.normalize(model(input_ids=ids).pooler_output, dim=-1)

bug = sim(1)      # TEI's behavior: pads with config-default pad_token_id = 1 (<eos>)
canon = sim(0)    # canonical: pads with tokenizer pad id 0 (<pad>)

print(f"{'probe':<50s} {'sim-bug(pad=1) vs TEI':>22s} {'canonical(pad=0) vs TEI':>24s}")
for i, t in enumerate(texts):
    print(f"{t[:48]:<50s} {float((bug[i]*tei[i]).sum()):>22.6f} {float((canon[i]*tei[i]).sum()):>24.6f}")
print("min sim-bug cos:", float((bug*tei).sum(-1).min()))
