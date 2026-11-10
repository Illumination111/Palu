"""Deterministic, token-window calibration with content-addressed local caches."""
import hashlib
import json
import os
import random
from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT.parent / "kvtc/dataset-benchmark/wikitext-2-raw-v1-arrow"
CACHE = Path(os.environ.get("PALU_CACHE", str(ROOT / "cache/local_v2")))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:24]

def model_identity(model_id):
    path = Path(model_id)
    files = sorted(path.glob("*.json")) + sorted(path.glob("*.safetensors"))
    return {"path": str(path.resolve()), "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]}

class CalibrationData(list):
    pass

def wiki_split(split):
    path = Path(os.environ.get("PALU_WIKITEXT", str(WIKI)))
    if not path.exists():
        raise FileNotFoundError(f"Local WikiText Arrow dataset missing: {path}")
    return load_from_disk(str(path))[split]

def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=42):
    if name != "wikitext2":
        raise ValueError("Local calibrated protocol currently supports wikitext2 only")
    if nsamples < 1 or seqlen < 2:
        raise ValueError("Calibration requires positive sample count and >=2 tokens")
    data = wiki_split("train")
    meta = {"protocol": "token_windows_v2", "model": model_identity(model_id),
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1", "split": "train",
            "fingerprint": data._fingerprint, "seed": seed, "nsamples": nsamples,
            "seqlen": seqlen, "dtype": "float16", "tokenizer": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer), "source": digest(Path(__file__).read_text())}
    CACHE.mkdir(parents=True, exist_ok=True)
    file = CACHE / f"windows-{digest(meta)}.pt"
    if file.exists():
        saved = torch.load(file, map_location="cpu", weights_only=False)
        assert saved["meta"] == meta
        ids, offsets = saved["ids"], saved["offsets"]
    else:
        ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
        rng = random.Random(seed)
        offsets = [rng.randrange(ids.shape[1] - seqlen + 1) for _ in range(nsamples)]
        ids = torch.cat([ids[:, i:i+seqlen] for i in offsets], dim=0)
        torch.save({"meta": meta, "ids": ids, "offsets": offsets}, file)
    assert ids.shape == (nsamples, seqlen)
    result = CalibrationData({"input_ids": row[None], "attention_mask": torch.ones_like(row[None])} for row in ids)
    result.metadata = dict(meta, offsets=offsets)
    result.cache_key = digest(result.metadata)
    return result
