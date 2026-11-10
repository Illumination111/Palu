"""Fisher rank selection; HF performs the causal-label shift exactly once."""
import json
from pathlib import Path
import torch
from tqdm import tqdm
from loguru import logger
from .data_utils import get_calib_data, CACHE, digest

def kv_modules(model):
    return {n: m for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n.endswith(("k_proj", "v_proj"))}

def calib_fisher_info(model, calib_loader, device, use_cache=True, legacy_double_shift=False):
    targets = kv_modules(model)
    meta = {"data": calib_loader.metadata, "protocol": "legacy_double_shift" if legacy_double_shift else "hf_single_shift_v2",
            "source": digest(Path(__file__).read_text())}
    cache_file = CACHE / f"fisher-{digest(meta)}.pt"
    if use_cache and cache_file.exists():
        saved = torch.load(cache_file, map_location="cpu", weights_only=False)
        assert saved["meta"] == meta
        for n, m in targets.items():
            m.fisher_info = saved["fisher"][n]
        logger.info(f"Fisher cache: {cache_file}")
        return
    model.to(device).eval()  # Preserve FP32 rotary-frequency buffers.
    original_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    original_cache = model.config.use_cache
    model.config.use_cache = False
    model.requires_grad_(False)
    for m in targets.values():
        m.weight.requires_grad_(True)
        m.fisher_info = torch.zeros_like(m.weight, dtype=torch.float32)
    losses = []
    try:
        for batch in tqdm(calib_loader, desc="Fisher samples", mininterval=5):
            ids = batch["input_ids"].to(device)
            inputs, labels = (ids[:, :-1], ids[:, 1:]) if legacy_double_shift else (ids, ids)
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Fisher loss")
            loss.backward()
            losses.append(loss.item())
            for n, m in targets.items():
                if m.weight.grad is None or not torch.isfinite(m.weight.grad).all():
                    raise FloatingPointError(f"Invalid Fisher gradient: {n}")
                m.fisher_info.add_(m.weight.grad.detach().float().square())
        for m in targets.values():
            m.fisher_info = (m.fisher_info / len(calib_loader)).sqrt().cpu()
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": meta, "losses": losses, "fisher": {n:m.fisher_info for n,m in targets.items()}}, cache_file)
        logger.info(f"Fisher saved: {cache_file}; mean loss={sum(losses)/len(losses):.6f}")
    finally:
        model.zero_grad(set_to_none=True)
        for n, p in model.named_parameters():
            p.requires_grad_(original_grad[n])
        model.config.use_cache = original_cache
        torch.cuda.empty_cache()

def allocate_ranks(weights, cap, ratio, multiplicity=1, block=32):
    # Allocation is in integer blocks; equal groups in a module share one decision.
    count = len(weights)
    cap_blocks = cap // block
    target = int(count * cap_blocks * ratio + 0.5)
    if not count <= target <= count * cap_blocks:
        raise ValueError("Target cannot satisfy the minimum rank block")
    weights = [max(float(w), 1e-20) for w in weights]
    ranks = [1] * count
    # Discrete proportional allocation with saturation and deterministic tie order.
    for _ in range(target-count):
        eligible = [i for i in range(count) if ranks[i] < cap_blocks]
        best = max(eligible, key=lambda i: weights[i] / (ranks[i] + 0.5))
        ranks[best] += 1
    return [r * block for r in ranks]

def rank_search(model, tokenizer, args):
    if not 0 < args.param_ratio_target <= 1:
        raise ValueError("param_ratio_target must be in (0, 1]")
    kv_heads = model.config.num_key_value_heads
    if kv_heads % args.head_group_size:
        raise ValueError("KV heads must be divisible by group size")
    groups = kv_heads // args.head_group_size
    cap = args.head_group_size * (model.config.hidden_size // model.config.num_attention_heads)
    targets = kv_modules(model)
    if args.search_method != "uniform":
        loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id,
                               args.n_fisher_calib_samples, args.calib_seqlen, args.seed)
        calib_fisher_info(model, loader, args.device, args.use_cache,
                          getattr(args, "legacy_double_shift", False))
    keys, weights = [], []
    for n, m in targets.items():
        if args.search_method == "fisher":
            for i, f in enumerate(m.fisher_info.reshape(groups, -1, m.in_features)):
                keys.append((n, i)); weights.append(f.mean().item())
        else:
            keys.append((n, None))
            weights.append(1.0 if args.search_method == "uniform" else m.fisher_info.mean().item())
    values = allocate_ranks(weights, cap, args.param_ratio_target)
    selected = {n: [0]*groups for n in targets}
    for (n, i), value in zip(keys, values):
        if i is None:
            selected[n] = [value]*groups
        else:
            selected[n][i] = value
    for m in targets.values():
        if hasattr(m, "fisher_info"):
            del m.fisher_info
    rank_sum = sum(map(sum, selected.values()))
    total = len(targets)*groups*cap
    logger.info(f"Selected rank fraction {rank_sum/total:.6f}; requested={args.param_ratio_target}")
    return selected, rank_sum, total
