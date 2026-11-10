from loguru import logger
import torch.nn as nn
import torch
import os
import click
from tqdm import tqdm
from .data_utils import get_calib_data
from .model import HeadwiseLowRankModule

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def get_whiten_scale_matrix(model, tokenizer, args, dev):
    from .data_utils import CACHE, digest
    from pathlib import Path
    loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id,
                           args.n_whiten_calib_samples, args.whiten_seqlen, args.seed)
    meta = {"data": loader.metadata, "protocol": "kv_input_cov_v2", "source": digest(Path(__file__).read_text())}
    cache_file = CACHE / f"whiten-{digest(meta)}.pt"
    layers = model.model.layers
    if args.use_cache and cache_file.exists():
        saved = torch.load(cache_file, map_location="cpu", weights_only=False)
        assert saved["meta"] == meta
        scales = saved["scales"]
    else:
        model.to(dev).eval()
        covs = [torch.zeros((model.config.hidden_size,)*2, device=dev, dtype=torch.float32) for _ in layers]
        handles = []
        def hook_for(index):
            def hook(module, inputs):
                x = inputs[0].detach().reshape(-1, model.config.hidden_size).float()
                covs[index].addmm_(x.T, x)
            return hook
        for i, layer in enumerate(layers):
            handles.append(layer.self_attn.k_proj.register_forward_pre_hook(hook_for(i)))
        old_cache = model.config.use_cache
        model.config.use_cache = False
        try:
            for batch in tqdm(loader, desc="Whitening samples", mininterval=5):
                model.model(input_ids=batch["input_ids"].to(dev), use_cache=False)
        finally:
            for h in handles: h.remove()
            model.config.use_cache = old_cache
        scales, jitters = [], []
        for i in tqdm(range(len(layers)), desc="Whitening Cholesky", mininterval=5):
            cov = covs[i].double() / (len(loader)*args.whiten_seqlen)
            covs[i] = None
            if not torch.isfinite(cov).all():
                raise FloatingPointError(f"Nonfinite covariance layer {i}")
            cov = (cov+cov.T)/2
            diagonal_mean = cov.diagonal().mean().item()
            identity = torch.eye(cov.shape[0], device=dev, dtype=torch.float64)
            for relative in (0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
                chol, info = torch.linalg.cholesky_ex(cov + identity*(relative*diagonal_mean))
                if info.item() == 0:
                    break
            else:
                raise RuntimeError(f"Covariance not positive definite: layer {i}")
            scales.append(chol.float().cpu()); jitters.append(relative)
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": meta, "scales": scales, "relative_jitter": jitters}, cache_file)
        logger.info(f"Whitening saved: {cache_file}; jitter={jitters}")
    for layer, scale in zip(layers, scales):
        layer.self_attn.k_proj.scaling_diag_matrix = scale
        layer.self_attn.v_proj.scaling_diag_matrix = scale
    torch.cuda.empty_cache()

def compress_model_whiten(model, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    # NOTE(brian1009): Prepare whiten scaling matrix
    get_whiten_scale_matrix(model, tokenizer, args, dev)
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
    
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten(
            raw_linear,
            selected_head_rank
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

def compress_model_svd(model, selection_result):
    logger.info("Compressing model with svd decomposition...")
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        print("head-wise svd", layername, raw_linear)
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear(
            raw_linear,
            selected_head_rank
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

# Wrapper for different decompose methods
def compress_model(model, tokenizer, args, dev, selection_result):
    if args.decompose_method == "whiten":
        compress_model_whiten(model, tokenizer, args, dev, selection_result)
    elif args.decompose_method == "svd":
        compress_model_svd(model, selection_result)
    else:
        raise ValueError(f"Decomposition method {args.decompose_method} is not supported.")