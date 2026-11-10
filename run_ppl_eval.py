import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
from utils import load_model_and_tokenizer, add_common_args
from palu.quant_utils import configure_latent_quantizer
from loguru import logger

def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    from palu.data_utils import wiki_split
    if name == "wikitext2":
        return tokenizer("\n\n".join(wiki_split("test")["text"]), return_tensors="pt")
    if name == "c4":
        from pathlib import Path
        path = Path(__file__).parent / "dataset-benchmark/c4/c4-validation.00000-of-00008.json.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        valdata = load_dataset("json", data_files=str(path), split="train")
        result = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        result["input_ids"] = result.input_ids[:, :256*seqlen]
        return result
    raise ValueError(name)

@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda", max_samples=None):
    import math
    model.to(device).eval()
    results = {}
    stats = {}
    original_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        for dataset in datasets.split(","):
            ids = get_ppl_eval_loaders(dataset, tokenizer, seqlen).input_ids
            count = ids.numel() // seqlen
            if max_samples is not None:
                count = min(count, max_samples)
            if count < 1:
                raise ValueError("No complete PPL blocks")
            nll, tokens = 0., 0
            for i in tqdm(range(count), desc=f"PPL {dataset}", mininterval=5):
                batch = ids[:, i*seqlen:(i+1)*seqlen].to(device)
                hidden = model.model(batch, use_cache=False)[0]
                # Chunk the vocabulary projection to bound FP32-logit memory.
                for offset in range(0, seqlen-1, 128):
                    end = min(offset+128, seqlen-1)
                    logits = model.lm_head(hidden[:, offset:end]).float()
                    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              batch[:, offset+1:end+1].reshape(-1), reduction="sum")
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PPL loss")
                    nll += loss.item(); tokens += end-offset
            results[dataset] = math.exp(nll/tokens)
            import hashlib
            stats[dataset] = {"blocks": count, "seqlen": seqlen, "scored_tokens": tokens,
                "nll_sum": nll, "input_ids_sha256": hashlib.sha256(ids[:, :count*seqlen].numpy().tobytes()).hexdigest(),
                "split": "test" if dataset == "wikitext2" else "validation"}
            logger.info(f"{dataset}: PPL={results[dataset]}, blocks={count}, targets={tokens}")
    finally:
        model.config.use_cache = original_cache
    eval_ppl.last_stats = stats
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--datasets', type=str, help='datasets to evaluate', default='wikitext2')
    parser.add_argument('--seqlen', type=int, help='sequence length for ppl evaluation', default=2048)
    parser.add_argument("--device", type=str, help="device to run the model on", default="cuda")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information or not.")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output_json", type=str)
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    
    configure_latent_quantizer(
        model, n_bits=args.lt_bits,
        group_size=args.lt_group_size,
        sym=args.lt_sym,
        clip_ratio=args.lt_clip_ratio,
        hadamard=args.lt_hadamard
    )
    logger.info(f"Start evaluating ppl...")
    logger.info(f"*model: {args.model_name_or_path}")
    logger.info(f"*datasets: {args.datasets}")
    logger.info(f"*sequence length {args.seqlen}")
    results = eval_ppl(model, tokenizer, args.model_name_or_path, args.datasets, args.seqlen, args.device, args.max_samples)
    for dataset, ppl in results.items():
        logger.info(f"PPL: {ppl}")
    if args.output_json:
        import json
        from pathlib import Path
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({"args": vars(args), "results": results, "statistics": eval_ppl.last_stats}, indent=2))
