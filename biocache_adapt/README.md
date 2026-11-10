# Palu BioCache proxy adapter

This package provides the shared text-proxy API:

`reset`, `prefill_archive`, `query`, `persistent_state`,
`static_extra_state`, and `temporary_state`.

The persistent metric contains only Palu's latent K/V cache. Full K/V
reconstruction is query-local workspace; Palu's added `VT`, `U_list`, and `B`
projection parameters are static extra state. `kernel.palu_attention` is
imported only by `load_palu_attention()`/`wrap_model_true_latent()`, so the
core adapter and dry-run do not require torch, transformers, or the kernel.

Run the no-model protocol check:

```bash
python -m biocache_adapt --dry-run --budget 25% \
  --output /tmp/palu.jsonl --manifest /tmp/palu-manifest.json
```

The accepted actual-byte budgets are 12.5%, 25%, and 50%. JSONL records use
`biocache-proxy-v1` for both needle retrieval and length scans, with separate
compression, recovery, generation, and network timing fields.
