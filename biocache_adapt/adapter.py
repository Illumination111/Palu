"""Palu text proxy with explicit latent-cache and reconstruction accounting."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .common import PhaseTimer, budget_report, normalize_budget_ratio, recursive_nbytes


class PaluAdapter:
    """Wrap Palu without importing torch/transformers or its kernel at import time.

    ``prefill`` must return the actual latent cache (or a mapping containing
    ``latent_cache``).  Dense reconstruction is query-local and is never
    reported as persistent state.
    """

    def __init__(
        self,
        *,
        budget_ratio: float | str = 0.25,
        prefill: Callable[[str], Any] | None = None,
        generate: Callable[[str, Any], Any] | None = None,
        reconstruct: Callable[[Any], Any] | None = None,
        model: Any = None,
        dry_run: bool = False,
        strict_budget: bool = False,
    ) -> None:
        self.budget_ratio = normalize_budget_ratio(budget_ratio)
        self._prefill = prefill
        self._generate = generate
        self._reconstruct = reconstruct
        self.model = model
        self.dry_run = dry_run
        self.strict_budget = strict_budget
        self._timer = PhaseTimer()
        self.reset()

    @property
    def timings_ms(self) -> Mapping[str, float]:
        return dict(self._timer.ms)

    def reset(self) -> None:
        self._latent_cache = None
        self._dense_workspace = None
        self._dense_reference = None
        self._segment_id = None
        self.peak_temporary_bytes = 0
        self.last_budget: dict[str, Any] = {}
        self._timer.reset()

    def prefill_archive(self, text: str, segment_id: str | None = None) -> dict[str, Any]:
        self._segment_id = segment_id
        if self.dry_run:
            dense_size = max(8, len(text.encode("utf-8")) * 16)
            self._dense_reference = bytearray(dense_size)
            self._latent_cache = bytearray(int(dense_size * self.budget_ratio))
        else:
            if self._prefill is None:
                raise RuntimeError("prefill callback is required")
            with self._timer.measure("compress"):
                result = self._prefill(text)
            if isinstance(result, Mapping):
                self._latent_cache = result.get("latent_cache")
                self._dense_reference = result.get("dense_reference", result.get("dense_kv"))
            else:
                self._latent_cache = result
            if self._latent_cache is None:
                raise ValueError("Palu prefill must expose the true latent cache")
        self.last_budget = budget_report(self.budget_ratio, self._dense_reference, self._latent_cache)
        if self.strict_budget and not self.last_budget["within_budget"]:
            raise MemoryError(f"Palu latent cache exceeds byte budget: {self.last_budget}")
        return dict(self.last_budget)

    def query(self, text: str, **generate_kwargs: Any) -> Any:
        if self._latent_cache is None:
            raise RuntimeError("prefill_archive must be called before query")
        if self.dry_run:
            return "314159"
        cache_for_generation = self._latent_cache
        if self._reconstruct is not None:
            with self._timer.measure("recover"):
                self._dense_workspace = self._reconstruct(self._latent_cache)
            self.peak_temporary_bytes = max(
                self.peak_temporary_bytes, recursive_nbytes(self._dense_workspace)
            )
            cache_for_generation = self._dense_workspace
        if self._generate is None:
            raise RuntimeError("generate callback is required")
        try:
            with self._timer.measure("generate"):
                return self._generate(text, cache_for_generation, **generate_kwargs)
        finally:
            self._dense_workspace = None

    def persistent_state(self) -> Mapping[str, Any]:
        return {"latent_cache": self._latent_cache}

    def static_extra_state(self) -> Mapping[str, Any]:
        """Projection parameters added by Palu, excluding base-model weights."""
        out: dict[str, Any] = {}
        if self.model is None:
            return out
        for module_name, module in getattr(self.model, "named_modules", lambda: [])():
            if module.__class__.__name__ not in {"HeadwiseLowRankModule", "LlamaPaluAttention"}:
                continue
            for name in ("VT", "U_list", "B"):
                value = getattr(module, name, None)
                if value is not None:
                    out[f"{module_name}.{name}"] = value
        return out

    def temporary_state(self) -> Mapping[str, Any]:
        return {"dense_reconstruction_workspace": self._dense_workspace}


def load_palu_attention() -> tuple[type, type]:
    """Delayed official import; importing ``biocache_adapt`` remains CPU-only."""
    from kernel.palu_attention import HeadwiseLowRankModule, LlamaPaluAttention

    return HeadwiseLowRankModule, LlamaPaluAttention


def wrap_model_true_latent(model: Any, config: Any, *, no_fusion: bool = False) -> int:
    """Replace compatible Llama attention modules with Palu's latent-cache path."""
    _, palu_attention = load_palu_attention()
    replaced = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == "LlamaAttention":
                setattr(parent, name, palu_attention.from_attention(child, config, no_fusion=no_fusion))
                replaced += 1
    return replaced
