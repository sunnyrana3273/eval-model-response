"""Forward-hook activation capture for Hugging Face causal LMs.

Activations are captured on a forward pass over the *prompt only* (everything the
agent has seen up to the decision point), before the action is generated. This
guarantees a recorded representation can never contain information from the
action it is used to predict. ``tests/test_activations.py`` checks this property.
"""

from __future__ import annotations

from typing import Any

import numpy as np

STRATEGIES = ("last", "mean", "mean_tail")


def find_decoder_layers(model: Any):
    """Locate the list of transformer blocks for common architectures."""
    import torch.nn as nn

    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers",
                 "transformer.layers", "language_model.model.layers", "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    candidates = [m for m in model.modules() if isinstance(m, nn.ModuleList)]
    if not candidates:
        raise ValueError("could not locate decoder layers")
    return max(candidates, key=len)


def resolve_layers(spec: Any, n_layers: int) -> list[int]:
    """'all' | int (every k-th) | list of indices (negative allowed)."""
    if spec in (None, "all"):
        return list(range(n_layers))
    if isinstance(spec, int):
        return list(range(0, n_layers, spec))
    out = sorted({i % n_layers for i in spec})
    return out


def pool(hidden: "np.ndarray", strategy: str, tail: int = 16) -> np.ndarray:
    """Pool a [seq, d] hidden-state matrix into a [d] vector."""
    if strategy == "last":
        return hidden[-1]
    if strategy == "mean":
        return hidden.mean(axis=0)
    if strategy == "mean_tail":
        return hidden[-tail:].mean(axis=0)
    raise ValueError(f"unknown token strategy {strategy!r}; options: {STRATEGIES}")


class ActivationRecorder:
    """Registers hooks on selected blocks, pools their outputs, and keeps only the pooled vectors.

    Usage::

        rec = ActivationRecorder(model, layers=[4, 8], strategy="last")
        acts = rec.capture(input_ids)   # np.ndarray [len(layers), hidden_dim]
    """

    def __init__(self, model: Any, layers: Any = "all", strategy: str = "last", tail: int = 16):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown token strategy {strategy!r}")
        self.model = model
        self.blocks = find_decoder_layers(model)
        self.layer_indices = resolve_layers(layers, len(self.blocks))
        self.strategy = strategy
        self.tail = tail
        self._buffer: dict[int, Any] = {}

    def _hook(self, idx: int):
        def fn(_module, _inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            # [batch=1, seq, d] -> pooled [d]; pool immediately so full tensors are not retained.
            h = hs[0].detach().float()
            if self.strategy == "last":
                v = h[-1]
            elif self.strategy == "mean":
                v = h.mean(0)
            else:
                v = h[-self.tail:].mean(0)
            self._buffer[idx] = v.cpu().numpy()

        return fn

    def capture(self, input_ids: Any) -> np.ndarray:
        import torch

        self._buffer.clear()
        handles = [self.blocks[i].register_forward_hook(self._hook(i)) for i in self.layer_indices]
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return np.stack([self._buffer[i] for i in self.layer_indices]).astype(np.float32)

    @property
    def hidden_dim(self) -> int:
        cfg = self.model.config
        return int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd"))

    def describe(self) -> dict:
        return {
            "architecture": type(self.model).__name__,
            "n_model_layers": len(self.blocks),
            "layers": self.layer_indices,
            "hidden_dim": self.hidden_dim,
            "strategy": self.strategy,
            "tail": self.tail if self.strategy == "mean_tail" else None,
            "position": "prompt forward pass before generation (residual stream output of each block)",
        }
