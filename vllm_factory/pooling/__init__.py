"""Pooler abstraction layer.

Provides the stable ``FactoryPooler`` protocol and ``VllmPoolerAdapter``
so that business pooling logic never depends on vLLM internals.
"""

from vllm_factory.pooling.protocol import (
    FactoryPooler,
    PassthroughPooler,
    PoolerContext,
    split_hidden_states,
)
from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

__all__ = [
    "FactoryPooler",
    "PassthroughPooler",
    "PoolerContext",
    "VllmPoolerAdapter",
    "split_hidden_states",
]
