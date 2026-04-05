"""
Patches for vLLM Factory.

Note: The legacy ``pooling_extra_kwargs`` patch for vLLM 0.15.x has been
removed.  vLLM >= 0.19 supports ``extra_kwargs`` and custom IO processors
natively; no disk-patching is required.
"""
