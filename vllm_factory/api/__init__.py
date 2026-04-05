"""Stable factory-side request / response models.

Everything above the compatibility layer speaks FactoryRequest and
FactoryResponse.  Nothing outside ``vllm_factory.compat`` should import
vLLM protocol classes directly.
"""

from vllm_factory.api.request_models import FactoryRequest
from vllm_factory.api.response_models import FactoryResponse

__all__ = ["FactoryRequest", "FactoryResponse"]
