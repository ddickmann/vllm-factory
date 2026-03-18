"""ColQwen3 Configuration — extends Qwen3VLConfig with ColPali parameters."""
from transformers.models.qwen3_vl import Qwen3VLConfig


class ColQwen3Config(Qwen3VLConfig):
    model_type = "colqwen3"
    def __init__(self, dim: int = 128, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mask_non_image_embeddings = mask_non_image_embeddings
