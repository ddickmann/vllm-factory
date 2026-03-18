# SPDX-License-Identifier: Apache-2.0
# Fused LayerNorm for ModernBERT
# Single-pass mean/variance/normalize with optional affine transformation

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['hidden_size'],
)
@triton.jit
def layernorm_kernel(
    X_ptr,           # Input tensor
    Y_ptr,           # Output tensor
    W_ptr,           # Weight (gamma)
    B_ptr,           # Bias (beta), can be None
    Mean_ptr,        # Mean output (for backward pass)
    Rstd_ptr,        # Reciprocal std output (for backward pass)
    stride_x_row,    # Stride for X rows
    stride_y_row,    # Stride for Y rows
    hidden_size,     # Hidden dimension
    eps,             # Epsilon for numerical stability
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm kernel.

    Computes: Y = (X - mean(X)) / sqrt(var(X) + eps) * W + B

    Args:
        X: Input tensor [batch * seq_len, hidden_size]
        Y: Output tensor [batch * seq_len, hidden_size]
        W: Weight (gamma) [hidden_size]
        B: Bias (beta) [hidden_size] or None
        Mean: Mean values [batch * seq_len] (for backward)
        Rstd: Reciprocal std [batch * seq_len] (for backward)
    """
    # Get row index
    row_idx = tl.program_id(0)

    # Pointers to this row
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    # Compute mean in single pass
    mean_acc = 0.0

    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean_acc += tl.sum(x)

    mean = mean_acc / hidden_size

    # Compute variance in single pass
    var_acc = 0.0

    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var_acc += tl.sum(diff * diff)

    var = var_acc / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store mean and rstd for backward pass
    tl.store(Mean_ptr + row_idx, mean)
    tl.store(Rstd_ptr + row_idx, rstd)

    # Normalize and apply affine transformation
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size

        # Load input
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Normalize
        x_norm = (x - mean) * rstd

        # Load weight (gamma)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)

        # Apply affine transformation
        y = x_norm * w

        # Add bias if present
        if has_bias:
            b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = y + b

        # Store output
        tl.store(Y_row_ptr + cols, y, mask=mask)


class FusedLayerNorm(torch.nn.Module):
    """
    Fused LayerNorm using Triton for maximum memory bandwidth efficiency.

    This implementation fuses the entire LayerNorm operation into a single kernel,
    minimizing memory reads/writes for optimal performance on high-bandwidth GPUs.

    Args:
        normalized_shape: Input shape (typically hidden_size)
        eps: Small constant for numerical stability
        elementwise_affine: If True, learnable affine parameters
        bias: If True, use bias in affine transformation (requires elementwise_affine=True)
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device)
            )
            if bias:
                self.bias = torch.nn.Parameter(
                    torch.zeros(normalized_shape, dtype=dtype, device=device)
                )
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fused LayerNorm.

        Args:
            x: Input tensor [..., normalized_shape]

        Returns:
            Normalized tensor with same shape as input
        """
        # Get dimensions
        input_shape = x.shape
        hidden_size = self.normalized_shape[0]

        # Reshape to 2D for kernel: [batch * seq_len, hidden_size]
        x_2d = x.view(-1, hidden_size).contiguous()
        num_rows = x_2d.shape[0]

        # Allocate output
        y_2d = torch.empty_like(x_2d)

        # Allocate mean and rstd for backward pass
        mean = torch.empty(num_rows, dtype=torch.float32, device=x.device)
        rstd = torch.empty(num_rows, dtype=torch.float32, device=x.device)

        # Launch kernel
        grid = (num_rows,)

        layernorm_kernel[grid](
            x_2d,
            y_2d,
            self.weight if self.weight is not None else x_2d,  # Dummy if no weight
            self.bias if self.bias is not None else x_2d,  # Dummy if no bias
            mean,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            hidden_size,
            self.eps,
            self.bias is not None,
        )

        # Reshape back to original shape
        y = y_2d.view(input_shape)

        return y

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


def fused_layernorm(
    x: torch.Tensor,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
) -> torch.Tensor:
    """
    Functional interface for fused LayerNorm.

    Args:
        x: Input tensor [..., normalized_shape]
        normalized_shape: Dimensions to normalize over
        weight: Optional affine weight (gamma)
        bias: Optional affine bias (beta)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    # Get dimensions
    if isinstance(normalized_shape, int):
        hidden_size = normalized_shape
    else:
        hidden_size = normalized_shape[0]

    input_shape = x.shape

    # Reshape to 2D
    x_2d = x.view(-1, hidden_size).contiguous()
    num_rows = x_2d.shape[0]

    # Allocate output
    y_2d = torch.empty_like(x_2d)

    # Allocate mean and rstd
    mean = torch.empty(num_rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(num_rows, dtype=torch.float32, device=x.device)

    # Create dummy weight/bias if not provided
    if weight is None:
        weight = torch.ones(hidden_size, dtype=x.dtype, device=x.device)
    if bias is None:
        has_bias = False
        bias = torch.zeros(hidden_size, dtype=x.dtype, device=x.device)  # Dummy
    else:
        has_bias = True

    # Launch kernel
    grid = (num_rows,)

    layernorm_kernel[grid](
        x_2d,
        y_2d,
        weight,
        bias,
        mean,
        rstd,
        x_2d.stride(0),
        y_2d.stride(0),
        hidden_size,
        eps,
        has_bias,
    )

    # Reshape back
    y = y_2d.view(input_shape)

    return y


__all__ = ['FusedLayerNorm', 'fused_layernorm']

