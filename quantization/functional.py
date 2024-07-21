import warnings

from torch import Tensor, abs, bool, clamp, sum, argmin, gather, bernoulli, empty, tensor
from torch.autograd import Function


def define_quantization_ranges(num_bits, signed):
    if num_bits < 1:
        raise ValueError("Number of bits must be a positive integer.")
    if num_bits == 1:
        return -1, 1
    if signed:
        qmin = -(2.0 ** (num_bits - 1))
        qmax = 2.0 ** (num_bits - 1) - 1
    else:
        qmin = 0.0
        qmax = 2.0 ** num_bits - 1.0
    return qmin, qmax


def interp1d_linear(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    # Ensure 'x' is 2D for broadcasting.
    x_ = x.unsqueeze(-1)

    # Ensure 'xp' and 'fp' are 2D for broadcasting.
    xp = xp.unsqueeze(0)
    fp = fp.unsqueeze(0)

    # TODO: The slope and intercept calculations can be cached since the `xp` and `fp` are `lut_qx`, and `lut_qy`.
    # Compute the slopes of the segments.
    numerator = fp[:, 1:] - fp[:, :-1]
    denominator = xp[:, 1:] - xp[:, :-1]
    denominator[denominator == 0] = 1e-6
    slopes = numerator / denominator
    # Compute the intercepts of the segments.
    intercepts = fp[:, :-1] - slopes * xp[:, :-1]

    # Determine which segment each x value falls within.
    indices = sum(x_ >= xp, axis=-1) - 1
    indices = clamp(indices, 0, slopes.shape[-1] - 1)  # Avoid index out of bounds

    # Calculate interpolated values.
    out = slopes[0, indices] * x + intercepts[0, indices]
    return out


def interp1d_nearest_neighbor(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    # Ensure 'x' is 2D for broadcasting.
    x_ = x.unsqueeze(-1)
    # Calculate differences and find nearest indices.
    differences = abs(x_ - xp)
    indices = argmin(differences, dim=1).unsqueeze(-1)
    # Since 'fp' might not be prepared for batched operations as typically expected:
    # We ensure 'indices' has proper shape considering 'fp's real batch-like structure.
    # Ensure fp is broadcast over the same batch dimensions as 'x'.
    expanded_fp = fp.expand(x.size(0), -1)
    # Correct gathering operation
    out = gather(expanded_fp, 1, indices)
    return out.squeeze(-1)


class FakeQuantize(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        q_x = zero_point + x / scale
        q_x = q_x.clamp(qmin, qmax).round()
        dq_x = scale * (q_x - zero_point)

        ctx.save_for_backward(x, scale, zero_point)
        ctx.qmin, ctx.qmax = qmin, qmax
        return dq_x

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        if x.grad_fn.__class__.__name__ == ctx.__class__.__name__:
            raise RuntimeError("Double quantization detected."
                               "Avoid calling FakeQuantize.apply(FakeQuantize.apply(...))")
        q_x = zero_point + x / (scale + 1e-16) # For numerical stability add epsilon to avoid division by small numbers

        grad_x = grad_output * ((q_x >= qmin) & (q_x <= qmax)).float()  # STE for input
        grad_scale = (grad_output * (q_x - zero_point)).mean(dim=0)  # Gradient for scale
        grad_zero_point = grad_output.mean(dim=0)  # Gradient for zero_point

        return grad_x, grad_scale, grad_zero_point, None, None

def torch_quantize(x, scale, zero_point, qmin, qmax):
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    return q_x


def torch_dequantize(q_x, scale, zero_point):
    return scale * (q_x - zero_point)


def torch_fake_quantize(x, *q_args):
    return FakeQuantize.apply(x, *q_args)


def sample_tensor(probability, x, sample_cutoff=1000):
    # Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, 2020

    if x.numel() < sample_cutoff:
        # warnings.warn(f"Sample size {x.numel()} is less than the cutoff value {sample_cutoff}. "
        #               f"Returning the input tensor as is.")
        return x

    cutoff_probability = sample_cutoff / x.numel()
    if cutoff_probability > probability:
        probability = cutoff_probability

    x = x.view(-1)
    probs = tensor([probability], device=x.device).expand_as(x)
    out = empty(probs.shape, dtype=bool, device=probs.device)
    mask = bernoulli(probs, out=out)
    return x[mask]
