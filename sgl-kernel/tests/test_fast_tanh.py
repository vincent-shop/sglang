import pytest
import torch

from sgl_kernel import gelu_tanh_and_mul


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [1, 64, 512])
@pytest.mark.parametrize("dim", [1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_tanh_numerical(batch_size, seq_len, dim, dtype):
    device = torch.device("cuda")
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device=device)

    out = gelu_tanh_and_mul(x)

    x1 = x[..., :dim]
    x2 = x[..., dim:]
    kAlpha = 0.044715
    kBeta = 0.7978845608028654
    ref = x1 * 0.5 * (1.0 + torch.tanh(kBeta * (x1 + kAlpha * x1 * x1 * x1)))
    ref = ref * x2

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
