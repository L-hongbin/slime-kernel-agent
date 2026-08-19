import torch

from custom_kernels.deepseek_v4.mhc import kernel as mhc_kernel


def test_compact_last_dim_packs_size_one_stride():
    raw_p = torch.arange(mhc_kernel.MIXP, dtype=torch.float32).reshape(1, mhc_kernel.MIXP)
    sliced = raw_p[:, : mhc_kernel.MIX]

    assert sliced.is_contiguous()
    assert sliced.stride() == (mhc_kernel.MIXP, 1)

    packed = mhc_kernel._compact_last_dim(raw_p, mhc_kernel.MIX)

    assert packed.shape == (1, mhc_kernel.MIX)
    assert packed.stride() == (mhc_kernel.MIX, 1)
    assert packed.data_ptr() != raw_p.data_ptr()
    torch.testing.assert_close(packed, sliced)
