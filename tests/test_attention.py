import torch

from ego_hand_wm.models.attention import (
    CachedContextAttentionBlock,
    HeadRMSNorm,
    SharedContextKVProjector,
)


def test_head_rms_norm_is_finite_and_unit_scale() -> None:
    normalization = HeadRMSNorm(8, eps=1e-6)
    value = torch.randn(2, 4, 5, 8) * 1e4
    normalized = normalization(value)
    assert torch.isfinite(normalized).all()
    rms = normalized.float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)


def test_cached_and_expert_keys_are_qk_normalized_with_finite_gradients() -> None:
    torch.manual_seed(3)
    projector = SharedContextKVProjector(32, 4, 1, qk_norm=True)
    block = CachedContextAttentionBlock(32, 4, 2.0, 0.0, qk_norm=True)
    context = torch.randn(2, 7, 32) * 100.0
    context_valid = torch.ones(2, 7, dtype=torch.bool)
    cache = projector(context, context_valid)
    context_key_rms = cache.keys[0].float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(
        context_key_rms,
        torch.ones_like(context_key_rms),
        atol=1e-4,
        rtol=1e-4,
    )

    tokens = (torch.randn(2, 9, 32) * 100.0).requires_grad_()
    conditioning = torch.randn(2, 32)
    valid = torch.ones(2, 9, dtype=torch.bool)
    output = block(
        tokens,
        conditioning,
        valid,
        cache.keys[0],
        cache.values[0],
        cache.valid,
    )
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
