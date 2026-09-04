from PIL import Image

from gait_phase.hashing import perceptual_hash
from gait_phase.models import build_frame_cnn, build_temporal_tcn, build_thermal_gait_phasenet, hog_descriptor
from gait_phase.training import dense_temporal_loss


def test_hog_descriptor_has_stable_shape():
    descriptor = hog_descriptor(Image.new("L", (32, 32), 0))
    assert descriptor.shape == (8 * 8 * 9,)


def test_perceptual_hash_has_stable_shape(tmp_path):
    path = tmp_path / "image.png"
    Image.new("L", (32, 32), 0).save(path)
    assert len(perceptual_hash(path)) == 16


def test_neural_models_have_exactly_four_outputs():
    import torch

    frame_model = build_frame_cnn(pretrained=False).eval()
    with torch.no_grad():
        assert frame_model(torch.zeros(1, 3, 64, 64)).shape == (1, 4)
    temporal_model = build_temporal_tcn(pretrained=False).eval()
    with torch.no_grad():
        assert temporal_model(torch.zeros(1, 3, 3, 64, 64)).shape == (1, 4)
    thermal_model = build_thermal_gait_phasenet(
        pretrained=False,
        feature_dim=32,
        dilations=(1, 2),
        attention_heads=4,
    ).eval()
    with torch.no_grad():
        output = thermal_model(torch.zeros(1, 9, 1, 64, 44))
    assert output["phase_logits"].shape == (1, 9, 4)
    assert output["boundary_logits"].shape == (1, 9)


def test_dense_temporal_loss_is_finite_and_differentiable():
    import torch

    phase_logits = torch.randn(2, 5, 4, requires_grad=True)
    boundary_logits = torch.randn(2, 5, requires_grad=True)
    targets = torch.tensor([[0, 0, 1, 1, 2], [2, 3, 3, 0, 0]])
    boundaries = torch.tensor([[0, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    losses = dense_temporal_loss(
        {"phase_logits": phase_logits, "boundary_logits": boundary_logits},
        targets,
        boundaries,
        mask,
        torch.ones(4),
        torch.tensor(2.0),
        0.2,
        0.1,
        0.05,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert phase_logits.grad is not None
    assert boundary_logits.grad is not None
