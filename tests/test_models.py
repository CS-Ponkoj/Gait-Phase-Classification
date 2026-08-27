from PIL import Image

from gait_phase.hashing import perceptual_hash
from gait_phase.models import build_frame_cnn, build_temporal_tcn, hog_descriptor


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
