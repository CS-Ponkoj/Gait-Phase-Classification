from copy import deepcopy

import pytest

from gait_phase.config import load_config
from gait_phase.constants import PHASES


def test_study_configuration_is_locked_to_four_classes():
    config = load_config("configs/study.yaml")
    assert tuple(config["study"]["labels"]) == PHASES
    assert config["data"]["development_folds"] == 5


def test_wrong_label_order_is_rejected(tmp_path):
    source = load_config("configs/study.yaml")
    config = deepcopy(source)
    config["study"]["labels"] = list(reversed(config["study"]["labels"]))
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="locked four-class order"):
        load_config(path)
