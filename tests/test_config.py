import pytest
import yaml

from validation.config import Config, load_config


def test_nested_sections_become_dataclasses():
    cfg = Config.from_dict({"feature_selection": {"spearman": {"target_min_abs": 0.02}}})
    assert cfg.feature_selection.spearman.target_min_abs == 0.02
    assert cfg.analysis.shap.enabled is True


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        Config.from_dict({"model": {"not_a_key": 1}})
    with pytest.raises(ValueError, match="unknown top-level"):
        Config.from_dict({"nope": {}})


def test_validate_catches_bad_variant_and_task():
    with pytest.raises(ValueError, match="variants"):
        Config.from_dict({"model": {"variants": ["base", "magic"]}}).validate()
    with pytest.raises(ValueError, match="task"):
        Config.from_dict({"model": {"task": "multiclass"}}).validate()


def test_dotted_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"run": {"name": "a"}, "data": {"target": "y"}}))
    cfg = load_config(path, {"run.n_jobs": 3, "feature_selection.spearman.enabled": False})
    assert cfg.run.n_jobs == 3 and cfg.feature_selection.spearman.enabled is False


def test_shipped_configs_load():
    """Every config in configs/ must parse and validate - including templates."""
    from pathlib import Path

    shipped = sorted(Path("configs").glob("*.yaml"))
    assert shipped, "no configs found"
    for path in shipped:
        load_config(path)
