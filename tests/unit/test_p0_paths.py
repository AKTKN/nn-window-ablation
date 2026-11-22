import pytest

from nn_window_ablation.config import ConfigError, new_run_name, resolve_config
from nn_window_ablation.workflows import reserve_run_directory


def test_paths_resolve_against_invocation_dir(compose_settings, monkeypatch, tmp_path):
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("NNWA_DATA_ROOT", "../combined/inputs")
    monkeypatch.setenv("NNWA_RUN_ROOT", "outputs")
    config = resolve_config(
        compose_settings("checkpoint.path=../weights/model.pt"), invocation_dir=invocation
    )

    assert config.paths.data_root == tmp_path / "combined" / "inputs"
    assert config.paths.run_root == invocation / "outputs"
    assert config.checkpoint.path == tmp_path / "weights" / "model.pt"
    assert list(invocation.iterdir()) == []
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".", "..", "foo/bar", "quoted space", "bad:name", "trailing."])
def test_unsafe_run_names_are_rejected(compose_settings, tmp_path, name):
    with pytest.raises(ConfigError, match="run_name must be a portable name"):
        resolve_config(compose_settings(f'run_name="{name}"'), invocation_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("root", ["ordinary-file", "ordinary-file/subdirectory"])
def test_run_root_must_be_a_directory(compose_settings, tmp_path, root):
    (tmp_path / "ordinary-file").touch()
    with pytest.raises(ConfigError, match="expected a directory"):
        resolve_config(compose_settings(f"paths.run_root={root}"), invocation_dir=tmp_path)


def test_run_already_exists_and_contents_are_preserved(compose_settings, tmp_path):
    config = resolve_config(compose_settings("paths.run_root=runs"), invocation_dir=tmp_path)
    result = reserve_run_directory(config)
    assert result == tmp_path / "runs" / "p0-unit"
    (result / "existing.txt").write_text("owned by the first invocation")

    with pytest.raises(ConfigError, match="run directory already exists"):
        reserve_run_directory(config)
    assert (result / "existing.txt").read_text() == "owned by the first invocation"


def test_generated_run_names_reserve_distinct_directories(compose_settings, tmp_path):
    directories = [
        reserve_run_directory(resolve_config(
            compose_settings(f"run_name={new_run_name()}", "paths.run_root=runs"), invocation_dir=tmp_path
        ))
        for _ in range(2)
    ]
    assert directories[0] != directories[1]
    assert all(directory.parent == tmp_path / "runs" and directory.is_dir() for directory in directories)
