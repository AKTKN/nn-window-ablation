import numpy as np
import pytest

from nn_window_ablation import data, ddp_launch
from nn_window_ablation.config import ConfigError, resolve_config
from nn_window_ablation.execution import check_execution, generate_workflow
from nn_window_ablation.streams import ChunkCursor, TrajectoryStream
from tests.p4_checks import setup_plan


@pytest.fixture
def config(compose_settings, tmp_path):
    def resolve(*extra):
        return resolve_config(compose_settings(
            "runtime=test_cpu", "data.split=train", "data.id=stored", "data.train_source=stored_shards",
            "training.execution=ddp", "runtime.processes=2", "training.batch_size=6", "training.microbatch_size=2",
            "training.max_steps=1", "training.learning_rate=0.001", "training.optimizer=adam", "training.scheduler=none",
            *extra,
        ), invocation_dir=tmp_path)
    return resolve


@pytest.mark.parametrize(("override", "message"), [
    ("training.batch_size=5", "divisible"),
    ("training.microbatch_size=4", "on one rank"),
    ("runtime.interop_threads=2", "one inter-op"),
    ("data.train_source=generated_chunks", "requires stored_shards"),
    ("training.execution=single", "only DDP train"),
    ("training.ddp_timeout_seconds=0", "positive"),
    ("data.num_workers=1", "zero loader"),
    ("data.sampling_processes=2", "inline sampling"),
])
def test_ddp_invalid_budgets_fail_before_output(config, tmp_path, override, message):
    extra = ("data.training_chunk_shots=2",) if override == "data.train_source=generated_chunks" else ()
    with pytest.raises(ConfigError, match=message):
        check_execution("train", config(override, *extra))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "null", "true"])
def test_log_interval_requires_a_positive_integer(config, value):
    with pytest.raises(ConfigError, match="training.log_every_updates"):
        config("training.force_large_log=true", f"training.log_every_updates={value}")


def test_update_log_preflight(config, tmp_path):
    from nn_window_ablation.workflows import check_workflow

    single = ("training.execution=single", "runtime.processes=1")
    for layout, allowed, rejected, rows in [(single, 100099, 100100, 1001), ((), 50099, 50100, 1002)]:
        for command in ("validate", "train"):
            check_workflow(command, config(*layout, f"training.max_steps={allowed}"))
            with pytest.raises(ConfigError, match=f"{rows} update log rows"):
                check_workflow(command, config(*layout, f"training.max_steps={rejected}"))
            check_workflow(command, config(*layout, f"training.max_steps={rejected}", "training.force_large_log=true"))
    with pytest.raises(ConfigError, match="training.force_large_log"):
        config("training.force_large_log=1")
    assert not list(tmp_path.iterdir())


def test_ddp_placement_uses_affinity_and_visible_cpu_quota(config, monkeypatch):
    request = config("runtime.allocated_cpus=4", "runtime.intraop_threads=2")
    monkeypatch.setattr(ddp_launch.os, "sched_getaffinity", lambda _: {4, 5, 8})
    with pytest.raises(ConfigError, match="effective process affinity"):
        ddp_launch.training_placement(request)
    monkeypatch.setattr(ddp_launch.os, "sched_getaffinity", lambda _: {4, 5, 8, 9})
    monkeypatch.setattr(ddp_launch, "_cpu_quota", lambda: 3.5)
    with pytest.raises(ConfigError, match="visible cgroup CPU quota"):
        ddp_launch.training_placement(request)
    monkeypatch.setattr(ddp_launch, "_cpu_quota", lambda: None)
    assert ddp_launch.training_placement(request) == ((4, 5), (8, 9))


def test_generation_keeps_single_process_guard(config):
    with pytest.raises(ConfigError, match="only DDP train"):
        generate_workflow(config("data.shots=6", "data.shard_shots=3"))


def test_excessive_inherited_threads_fail_before_spawn(config, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    with pytest.raises(ConfigError, match="inherited OPENBLAS_NUM_THREADS"):
        ddp_launch.launch_training(config())


def test_preview_shard_boundary_and_finite_source_without_opening_payloads(tmp_path, monkeypatch):
    bundle, plan = setup_plan(total=7, chunk=3)
    root = tmp_path / "train"
    data.write_dataset(root, plan, bundle, provenance={})
    mapped = data.open_dataset(root)
    wanted = []
    for shard, rows in enumerate((3, 3, 1)):
        rng = np.random.Generator(np.random.PCG64(plan.namespace.seed("trajectory_order", shard)))
        wanted.extend((shard, int(row)) for row in rng.permutation(rows))
    with TrajectoryStream(plan, bundle, mapped) as stream:
        def forbidden(*_args, **_kwargs):
            pytest.fail("planned identities read data")
        with monkeypatch.context() as patch:
            patch.setattr(data.MappedDataset, "open_shard", forbidden)
            assert [(row.shard, row.row) for row in stream.planned_identities(7)] == wanted
            assert stream.position == 0
        stream.seek(ChunkCursor(0, 2))
        local = stream.take(4)
        assert [(item.shard, item.row) for item in local.identities] == wanted[2:6]
        assert stream.position == 6
        with pytest.raises(ValueError, match="complete trajectory batch"):
            stream.planned_identities(2)
        assert stream.position == 6
        assert len(stream.take(1).identities) == 1
        assert stream.position == 7 and not mapped._active
