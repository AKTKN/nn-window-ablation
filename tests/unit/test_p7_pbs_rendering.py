import json
import shlex
import subprocess

import pytest

from nn_window_ablation.config import ConfigError
from nn_window_ablation.pbs import JobSpec, render_script


@pytest.fixture
def job():
    return {
        "stage": "train", "job_name": "pilot-base", "run_name": "base.01",
        "repository": "/shared/fixed checkout", "source_revision": "a" * 40,
        "conda_executable": "/shared/conda tools/bin/conda", "conda_prefix": "/shared/envs/test prefix",
        "data_root": "/shared/run data", "run_root": "/shared/job runs", "receipt_dir": "/shared/job logs/train 1",
        "resources": {"ncpus": 8, "memory": "4gb", "walltime": "00:10:00"},
        "threads": 1,
        "overrides": [
            "runtime=test_cpu", "training.max_steps=1", "training.batch_size=1", "training.learning_rate=0.001",
            "training.optimizer=adam", "training.scheduler=none", "data.training_chunk_shots=2",
        ],
    }


def test_p7_shell_directives_and_argv_preserve_spaces_without_transport(job, tmp_path, monkeypatch):
    marker = tmp_path / "called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for utility in ("qsub", "ssh", "scp"):
        executable = bin_dir / utility
        executable.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 99\n")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    lines = render_script(JobSpec.from_record(job)).splitlines()
    assert lines[:5] == [
        "#!/bin/bash", "#PBS -N pilot-base", "#PBS -l select=1:ncpus=8:mem=4gb",
        "#PBS -l walltime=00:10:00", "#PBS -j oe",
    ]
    assert not any(line.startswith(("#PBS -q ", "#PBS -A ")) for line in lines)
    call = shlex.split(lines[-1])
    assert call[:-1] == [
        "exec", "/shared/conda tools/bin/conda", "run", "--no-capture-output", "-p",
        "/shared/envs/test prefix", "python", "-m", "nn_window_ablation.pbs",
    ]
    assert json.loads(call[-1])["data_root"] == "/shared/run data"
    assert "export OMP_NUM_THREADS=1" in lines
    path = tmp_path / "generated.pbs"
    path.write_text("\n".join(lines) + "\n")
    subprocess.run(["/bin/bash", "-n", path], check=True)
    assert not marker.exists()


@pytest.mark.parametrize(("mutation", "message"), [
    ({"conda_prefix": ""}, "conda_prefix"),
    ({"run_name": "../trial"}, "portable name"),
    ({"threads": 9}, "within resources.ncpus"),
    ({"resources": {"ncpus": 96, "memory": "350gb", "walltime": "01:00:00"}}, "at most eight"),
    ({"job_name": "bad\njob"}, "PBS job_name"),
])
def test_p7_rejects_bad_job_specs(job, mutation, message):
    with pytest.raises(ConfigError, match=message):
        render_script(JobSpec.from_record({**job, **mutation}))


def test_p7_rejects_missing_budgets_and_excess_worker_allocation(job):
    for tail, message in (
        (["training.max_steps=null"], "training.max_steps"),
        (["data.num_workers=8"], "exceeding runtime.allocated_cpus"),
        (["checkpoint.resume_from=/shared/job runs/base.01/checkpoint.pt"], "PBS resume requires"),
    ):
        with pytest.raises(ConfigError, match=message):
            render_script(JobSpec.from_record({**job, "overrides": job["overrides"] + tail}))


@pytest.mark.parametrize(("stage", "extra"), [
    ("generate", ["data.id=held", "data.split=test", "data.shots=3", "data.shard_shots=2"]),
    ("evaluate", ['checkpoint.path="/shared/job runs/base.01/checkpoint.pt"', "data.id=held", "evaluation=paired"]),
    ("profile", []), ("probe", []),
])
def test_p7_stage_plans_and_optional_site_directives(job, stage, extra):
    job["stage"] = stage
    job["resources"].update(queue="batch@server", account="research-001")
    job["overrides"] = ["runtime=test_cpu", *extra]
    script = render_script(JobSpec.from_record(job))
    assert "#PBS -q batch@server\n#PBS -A research-001\n" in script
    assert json.loads(shlex.split(script.splitlines()[-1])[-1])["stage"] == stage


def test_p7_resume_is_rendered_with_original_run_name(job):
    job["stage"] = "resume"
    job["overrides"].append('checkpoint.resume_from="/shared/job runs/base.01/checkpoint.pt"')
    assert render_script(JobSpec.from_record(job))
