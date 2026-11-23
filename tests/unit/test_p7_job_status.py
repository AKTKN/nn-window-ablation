import json
from types import SimpleNamespace

import pytest

from nn_window_ablation.artifacts import ArtifactError, file_integrity
from nn_window_ablation.pbs import finish_job, scheduler_status, start_job


def test_p7_job_receipts_require_exit_and_verified_artifact(tmp_path):
    planned = {"schema_version": 1, "status": "planned", "plan": {"resources": {"ncpus": 8}}}
    observed = {"pbs_job_id": "221.local", "cpu_affinity": [1, 3]}
    running = start_job(planned, observed)
    assert planned["status"] == "planned" and running["status"] == "running"
    assert running["plan"]["resources"] != running["observed"]
    path = tmp_path / "finished"
    path.write_text("verified artifact")
    artifact = {"path": str(path), **file_integrity(path)}
    assert finish_job(running, exit_code=None, artifact=artifact)["status"] == "incomplete"
    assert finish_job(running, exit_code=0)["status"] == "incomplete"
    assert finish_job(running, exit_code=0, artifact=artifact)["status"] == "complete"
    assert finish_job(running, exit_code=4, artifact=artifact)["status"] == "failed"
    assert finish_job(running, exit_code=0, error="artifact did not verify")["status"] == "failed"
    path.unlink()
    assert finish_job(running, exit_code=0, artifact=artifact)["status"] == "incomplete"
    with pytest.raises(ValueError, match="only a running job"):
        finish_job(planned, exit_code=0, artifact=artifact)


def test_p7_injected_qstat_argv_and_state_are_distinct_from_artifact_proof(tmp_path):
    job_id = "221.site"
    payload = {"job_state": "Q", "Resource_List": {"ncpus": 8}, "resources_used": {"ncpus": 2}}
    arguments = []

    def transport(argv, **kwargs):
        arguments.append((argv, kwargs))
        return SimpleNamespace(stdout=json.dumps({"Jobs": {job_id: payload}}))

    assert scheduler_status(job_id, "/opt/pbs/bin/qstat", transport=transport)["status"] == "planned"
    assert arguments[0] == (
        ["/opt/pbs/bin/qstat", "-x", "-f", "-F", "json", job_id],
        {"capture_output": True, "text": True, "check": True},
    )
    payload["job_state"] = "R"
    assert scheduler_status(job_id, "qstat", transport=transport)["status"] == "running"
    payload["job_state"] = "F"
    assert scheduler_status(job_id, "qstat", transport=transport)["status"] == "incomplete"
    payload["Exit_status"] = 0
    assert scheduler_status(job_id, "qstat", transport=transport)["status"] == "incomplete"
    artifact = tmp_path / "manifest.json"
    artifact.write_text("verified artifact")
    receipt = {
        "status": "complete", "exit_code": 0, "observed": {"pbs_job_id": job_id},
        "artifact": {"path": str(artifact), **file_integrity(artifact)},
    }
    observed = scheduler_status(job_id, "qstat", receipt=receipt, transport=transport)
    assert observed["status"] == "complete"
    assert observed["reported_request"] == {"ncpus": 8} and observed["resources_used"] == {"ncpus": 2}
    artifact.write_text("changed")
    assert scheduler_status(job_id, "qstat", receipt=receipt, transport=transport)["status"] == "incomplete"
    payload["Exit_status"] = 5
    assert scheduler_status(job_id, "qstat", receipt=receipt, transport=transport)["status"] == "failed"
    with pytest.raises(ArtifactError, match="requested job"):
        scheduler_status("222.site", "qstat", transport=transport)
