"""CLI for PBS planning and portable result bundles."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import read_json


def main(command: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"nnwa {command}", description=__doc__)
    if command == "pbs":
        actions = parser.add_subparsers(dest="action", required=True)
        render = actions.add_parser("render", help="validate a JSON job specification and write a PBS script")
        render.add_argument("--spec", type=Path, required=True)
        render.add_argument("--output", type=Path, required=True)
        status = actions.add_parser("status", help="inspect PBS state and optionally verify a job receipt")
        status.add_argument("job_id")
        status.add_argument("--qstat", required=True)
        status.add_argument("--receipt", type=Path)
    elif command == "export":
        for name in ("dataset", "report", "output"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--include-detectors", action="store_true")
        parser.add_argument("--include-checkpoints", action="store_true")
        parser.add_argument("--training-dataset", action="append", type=Path, default=[])
        parser.add_argument("--log", action="append", type=Path, default=[])
    elif command == "extract":
        parser.add_argument("archive", type=Path)
        parser.add_argument("destination", type=Path)
    elif command == "reanalyze":
        parser.add_argument("--bundle", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
    else:
        raise ValueError("unknown P7 command")

    args = parser.parse_args(argv)
    try:
        if command == "pbs":
            from .pbs import JobSpec, scheduler_status, write_script

            if args.action == "render":
                result = write_script(JobSpec.from_record(read_json(args.spec)), args.output)
            else:
                result = scheduler_status(
                    args.job_id, args.qstat, receipt=None if args.receipt is None else read_json(args.receipt),
                )
            print(json.dumps(result, sort_keys=True, indent=2))
        elif command == "extract":
            from .archive import extract_bundle

            print(extract_bundle(args.archive, args.destination))
        elif command == "export":
            from .export import export_bundle

            print(export_bundle(
                args.dataset, args.report, args.output, include_detectors=args.include_detectors,
                include_checkpoints=args.include_checkpoints, training_datasets=tuple(args.training_dataset),
                logs=tuple(args.log),
            ))
        else:
            from .pbs import THREAD_VARS
            from .export import reanalyze_bundle

            for name in THREAD_VARS:
                os.environ[name] = "1"
            print(reanalyze_bundle(args.bundle, args.output))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"nnwa {command}: {exc}", file=sys.stderr)
        return 1
    return 0
