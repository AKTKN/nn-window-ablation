"""Pass YAML lists of Hydra overrides to the installed nnwa command."""

import os
from pathlib import Path
import sys

import yaml


def read_arguments(path):
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    arguments = record.get("arguments")
    if not isinstance(arguments, list) or not all(isinstance(arg, str) and "=" in arg for arg in arguments):
        raise ValueError(f"{path}: arguments must be a YAML list of key=value strings")
    return record, arguments


def command_line(command, path):
    stage, arguments = read_arguments(path)
    if stage.get("command") != command:
        raise ValueError(f"{path}: expected command: {command}")
    common = stage.get("common")
    if not isinstance(common, str) or not common:
        raise ValueError(f"{path}: common must name the shared YAML file")
    _, shared = read_arguments(path.parent / common)
    return ["nnwa", command, *shared, *arguments]


if __name__ == "__main__":
    try:
        argv = command_line(sys.argv[1], Path(sys.argv[2]))
        os.execvp(argv[0], argv)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        sys.exit(f"nnwa PBS: {exc}")
