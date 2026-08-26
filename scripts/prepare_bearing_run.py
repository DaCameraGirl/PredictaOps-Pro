#!/usr/bin/env python
"""Validate and feature-extract one documented NASA/IMS bearing run.

This command prepares run-specific feature caches only. It deliberately does not
train a model; cross-run/cross-trajectory validation must exist before the newly
registered runs are used for training.

Examples:
    python scripts/prepare_bearing_run.py --run ims_test1
    python scripts/prepare_bearing_run.py --run ims_test3 --raw-dir D:/IMS/3rd_test
"""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bearing_data import (  # noqa: E402
    ALL_RUN_SPECS,
    build_feature_table,
    get_run_spec,
    raw_dataset_checksum,
    validate_raw_dataset,
    validate_run_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted(ALL_RUN_SPECS), required=True)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="override the run's expected raw directory without changing cache paths",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing processed feature cache",
    )
    return parser.parse_args(argv)


def prepare_run(run_id: str, raw_dir: Path | None = None, *, force: bool = False) -> Path:
    run_spec = get_run_spec(run_id)
    spec_errors = validate_run_spec(run_spec)
    if spec_errors:
        raise ValueError("; ".join(spec_errors))

    if run_spec.features_cache.exists() and not force:
        print(f"{run_spec.features_cache} already exists; use --force to rebuild it.")
        return run_spec.features_cache

    source_dir = raw_dir or run_spec.raw_dir
    validation = validate_raw_dataset(source_dir, run_spec)
    table = build_feature_table(source_dir, run_spec)

    run_spec.features_cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_spec.features_cache, index=False)

    metadata = {
        "run_id": run_spec.run_id,
        "dataset": run_spec.dataset_name,
        "source_note": run_spec.source_note,
        "documented_failures": [
            {
                "bearing": failure.bearing,
                "label_endpoint_timestamp": failure.endpoint_timestamp.isoformat(),
                "failure_mode": failure.failure_mode,
            }
            for failure in run_spec.failures
        ],
        "channels": [
            {
                "channel_index": channel.channel_index,
                "bearing": channel.bearing,
                "sensor_id": channel.sensor_id,
            }
            for channel in run_spec.channel_map
        ],
        "generated_at": datetime.now(UTC).isoformat(),
        **validation,
        "raw_sha256": raw_dataset_checksum(source_dir, run_spec),
    }
    run_spec.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    run_spec.metadata_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"prepared {run_spec.run_id}: {len(table)} sensor-feature rows -> "
        f"{run_spec.features_cache}"
    )
    print(f"metadata -> {run_spec.metadata_path}")
    return run_spec.features_cache


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_run(args.run, args.raw_dir, force=args.force)


if __name__ == "__main__":
    main()
