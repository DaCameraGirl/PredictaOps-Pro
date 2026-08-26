#!/usr/bin/env python
"""Run leave-one-entire-IMS-run-out RUL validation on prepared feature caches.

This command never overwrites the Test 2 model served by the dashboard. It writes
validation evidence only.

Usage:
    python scripts/validate_cross_run.py
    python scripts/validate_cross_run.py --runs ims_test1 ims_test2 ims_test3
"""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bearing_data import ALL_RUN_SPECS  # noqa: E402
from cross_run_validation import default_output_path, validate_prepared_runs  # noqa: E402

DEFAULT_RUNS = ("ims_test1", "ims_test2", "ims_test3")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=sorted(ALL_RUN_SPECS),
        default=list(DEFAULT_RUNS),
        help="prepared IMS runs to include in leave-one-run-out validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="validation metrics JSON path; does not contain or replace a served model",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_prepared_runs(args.runs)
    result["generated_at"] = datetime.now(UTC).isoformat()
    result["runs"] = list(args.runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    overall = result["overall"]
    print(f"validation: {result['validation_method']}")
    print(f"runs: {', '.join(args.runs)}")
    print(f"overall MAE: {overall['mae_hours']:.2f} h")
    print(f"overall RMSE: {overall['rmse_hours']:.2f} h")
    print(f"dangerous over-predictions: {overall['dangerous_overprediction_pct']:.1f}%")
    for fold in result["folds"]:
        print(
            f"held out {fold['held_out_run']}: MAE {fold['mae_hours']:.2f} h, "
            f"RMSE {fold['rmse_hours']:.2f} h, "
            f"late-life MAE {fold['late_life_mae_hours']:.2f} h"
            if fold["late_life_mae_hours"] is not None
            else (
                f"held out {fold['held_out_run']}: MAE {fold['mae_hours']:.2f} h, "
                f"RMSE {fold['rmse_hours']:.2f} h"
            )
        )
    print(f"wrote validation evidence to {args.output}")


if __name__ == "__main__":
    main()
