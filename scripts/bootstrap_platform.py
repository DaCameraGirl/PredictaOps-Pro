"""Register the NASA/IMS runs as Platform Core entities.

Run after migrations:
    alembic upgrade head
    python scripts/bootstrap_platform.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from platform_core.database import session_scope  # noqa: E402
from platform_core.services import bootstrap_ims_registry  # noqa: E402


def main() -> None:
    with session_scope() as session:
        summary = bootstrap_ims_registry(session)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
