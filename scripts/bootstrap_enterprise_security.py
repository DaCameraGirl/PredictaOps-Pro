"""Operator CLI for initial enterprise security provisioning.

This script intentionally accepts identity metadata only. It does not collect or
persist passwords, bearer tokens, client secrets, or connector credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enterprise_security.contracts import BootstrapSecurityRequest
from enterprise_security.service import SecurityService
from platform_core.database import session_scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap enterprise security for an organization.")
    parser.add_argument("--organization-slug", required=True)
    parser.add_argument("--organization-name", required=True)
    parser.add_argument("--owner-email", required=True)
    parser.add_argument("--owner-full-name")
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--audience", required=True)
    parser.add_argument("--jwks-uri", required=True)
    parser.add_argument("--idp-name", default="primary-oidc")
    parser.add_argument("--allowed-algorithm", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = BootstrapSecurityRequest(
        organization_slug=args.organization_slug,
        organization_name=args.organization_name,
        owner_email=args.owner_email,
        owner_full_name=args.owner_full_name,
        issuer=args.issuer,
        subject=args.subject,
        audience=args.audience,
        jwks_uri=args.jwks_uri,
        idp_name=args.idp_name,
        allowed_algorithms=args.allowed_algorithm or ["RS256"],
    )
    with session_scope() as session:
        result = SecurityService(session).bootstrap_initial_owner(request)
    print(
        "enterprise security bootstrap complete: "
        f"organization_id={result['organization_id']} user_id={result['user_id']} "
        f"identity_provider_id={result['identity_provider_id']} user_identity_id={result['user_identity_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
