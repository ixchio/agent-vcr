"""Create Agent VCR's hosted-control-plane Postgres tables.

Run this once as Railway's pre-deploy command after setting ``DATABASE_URL``
and the required hosted environment variables.
"""

from __future__ import annotations

from agent_vcr.hosted import HostedAccountStore, HostedConfig


def main() -> None:
    """Initialize the idempotent account and webhook-ledger schema."""
    config = HostedConfig.from_environment()
    HostedAccountStore(config.database_url).ensure_schema()
    print("Agent VCR hosted account schema is ready")


if __name__ == "__main__":
    main()
