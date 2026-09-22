"""Optional hosted-control-plane integrations for Agent VCR.

The local package deliberately has no dependency on an identity provider or a
payment processor.  This module is enabled only by ``vcr-server --hosted`` and
keeps those concerns at the boundary:

* Supabase Auth access tokens are verified against its public JWKS endpoint.
  No Supabase service-role key or JWT signing secret is ever accepted here.
* Stripe Checkout runs on Stripe-hosted pages; webhook signatures are verified
  before an event can change an account's subscription state.
* A Postgres ledger makes webhook delivery idempotent and survives restarts.

It is intentionally a control-plane foundation, not a replacement for the
local-first recorder or the self-hosted ``CloudStore`` API.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


class HostedConfigurationError(ValueError):
    """Raised when hosted mode has incomplete or unsafe configuration."""


class HostedAuthenticationError(ValueError):
    """Raised when a Supabase bearer token is absent or invalid."""


class HostedDependencyError(RuntimeError):
    """Raised when an optional hosted dependency was not installed."""


class HostedBillingError(RuntimeError):
    """Raised for a billing request that cannot safely be completed."""


@dataclass(frozen=True)
class HostedConfig:
    """Runtime configuration for the optional hosted control plane.

    ``DATABASE_URL`` intentionally follows Railway's conventional variable
    name, so a Railway Postgres reference variable can be used unchanged.
    """

    supabase_url: str
    database_url: str
    app_url: str
    supabase_audience: str = "authenticated"
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_price_id: str | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> HostedConfig:
        """Load hosted settings without ever printing secret values."""
        values = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise HostedConfigurationError(f"{name} is required when hosted mode is enabled")
            return value

        config = cls(
            supabase_url=required("AGENT_VCR_SUPABASE_URL").rstrip("/"),
            database_url=required("DATABASE_URL"),
            app_url=required("AGENT_VCR_APP_URL").rstrip("/"),
            supabase_audience=values.get("AGENT_VCR_SUPABASE_AUDIENCE", "authenticated").strip()
            or "authenticated",
            stripe_secret_key=values.get("AGENT_VCR_STRIPE_SECRET_KEY", "").strip() or None,
            stripe_webhook_secret=values.get("AGENT_VCR_STRIPE_WEBHOOK_SECRET", "").strip() or None,
            stripe_price_id=values.get("AGENT_VCR_STRIPE_PRICE_ID", "").strip() or None,
        )
        config._validate()
        return config

    @property
    def billing_enabled(self) -> bool:
        """Whether all billing secrets needed for a safe payment flow exist."""
        return bool(self.stripe_secret_key and self.stripe_webhook_secret and self.stripe_price_id)

    @property
    def issuer(self) -> str:
        """Expected issuer claim for Supabase access tokens."""
        return f"{self.supabase_url}/auth/v1"

    @property
    def jwks_url(self) -> str:
        """Supabase's public key-discovery endpoint."""
        return f"{self.issuer}/.well-known/jwks.json"

    def _validate(self) -> None:
        _require_web_url("AGENT_VCR_SUPABASE_URL", self.supabase_url, local_http=False)
        _require_web_url("AGENT_VCR_APP_URL", self.app_url, local_http=True)
        parsed_db = urlparse(self.database_url)
        if parsed_db.scheme not in {"postgres", "postgresql"} or not parsed_db.netloc:
            raise HostedConfigurationError("DATABASE_URL must be a PostgreSQL connection URL")

        supplied_stripe_settings = [
            self.stripe_secret_key,
            self.stripe_webhook_secret,
            self.stripe_price_id,
        ]
        if any(supplied_stripe_settings) and not all(supplied_stripe_settings):
            raise HostedConfigurationError(
                "Stripe requires AGENT_VCR_STRIPE_SECRET_KEY, "
                "AGENT_VCR_STRIPE_WEBHOOK_SECRET, and AGENT_VCR_STRIPE_PRICE_ID together"
            )


def _require_web_url(name: str, value: str, *, local_http: bool) -> None:
    parsed = urlparse(value)
    is_local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    valid_scheme = parsed.scheme == "https" or (local_http and is_local and parsed.scheme == "http")
    if not valid_scheme or not parsed.netloc or parsed.params or parsed.query or parsed.fragment:
        raise HostedConfigurationError(f"{name} must be an HTTPS origin without a path or query")
    if parsed.path not in {"", "/"}:
        raise HostedConfigurationError(f"{name} must be an origin, not a path")


@dataclass(frozen=True)
class HostedUser:
    """Verified identity claims safe for the application to consume."""

    id: str
    email: str | None


@dataclass(frozen=True)
class HostedAccount:
    """Subscription state mirrored from Stripe webhooks."""

    user_id: str
    email: str | None
    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    plan: str
    subscription_status: str

    @property
    def entitled(self) -> bool:
        """Whether the account currently has a paid entitlement."""
        return self.plan != "free" and self.subscription_status in {"active", "trialing"}


class SupabaseJWTVerifier:
    """Verify Supabase asymmetric access tokens with a small, locked JWKS cache."""

    _CACHE_SECONDS = 600.0

    def __init__(self, config: HostedConfig, *, timeout_seconds: float = 5.0):
        self.config = config
        self.timeout_seconds = timeout_seconds
        self._keys: dict[str, dict[str, Any]] = {}
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def verify(self, token: str) -> HostedUser:
        """Return trusted subject claims or raise a non-sensitive auth error."""
        if not token or len(token) > 16_384:
            raise HostedAuthenticationError("Missing or malformed bearer token")

        jwt = self._jwt_module()
        try:
            header = jwt.get_unverified_header(token)
            key_id = header.get("kid")
            algorithm = header.get("alg")
        except Exception as err:
            raise HostedAuthenticationError("Malformed bearer token") from err
        if not isinstance(key_id, str) or not isinstance(algorithm, str):
            raise HostedAuthenticationError("Bearer token has no signing key")

        key_data = await self._key_for(key_id)
        if key_data.get("alg") != algorithm or algorithm not in {"RS256", "ES256"}:
            raise HostedAuthenticationError("Bearer token uses an unsupported signing algorithm")
        try:
            signing_key = jwt.PyJWK.from_dict(key_data).key
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=[algorithm],
                audience=self.config.supabase_audience,
                issuer=self.config.issuer,
                options={"require": ["exp", "sub"]},
            )
        except Exception as err:
            raise HostedAuthenticationError("Invalid or expired bearer token") from err

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise HostedAuthenticationError("Bearer token has no subject")
        email = claims.get("email")
        return HostedUser(id=subject, email=email if isinstance(email, str) else None)

    async def _key_for(self, key_id: str) -> dict[str, Any]:
        async with self._lock:
            if time.monotonic() >= self._expires_at or key_id not in self._keys:
                await self._refresh_keys_locked()
            key = self._keys.get(key_id)
            if key is None:
                raise HostedAuthenticationError("Unknown signing key")
            return dict(key)

    async def _refresh_keys_locked(self) -> None:
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(_fetch_json, self.config.jwks_url, self.timeout_seconds),
                timeout=self.timeout_seconds + 1.0,
            )
        except (asyncio.TimeoutError, OSError, urllib.error.URLError, ValueError) as err:
            raise HostedAuthenticationError("Identity provider keys are unavailable") from err
        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list):
            raise HostedAuthenticationError("Identity provider returned invalid signing keys")
        refreshed: dict[str, dict[str, Any]] = {}
        for key in keys:
            if not isinstance(key, dict):
                continue
            key_id = key.get("kid")
            if isinstance(key_id, str) and key.get("alg") in {"RS256", "ES256"}:
                refreshed[key_id] = dict(key)
        if not refreshed:
            raise HostedAuthenticationError("Identity provider has no supported signing keys")
        self._keys = refreshed
        self._expires_at = time.monotonic() + self._CACHE_SECONDS

    @staticmethod
    def _jwt_module() -> Any:
        try:
            import jwt
        except ImportError as err:
            raise HostedDependencyError(
                'Hosted authentication needs: pip install "ai-agent-vcr[hosted]"'
            ) from err
        return jwt


def _fetch_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    """Fetch a bounded JSON document for the verifier's background thread."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("response is too large")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("response is not an object")
    return parsed


class HostedAccountStore:
    """Small Postgres ledger for customer mappings and idempotent Stripe events."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    def ensure_schema(self) -> None:
        """Create the hosted-only tables. Safe to run for every deployment."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_vcr_hosted_accounts (
                    user_id TEXT PRIMARY KEY,
                    email TEXT,
                    stripe_customer_id TEXT UNIQUE,
                    stripe_subscription_id TEXT UNIQUE,
                    plan TEXT NOT NULL DEFAULT 'free',
                    subscription_status TEXT NOT NULL DEFAULT 'inactive',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_vcr_stripe_events (
                    event_id TEXT PRIMARY KEY,
                    received_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def ensure_account(self, user: HostedUser) -> HostedAccount:
        """Insert the authenticated user once and return its subscription state."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_vcr_hosted_accounts (user_id, email)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET email = COALESCE(EXCLUDED.email, agent_vcr_hosted_accounts.email),
                    updated_at = CURRENT_TIMESTAMP
                RETURNING user_id, email, stripe_customer_id, stripe_subscription_id,
                          plan, subscription_status
                """,
                (user.id, user.email),
            )
            row = cursor.fetchone()
        return _account_from_row(row)

    def get_account(self, user_id: str) -> HostedAccount | None:
        """Load an account without leaking whether other users exist."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, stripe_customer_id, stripe_subscription_id,
                       plan, subscription_status
                FROM agent_vcr_hosted_accounts WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
        return _account_from_row(row) if row else None

    def apply_stripe_event(self, event: Mapping[str, Any]) -> bool:
        """Record one verified Stripe event exactly once and update entitlement state."""
        event_id = event.get("id")
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(event_id, str) or not event_id or not isinstance(event_type, str):
            raise HostedBillingError("Stripe event has no id or type")
        obj = data.get("object") if isinstance(data, Mapping) else None
        if not isinstance(obj, Mapping):
            raise HostedBillingError("Stripe event has no object")

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_vcr_stripe_events (event_id) VALUES (%s)
                ON CONFLICT (event_id) DO NOTHING RETURNING event_id
                """,
                (event_id,),
            )
            if cursor.fetchone() is None:
                return False
            if event_type == "checkout.session.completed":
                self._apply_checkout_completed(cursor, obj)
            elif event_type.startswith("customer.subscription."):
                self._apply_subscription(cursor, obj)
        return True

    def _apply_checkout_completed(self, cursor: Any, checkout: Mapping[str, Any]) -> None:
        metadata = checkout.get("metadata")
        user_id = checkout.get("client_reference_id")
        if not isinstance(user_id, str) or not user_id:
            user_id = metadata.get("agent_vcr_user_id") if isinstance(metadata, Mapping) else None
        customer_id = _stripe_id(checkout.get("customer"))
        subscription_id = _stripe_id(checkout.get("subscription"))
        if not isinstance(user_id, str) or not user_id:
            return
        cursor.execute(
            """
            INSERT INTO agent_vcr_hosted_accounts
                (user_id, stripe_customer_id, stripe_subscription_id, subscription_status)
            VALUES (%s, %s, %s, 'pending')
            ON CONFLICT (user_id) DO UPDATE
            SET stripe_customer_id = COALESCE(EXCLUDED.stripe_customer_id,
                                              agent_vcr_hosted_accounts.stripe_customer_id),
                stripe_subscription_id = COALESCE(EXCLUDED.stripe_subscription_id,
                                                  agent_vcr_hosted_accounts.stripe_subscription_id),
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, customer_id, subscription_id),
        )

    def _apply_subscription(self, cursor: Any, subscription: Mapping[str, Any]) -> None:
        metadata = subscription.get("metadata")
        user_id = metadata.get("agent_vcr_user_id") if isinstance(metadata, Mapping) else None
        customer_id = _stripe_id(subscription.get("customer"))
        subscription_id = _stripe_id(subscription.get("id"))
        status = subscription.get("status")
        if not isinstance(status, str):
            status = "inactive"
        plan = "pro" if status in {"active", "trialing"} else "free"

        if isinstance(user_id, str) and user_id:
            cursor.execute(
                """
                INSERT INTO agent_vcr_hosted_accounts
                    (user_id, stripe_customer_id, stripe_subscription_id, plan, subscription_status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET stripe_customer_id = COALESCE(EXCLUDED.stripe_customer_id,
                                                  agent_vcr_hosted_accounts.stripe_customer_id),
                    stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                    plan = EXCLUDED.plan,
                    subscription_status = EXCLUDED.subscription_status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, customer_id, subscription_id, plan, status),
            )
        elif customer_id:
            cursor.execute(
                """
                UPDATE agent_vcr_hosted_accounts
                SET stripe_subscription_id = %s, plan = %s, subscription_status = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE stripe_customer_id = %s
                """,
                (subscription_id, plan, status, customer_id),
            )

    def _connection(self) -> Any:
        try:
            import psycopg
        except ImportError as err:
            raise HostedDependencyError(
                'Hosted account storage needs: pip install "ai-agent-vcr[hosted]"'
            ) from err
        try:
            return psycopg.connect(self.database_url, connect_timeout=5)
        except Exception as err:
            raise HostedBillingError("Hosted account database is unavailable") from err


def _account_from_row(row: Any) -> HostedAccount:
    if not isinstance(row, tuple) or len(row) != 6:
        raise HostedBillingError("Hosted account database returned an invalid record")
    return HostedAccount(
        user_id=str(row[0]),
        email=str(row[1]) if row[1] is not None else None,
        stripe_customer_id=str(row[2]) if row[2] is not None else None,
        stripe_subscription_id=str(row[3]) if row[3] is not None else None,
        plan=str(row[4]),
        subscription_status=str(row[5]),
    )


def _stripe_id(value: Any) -> str | None:
    """Extract a Stripe expandable id without trusting nested data."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        nested = value.get("id")
        return nested if isinstance(nested, str) and nested else None
    return None


class StripeBilling:
    """Stripe-hosted checkout and webhook verification boundary."""

    def __init__(self, config: HostedConfig):
        if not config.billing_enabled:
            raise HostedConfigurationError("Stripe billing is not configured")
        self.config = config

    def create_checkout(self, user: HostedUser, account: HostedAccount) -> str:
        """Create a subscription Checkout session owned by the verified user."""
        client = self._client()
        params: dict[str, Any] = {
            "mode": "subscription",
            "line_items": [{"price": self.config.stripe_price_id, "quantity": 1}],
            "success_url": f"{self.config.app_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{self.config.app_url}/billing/cancelled",
            "client_reference_id": user.id,
            "metadata": {"agent_vcr_user_id": user.id},
            "subscription_data": {"metadata": {"agent_vcr_user_id": user.id}},
        }
        if account.stripe_customer_id:
            params["customer"] = account.stripe_customer_id
        elif user.email:
            params["customer_email"] = user.email
        try:
            checkout = client.v1.checkout.sessions.create(params)
        except Exception as err:
            raise HostedBillingError("Unable to start Stripe Checkout") from err
        checkout_url = getattr(checkout, "url", None)
        if not isinstance(checkout_url, str) or not checkout_url.startswith("https://"):
            raise HostedBillingError("Stripe did not return a safe Checkout URL")
        return checkout_url

    def create_portal(self, account: HostedAccount) -> str:
        """Create a Stripe-hosted billing portal for an existing customer only."""
        if not account.stripe_customer_id:
            raise HostedBillingError("No billing customer exists for this account")
        try:
            portal = self._client().v1.billing_portal.sessions.create(
                {"customer": account.stripe_customer_id, "return_url": self.config.app_url}
            )
        except Exception as err:
            raise HostedBillingError("Unable to start the Stripe billing portal") from err
        portal_url = getattr(portal, "url", None)
        if not isinstance(portal_url, str) or not portal_url.startswith("https://"):
            raise HostedBillingError("Stripe did not return a safe billing portal URL")
        return portal_url

    def verify_webhook(self, payload: bytes, signature: str | None) -> Mapping[str, Any]:
        """Verify Stripe's signature before returning an event to persist."""
        if not signature:
            raise HostedBillingError("Missing Stripe-Signature header")
        try:
            import stripe
        except ImportError as err:
            raise HostedDependencyError(
                'Stripe billing needs: pip install "ai-agent-vcr[hosted]"'
            ) from err
        try:
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=signature,
                secret=self.config.stripe_webhook_secret,
            )
        except Exception as err:
            raise HostedBillingError("Invalid Stripe webhook signature") from err
        if not isinstance(event, Mapping):
            raise HostedBillingError("Stripe returned an invalid webhook event")
        return event

    def _client(self) -> Any:
        try:
            from stripe import StripeClient
        except ImportError as err:
            raise HostedDependencyError(
                'Stripe billing needs: pip install "ai-agent-vcr[hosted]"'
            ) from err
        return StripeClient(self.config.stripe_secret_key)


class HostedControlPlane:
    """Composes authentication, billing, and durable account state for the server."""

    def __init__(self, config: HostedConfig):
        self.config = config
        self.auth = SupabaseJWTVerifier(config)
        self.accounts = HostedAccountStore(config.database_url)
        self.billing = StripeBilling(config) if config.billing_enabled else None
