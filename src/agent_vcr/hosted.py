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
import contextlib
import json
import os
import time
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


class HostedConfigurationError(ValueError):
    """Raised when hosted mode has incomplete or unsafe configuration."""


class HostedAuthenticationError(ValueError):
    """Raised when a Supabase bearer token is absent or invalid."""


class HostedDependencyError(RuntimeError):
    """Raised when an optional hosted dependency was not installed."""


class HostedUnavailableError(HostedDependencyError):
    """Raised when a hosted backing service (database, identity provider) is unreachable.

    It subclasses :class:`HostedDependencyError` so callers that already map
    dependency failures to "503 Service Unavailable" keep doing so.
    """


class HostedBillingError(RuntimeError):
    """Raised for a billing request that cannot safely be completed."""


class HostedConflictError(HostedBillingError):
    """Raised when a billing request conflicts with the account's current state."""


# Subscription statuses that mean "this account already has a subscription":
# starting another Checkout would bill the customer twice.
_LIVE_SUBSCRIPTION_STATUSES = frozenset(
    {"active", "trialing", "past_due", "unpaid", "paused", "pending"}
)
_ENTITLED_STATUSES = frozenset({"active", "trialing"})
# Stripe never moves a subscription out of these statuses.
_TERMINAL_STATUSES = frozenset({"canceled", "incomplete_expired"})


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
        return self.plan != "free" and self.subscription_status in _ENTITLED_STATUSES

    @property
    def has_live_subscription(self) -> bool:
        """Whether a (possibly unpaid) subscription exists that a new Checkout would duplicate."""
        return self.entitled or self.subscription_status in _LIVE_SUBSCRIPTION_STATUSES


class SupabaseJWTVerifier:
    """Verify Supabase asymmetric access tokens with a small JWKS cache.

    Cache hits never wait on a lock. Refreshes are single-flight, and an
    unknown ``kid`` can trigger at most one refetch per
    ``_MIN_REFETCH_SECONDS`` (a negative cache), so anonymous requests with
    random key ids cannot queue every login behind identity-provider fetches.
    """

    _CACHE_SECONDS = 600.0
    _MIN_REFETCH_SECONDS = 60.0
    _FAILURE_BACKOFF_SECONDS = 5.0

    def __init__(self, config: HostedConfig, *, timeout_seconds: float = 5.0):
        self.config = config
        self.timeout_seconds = timeout_seconds
        self._keys: dict[str, dict[str, Any]] = {}
        self._expires_at = 0.0
        self._last_fetch_at: float | None = None
        self._last_fetch_failed = False
        # Created lazily inside the running loop: on Python 3.9 a Lock built in
        # __init__ binds to whatever loop exists at import time, not uvicorn's.
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

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

    def _refresh_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _cached_key(self, key_id: str) -> dict[str, Any] | None:
        if time.monotonic() >= self._expires_at:
            return None
        key = self._keys.get(key_id)
        return dict(key) if key is not None else None

    def _reject_during_cooldown(self) -> None:
        """Fail fast instead of refetching when a fetch just happened."""
        if self._last_fetch_at is None:
            return
        elapsed = time.monotonic() - self._last_fetch_at
        if self._last_fetch_failed:
            if elapsed < self._FAILURE_BACKOFF_SECONDS:
                raise HostedUnavailableError("Identity provider keys are unavailable")
            return
        if time.monotonic() < self._expires_at and elapsed < self._MIN_REFETCH_SECONDS:
            # Keys are fresh and were fetched moments ago: this kid is simply unknown.
            raise HostedAuthenticationError("Unknown signing key")

    async def _key_for(self, key_id: str) -> dict[str, Any]:
        key = self._cached_key(key_id)
        if key is not None:
            return key  # lock-free fast path
        self._reject_during_cooldown()
        async with self._refresh_lock():
            # Another coroutine may have refreshed while this one waited.
            key = self._cached_key(key_id)
            if key is not None:
                return key
            self._reject_during_cooldown()
            await self._refresh_keys_locked()
            key = self._cached_key(key_id)
            if key is None:
                raise HostedAuthenticationError("Unknown signing key")
            return key

    async def _refresh_keys_locked(self) -> None:
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(_fetch_json, self.config.jwks_url, self.timeout_seconds),
                timeout=self.timeout_seconds + 1.0,
            )
        except Exception as err:
            self._record_fetch(failed=True)
            raise HostedUnavailableError("Identity provider keys are unavailable") from err
        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list):
            self._record_fetch(failed=True)
            raise HostedUnavailableError("Identity provider returned invalid signing keys")
        refreshed: dict[str, dict[str, Any]] = {}
        for key in keys:
            if not isinstance(key, dict):
                continue
            key_id = key.get("kid")
            if isinstance(key_id, str) and key.get("alg") in {"RS256", "ES256"}:
                refreshed[key_id] = dict(key)
        if not refreshed:
            self._record_fetch(failed=True)
            raise HostedUnavailableError("Identity provider has no supported signing keys")
        self._keys = refreshed
        self._expires_at = time.monotonic() + self._CACHE_SECONDS
        self._record_fetch(failed=False)

    def _record_fetch(self, *, failed: bool) -> None:
        self._last_fetch_at = time.monotonic()
        self._last_fetch_failed = failed

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


_ACCOUNTS = "agent_vcr_hosted_accounts"
_SUBSCRIPTIONS = "agent_vcr_stripe_subscriptions"
_STATEMENT_TIMEOUT = "10s"


def _sql_list(values: frozenset[str]) -> str:
    """Render trusted module constants as a SQL string-literal list."""
    return ", ".join(f"'{value}'" for value in sorted(values))


class HostedAccountStore:
    """Small Postgres ledger for customer mappings and idempotent Stripe events.

    Every Stripe subscription is tracked in its own row, so a customer with
    two subscriptions keeps its entitlement when either one is cancelled. Each
    row only accepts an event at least as new (Stripe ``created``) as the last
    one applied, and a terminal status (``canceled``) is never overwritten, so
    out-of-order deliveries cannot resurrect or revoke a plan. The account's
    ``plan`` is then derived from its best subscription.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url

    def ensure_schema(self) -> None:
        """Create the hosted-only tables. Safe to run for every deployment."""
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_ACCOUNTS} (
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
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_SUBSCRIPTIONS} (
                    subscription_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    stripe_customer_id TEXT,
                    status TEXT NOT NULL,
                    event_created BIGINT NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Backfill subscriptions recorded before per-subscription tracking existed.
            cursor.execute(
                f"""
                INSERT INTO {_SUBSCRIPTIONS}
                    (subscription_id, user_id, stripe_customer_id, status, event_created)
                SELECT stripe_subscription_id, user_id, stripe_customer_id,
                       subscription_status, 0
                FROM {_ACCOUNTS} WHERE stripe_subscription_id IS NOT NULL
                ON CONFLICT (subscription_id) DO NOTHING
                """
            )

    def ensure_account(self, user: HostedUser) -> HostedAccount:
        """Insert the authenticated user once and return its subscription state."""
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {_ACCOUNTS} (user_id, email)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET email = COALESCE(EXCLUDED.email, {_ACCOUNTS}.email),
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
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT user_id, email, stripe_customer_id, stripe_subscription_id,
                       plan, subscription_status
                FROM {_ACCOUNTS} WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
        return _account_from_row(row) if row else None

    def apply_stripe_event(self, event: Mapping[str, Any]) -> bool:
        """Record one verified Stripe event exactly once and update entitlement state.

        Returns False for a redelivered event id. The ledger insert and the state
        change share one transaction, so a failure leaves the event retryable.
        """
        event_id = event.get("id")
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(event_id, str) or not event_id or not isinstance(event_type, str):
            raise HostedBillingError("Stripe event has no id or type")
        obj = data.get("object") if isinstance(data, Mapping) else None
        if not isinstance(obj, Mapping):
            raise HostedBillingError("Stripe event has no object")
        created = event.get("created")
        event_created = (
            int(created)
            if isinstance(created, (int, float)) and not isinstance(created, bool)
            else 0
        )

        with self._cursor() as cursor:
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
                self._apply_subscription(cursor, obj, event_created)
        return True

    def _lock_account(self, cursor: Any, user_id: str, customer_id: str | None) -> None:
        """Upsert the account row; the row lock serializes concurrent webhooks per user."""
        cursor.execute(
            f"""
            INSERT INTO {_ACCOUNTS} (user_id, stripe_customer_id)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE
            SET stripe_customer_id = COALESCE(EXCLUDED.stripe_customer_id,
                                              {_ACCOUNTS}.stripe_customer_id),
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, customer_id),
        )

    def _apply_checkout_completed(self, cursor: Any, checkout: Mapping[str, Any]) -> None:
        metadata = checkout.get("metadata")
        user_id = checkout.get("client_reference_id")
        if not isinstance(user_id, str) or not user_id:
            user_id = metadata.get("agent_vcr_user_id") if isinstance(metadata, Mapping) else None
        customer_id = _stripe_id(checkout.get("customer"))
        subscription_id = _stripe_id(checkout.get("subscription"))
        if not isinstance(user_id, str) or not user_id:
            return
        self._lock_account(cursor, user_id, customer_id)
        if subscription_id:
            # A placeholder until the subscription's own events arrive (they may
            # already have): event_created=0 lets any real event supersede it,
            # and an existing row keeps its status.
            cursor.execute(
                f"""
                INSERT INTO {_SUBSCRIPTIONS} AS s
                    (subscription_id, user_id, stripe_customer_id, status, event_created)
                VALUES (%s, %s, %s, 'pending', 0)
                ON CONFLICT (subscription_id) DO UPDATE
                SET user_id = COALESCE(s.user_id, EXCLUDED.user_id),
                    stripe_customer_id = COALESCE(s.stripe_customer_id,
                                                  EXCLUDED.stripe_customer_id)
                """,
                (subscription_id, user_id, customer_id),
            )
        if customer_id:
            cursor.execute(
                f"""
                UPDATE {_SUBSCRIPTIONS} SET user_id = %s
                WHERE stripe_customer_id = %s AND user_id IS NULL
                """,
                (user_id, customer_id),
            )
        self._recompute_entitlement(cursor, user_id)

    def _apply_subscription(
        self, cursor: Any, subscription: Mapping[str, Any], event_created: int
    ) -> None:
        metadata = subscription.get("metadata")
        user_id = metadata.get("agent_vcr_user_id") if isinstance(metadata, Mapping) else None
        customer_id = _stripe_id(subscription.get("customer"))
        subscription_id = _stripe_id(subscription.get("id"))
        status = subscription.get("status")
        if not isinstance(status, str) or not status:
            status = "inactive"
        if not subscription_id:
            return
        if not (isinstance(user_id, str) and user_id) and customer_id:
            cursor.execute(
                f"SELECT user_id FROM {_ACCOUNTS} WHERE stripe_customer_id = %s",
                (customer_id,),
            )
            row = cursor.fetchone()
            user_id = str(row[0]) if row else None
        if isinstance(user_id, str) and user_id:
            self._lock_account(cursor, user_id, customer_id)
        else:
            user_id = None  # unknown customer: linked later by checkout completion
        terminal = _sql_list(_TERMINAL_STATUSES)
        cursor.execute(
            f"""
            INSERT INTO {_SUBSCRIPTIONS} AS s
                (subscription_id, user_id, stripe_customer_id, status, event_created)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (subscription_id) DO UPDATE
            SET user_id = COALESCE(EXCLUDED.user_id, s.user_id),
                stripe_customer_id = COALESCE(EXCLUDED.stripe_customer_id,
                                              s.stripe_customer_id),
                status = EXCLUDED.status,
                event_created = EXCLUDED.event_created,
                updated_at = CURRENT_TIMESTAMP
            WHERE s.status NOT IN ({terminal})
              AND (EXCLUDED.status IN ({terminal})
                   OR EXCLUDED.event_created >= s.event_created)
            """,
            (subscription_id, user_id, customer_id, status, event_created),
        )
        if user_id:
            self._recompute_entitlement(cursor, user_id)

    def _recompute_entitlement(self, cursor: Any, user_id: str) -> None:
        """Mirror the user's best subscription (entitled first, then live, then newest)."""
        cursor.execute(
            f"""
            SELECT subscription_id, status FROM {_SUBSCRIPTIONS}
            WHERE user_id = %s
            ORDER BY CASE WHEN status IN ({_sql_list(_ENTITLED_STATUSES)}) THEN 0
                          WHEN status IN ({_sql_list(_TERMINAL_STATUSES)}) THEN 2
                          ELSE 1 END,
                     event_created DESC,
                     updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return
        subscription_id, status = str(row[0]), str(row[1])
        plan = "pro" if status in _ENTITLED_STATUSES else "free"
        cursor.execute(
            f"""
            UPDATE {_ACCOUNTS}
            SET stripe_subscription_id = %s, plan = %s, subscription_status = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
            """,
            (subscription_id, plan, status, user_id),
        )

    @contextlib.contextmanager
    def _cursor(self) -> Iterator[Any]:
        """One transaction; database failures surface as :class:`HostedUnavailableError`."""
        psycopg = self._driver()
        try:
            with psycopg.connect(
                self.database_url, connect_timeout=5
            ) as connection, connection.cursor() as cursor:
                # SET LOCAL is transaction-scoped, so it is also safe behind poolers.
                cursor.execute(f"SET LOCAL statement_timeout = '{_STATEMENT_TIMEOUT}'")
                yield cursor
        except psycopg.Error as err:
            raise HostedUnavailableError("Hosted account database is unavailable") from err
        except OSError as err:
            raise HostedUnavailableError("Hosted account database is unavailable") from err

    @staticmethod
    def _driver() -> Any:
        try:
            import psycopg
        except ImportError as err:
            raise HostedDependencyError(
                'Hosted account storage needs: pip install "ai-agent-vcr[hosted]"'
            ) from err
        return psycopg


def _account_from_row(row: Any) -> HostedAccount:
    if not isinstance(row, tuple) or len(row) != 6:
        raise HostedUnavailableError("Hosted account database returned an invalid record")
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
        """Create a subscription Checkout session owned by the verified user.

        Refuses (:class:`HostedConflictError`) when the account already has a
        subscription, so a second Checkout cannot bill the customer twice; the
        billing portal is the place to change an existing subscription.
        """
        if account.has_live_subscription:
            raise HostedConflictError(
                "This account already has a subscription; manage it in the billing portal"
            )
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
