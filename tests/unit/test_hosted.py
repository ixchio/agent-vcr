"""Unit tests for the opt-in Railway/Supabase/Stripe control plane."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent_vcr.hosted import (
    HostedAccount,
    HostedConfig,
    HostedConfigurationError,
    HostedControlPlane,
    HostedUser,
    StripeBilling,
    SupabaseJWTVerifier,
)
from agent_vcr.server import VCRServer


def _environment(**overrides: str) -> dict[str, str]:
    values = {
        "AGENT_VCR_SUPABASE_URL": "https://tenant.supabase.co",
        "DATABASE_URL": "postgresql://app:password@postgres:5432/agent_vcr",
        "AGENT_VCR_APP_URL": "https://app.agent-vcr.example",
        "AGENT_VCR_STRIPE_SECRET_KEY": "sk_test_123",
        "AGENT_VCR_STRIPE_WEBHOOK_SECRET": "whsec_test_123",
        "AGENT_VCR_STRIPE_PRICE_ID": "price_pro_123",
    }
    values.update(overrides)
    return values


def test_hosted_config_requires_complete_billing_and_safe_origins():
    config = HostedConfig.from_environment(_environment())
    assert config.billing_enabled is True
    assert config.issuer == "https://tenant.supabase.co/auth/v1"
    assert config.jwks_url.endswith("/.well-known/jwks.json")

    with pytest.raises(HostedConfigurationError, match="Stripe requires"):
        HostedConfig.from_environment(_environment(AGENT_VCR_STRIPE_PRICE_ID=""))
    with pytest.raises(HostedConfigurationError, match="HTTPS origin"):
        HostedConfig.from_environment(_environment(AGENT_VCR_APP_URL="http://public.example"))


def test_supabase_verifier_checks_key_issuer_audience_and_expiry(monkeypatch):
    received: dict[str, object] = {}
    fake_jwt = types.ModuleType("jwt")
    fake_jwt.get_unverified_header = lambda token: {"kid": "key-1", "alg": "RS256"}

    class FakeJWK:
        @classmethod
        def from_dict(cls, value):
            received["key_data"] = value
            return SimpleNamespace(key="public-key")

    def decode(token, key, **kwargs):
        received["key"] = key
        received.update(kwargs)
        return {"sub": "user-123", "email": "dev@example.com"}

    fake_jwt.PyJWK = FakeJWK
    fake_jwt.decode = decode
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)

    verifier = SupabaseJWTVerifier(HostedConfig.from_environment(_environment()))
    verifier._keys = {"key-1": {"kid": "key-1", "alg": "RS256", "kty": "RSA"}}
    verifier._expires_at = time.monotonic() + 60

    user = asyncio.run(verifier.verify("valid-token"))

    assert user == HostedUser(id="user-123", email="dev@example.com")
    assert received["key"] == "public-key"
    assert received["issuer"] == "https://tenant.supabase.co/auth/v1"
    assert received["audience"] == "authenticated"
    assert received["algorithms"] == ["RS256"]


def test_stripe_checkout_uses_verified_user_and_server_selected_price(monkeypatch):
    captured: dict[str, object] = {}
    fake_stripe = types.ModuleType("stripe")

    class FakeSessions:
        def create(self, params):
            captured.update(params)
            return SimpleNamespace(url="https://checkout.stripe.com/pay/test")

    class FakeClient:
        def __init__(self, secret_key):
            captured["secret_key"] = secret_key
            self.v1 = SimpleNamespace(checkout=SimpleNamespace(sessions=FakeSessions()))

    fake_stripe.StripeClient = FakeClient
    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)

    config = HostedConfig.from_environment(_environment())
    account = HostedAccount("user-123", "dev@example.com", None, None, "free", "inactive")
    checkout_url = StripeBilling(config).create_checkout(HostedUser("user-123", "dev@example.com"), account)

    assert checkout_url == "https://checkout.stripe.com/pay/test"
    assert captured["secret_key"] == "sk_test_123"
    assert captured["mode"] == "subscription"
    assert captured["line_items"] == [{"price": "price_pro_123", "quantity": 1}]
    assert captured["client_reference_id"] == "user-123"
    assert captured["subscription_data"] == {"metadata": {"agent_vcr_user_id": "user-123"}}


def test_hosted_routes_require_bearer_token_and_reject_unsigned_stripe_events(tmp_path):
    control_plane = HostedControlPlane(HostedConfig.from_environment(_environment()))
    server = VCRServer(vcr_dir=str(tmp_path), hosted=control_plane)
    client = TestClient(server.app, raise_server_exceptions=False)

    assert client.get("/api/hosted/me").status_code == 401
    assert client.post("/api/hosted/billing/webhook", json={"id": "evt_123"}).status_code == 400
