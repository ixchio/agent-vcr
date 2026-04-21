"""
OpenHands Sentinel Demo — Self-contained simulation.

This demo simulates an OpenHands agent writing code and shows how
Sentinel catches violations in real-time, warns the agent, and records
everything to agent-vcr.

Run:
    python examples/sentinel_demo.py

No OpenHands installation required — this simulates the EventStream
to demonstrate the full workflow.
"""

from __future__ import annotations

import os
import sys
import time

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_vcr import VCRRecorder
from openhands_sentinel import Sentinel, SentinelConfig
from openhands_sentinel.analyzer import Severity

# ──────────────────────────────────────────────
#  Terminal Colors
# ──────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


def print_step(step: int, description: str) -> None:
    print(f"\n{C.CYAN}{C.BOLD}{'─' * 60}")
    print(f"  STEP {step}: {description}")
    print(f"{'─' * 60}{C.RESET}\n")
    time.sleep(0.5)


def print_agent(message: str) -> None:
    print(f"  {C.BLUE}{C.BOLD}🤖 AGENT:{C.RESET} {message}")
    time.sleep(0.3)


def print_sentinel(message: str, is_warning: bool = False) -> None:
    if is_warning:
        print(f"  {C.RED}{C.BOLD}🛡️ SENTINEL:{C.RESET} {C.RED}{message}{C.RESET}")
    else:
        print(f"  {C.GREEN}{C.BOLD}🛡️ SENTINEL:{C.RESET} {message}")
    time.sleep(0.3)


# ──────────────────────────────────────────────
#  Simulated Agent Code (what the agent writes)
# ──────────────────────────────────────────────

# Step 1: Agent writes a clean utility file
CLEAN_UTILS = '''"""Authentication utilities."""

import hashlib
import secrets
from datetime import datetime, timedelta


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    salt, expected_hash = stored_hash.split(":")
    actual_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return actual_hash == expected_hash


def generate_token(user_id: str, expiry_hours: int = 24) -> str:
    """Generate a simple auth token."""
    payload = f"{user_id}:{datetime.utcnow().isoformat()}"
    return hashlib.sha256(payload.encode()).hexdigest()
'''

# Step 2: Agent writes a massive monolithic handler (VIOLATION)
BAD_HANDLER = '''"""API request handlers."""

import json
import logging
import os
import re
import hashlib
import secrets
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a password — DUPLICATE of auth/utils.py!"""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def handle_auth_request(request_type, username, password, email, phone,
                        address, role, permissions, metadata):
    """Handle ALL authentication in one massive function."""
    result = {}

    if request_type == "login":
        if not username or not password:
            result["error"] = "Missing credentials"
            return result
        if len(password) < 8:
            result["error"] = "Password too short"
            return result
        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            result["error"] = "Invalid username format"
            return result
        hashed = hash_password(password)
        if hashed:
            result["token"] = hashlib.sha256(username.encode()).hexdigest()
            result["expires"] = (datetime.utcnow() + timedelta(hours=24)).isoformat()
            result["user"] = {"username": username, "role": role}
            logger.info(f"Login successful: {username}")
        else:
            result["error"] = "Authentication failed"
            logger.warning(f"Login failed: {username}")

    elif request_type == "register":
        if not username or not password or not email:
            result["error"] = "Missing required fields"
            return result
        if len(password) < 8:
            result["error"] = "Password too short"
            return result
        if not re.match(r"^[\\w.+-]+@[\\w-]+\\.[\\w.]+$", email):
            result["error"] = "Invalid email"
            return result
        if phone and not re.match(r"^\\+?\\d{10,15}$", phone):
            result["error"] = "Invalid phone number"
            return result
        hashed = hash_password(password)
        user = {
            "username": username,
            "email": email,
            "phone": phone,
            "address": address,
            "role": role or "user",
            "permissions": permissions or [],
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "password_hash": hashed,
        }
        result["user"] = user
        result["message"] = "Registration successful"
        logger.info(f"User registered: {username}")

    elif request_type == "reset_password":
        if not email:
            result["error"] = "Email required"
            return result
        token = secrets.token_urlsafe(32)
        result["reset_token"] = token
        result["expires"] = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        logger.info(f"Password reset requested: {email}")

    elif request_type == "change_password":
        if not password:
            result["error"] = "New password required"
            return result
        if len(password) < 8:
            result["error"] = "Password too short"
            return result
        hashed = hash_password(password)
        result["message"] = "Password changed"
        logger.info(f"Password changed: {username}")

    elif request_type == "verify_email":
        result["verified"] = True
        logger.info(f"Email verified: {email}")

    elif request_type == "update_profile":
        updates = {}
        if email:
            updates["email"] = email
        if phone:
            updates["phone"] = phone
        if address:
            updates["address"] = address
        if metadata:
            updates["metadata"] = metadata
        result["updated"] = updates
        logger.info(f"Profile updated: {username}")

    elif request_type == "delete_account":
        result["deleted"] = True
        result["message"] = "Account scheduled for deletion"
        logger.info(f"Account deletion requested: {username}")

    elif request_type == "list_sessions":
        result["sessions"] = []
        logger.info(f"Sessions listed: {username}")

    elif request_type == "revoke_token":
        result["revoked"] = True
        logger.info(f"Token revoked: {username}")

    else:
        result["error"] = f"Unknown request type: {request_type}"
        logger.error(f"Unknown request type: {request_type}")

    return result
'''

# Step 3: Agent self-corrects after Sentinel warning
FIXED_HANDLER = '''"""API request handlers — refactored after Sentinel review."""

import logging
from datetime import datetime, timedelta

from auth.utils import hash_password, verify_password, generate_token

logger = logging.getLogger(__name__)


def handle_login(username: str, password: str) -> dict:
    """Handle user login."""
    if not username or not password:
        return {"error": "Missing credentials"}

    hashed = hash_password(password)
    token = generate_token(username)
    return {
        "token": token,
        "expires": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
        "user": {"username": username},
    }


def handle_register(username: str, password: str, email: str) -> dict:
    """Handle user registration."""
    if not all([username, password, email]):
        return {"error": "Missing required fields"}

    hashed = hash_password(password)
    return {
        "user": {"username": username, "email": email},
        "message": "Registration successful",
    }


def handle_password_reset(email: str) -> dict:
    """Handle password reset request."""
    if not email:
        return {"error": "Email required"}

    import secrets
    token = secrets.token_urlsafe(32)
    return {
        "reset_token": token,
        "expires": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
    }
'''


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────

def main() -> None:
    print(f"""
{C.CYAN}{C.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🛡️  OPENHANDS SENTINEL — LIVE DEMO                          ║
║                                                               ║
║   Watching an AI agent build an auth system.                   ║
║   Sentinel catches violations in real-time.                    ║
║   Agent self-corrects. Zero human intervention.                ║
║                                                               ║
║   Built on agent-vcr for full time-travel audit trails.        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
{C.RESET}""")

    time.sleep(1)

    # Initialize Sentinel with agent-vcr recording
    recorder = VCRRecorder(output_dir=".vcr/sentinel-demo")
    sentinel = Sentinel(
        config=SentinelConfig(
            max_function_lines=40,
            max_complexity=8,
            max_function_params=5,
        ),
        recorder=recorder,
    )
    sentinel.start_session("sentinel-demo")

    # ─── STEP 1: Agent writes clean utility code ───
    print_step(1, "Agent writes auth/utils.py — clean utility functions")
    print_agent("Writing authentication utilities...")
    time.sleep(0.5)

    result1 = sentinel.check_file("auth/utils.py", CLEAN_UTILS)

    if not result1.violations:
        print_sentinel("auth/utils.py — CLEAN ✓ No issues detected.")
        print(f"    {C.DIM}Metrics: {result1.metrics}{C.RESET}")
    time.sleep(1)

    # ─── STEP 2: Agent writes BAD monolithic handler ───
    print_step(2, "Agent writes handlers.py — MASSIVE monolithic function")
    print_agent("Writing request handler... (this is where things go wrong)")
    time.sleep(0.5)

    result2, warning = sentinel.check_and_warn("handlers.py", BAD_HANDLER)

    if warning:
        print()
        print_sentinel("VIOLATIONS DETECTED!", is_warning=True)
        print()

        for v in result2.violations:
            severity_colors = {
                Severity.WARNING: C.YELLOW,
                Severity.CRITICAL: C.RED,
                Severity.BLOCKER: f"{C.BG_RED}{C.WHITE}",
            }
            color = severity_colors.get(v.severity, C.DIM)
            print(f"    {color}{v.severity.value.upper():>8}{C.RESET}  {v.message}")
            time.sleep(0.2)

        print()
        print_sentinel("Sending warning to agent...", is_warning=True)
        print()
        print(f"{C.YELLOW}{C.DIM}{'─' * 55}")
        for line in warning.split("\n"):
            print(f"  {line}")
        print(f"{'─' * 55}{C.RESET}")

    time.sleep(1.5)

    # ─── STEP 3: Agent self-corrects ───
    print_step(3, "Agent SELF-CORRECTS after Sentinel warning")
    print_agent("I see the issues. Refactoring handlers.py...")
    print_agent("Removing duplicate hash_password() — using import from auth/utils.py")
    print_agent("Breaking handle_auth_request() into focused functions")
    print_agent("Reducing parameter count with focused function signatures")
    time.sleep(0.5)

    result3 = sentinel.check_file("handlers.py", FIXED_HANDLER)

    if not result3.violations:
        print()
        print_sentinel("handlers.py — CLEAN ✓ All issues resolved!")
        print(f"    {C.DIM}Metrics: {result3.metrics}{C.RESET}")
        sentinel.stats.self_corrections += 1

    time.sleep(1)

    # ─── FINAL REPORT ───
    print(f"\n{C.CYAN}{C.BOLD}{'═' * 60}")
    print("  FINAL SESSION REPORT")
    print(f"{'═' * 60}{C.RESET}\n")

    print(sentinel.get_report())

    # Save the recording
    path = sentinel.save()
    print(f"  {C.GREEN}{C.BOLD}📼 Full audit trail saved to:{C.RESET} {path}")
    print(f"  {C.DIM}Replay with: vcr play {path}{C.RESET}")

    # Cost savings pitch
    print(f"""
{C.MAGENTA}{C.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   💰 SAVINGS BREAKDOWN                                        ║
║                                                               ║
║   Without Sentinel:                                           ║
║     Agent writes bad code → human reviews → rejects PR →      ║
║     agent rewrites → human reviews again.                     ║
║     Cost: 2x LLM calls + human review time.                  ║
║                                                               ║
║   With Sentinel:                                              ║
║     Agent writes bad code → Sentinel catches instantly →      ║
║     agent self-corrects in the SAME session.                  ║
║     Cost: 0 human time. 1 extra LLM call.                    ║
║                                                               ║
║   3 lines to add to any OpenHands session:                    ║
║     recorder = VCRRecorder()                                  ║
║     sentinel = Sentinel(recorder=recorder)                    ║
║     sentinel.attach(runtime.event_stream)                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
{C.RESET}""")


if __name__ == "__main__":
    main()
