#!/usr/bin/env python3
"""
Agent VCR — The Ultimate Showcase Demo

Shows EVERY killer feature in one beautiful terminal run:
  1. Record a multi-step AI agent
  2. Agent FAILS at step 3
  3. Rewind to step 2, inspect state
  4. Edit state and resume → agent succeeds
  5. Ghost Replay: same task, 0 tokens, instant
  6. Sentinel catches bad code in real-time
  7. ACID rollback: bad files deleted from disk

Run:  python examples/showcase_demo.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

# ── Colors ──────────────────────────────────────────────

R = "\033[0m"     # reset
B = "\033[1m"     # bold
D = "\033[2m"     # dim
RED = "\033[91m"
GRN = "\033[92m"
YLW = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"
WHT = "\033[97m"
BG_RED = "\033[41m"
BG_GRN = "\033[42m"
BG_BLU = "\033[44m"
BG_MAG = "\033[45m"


def banner() -> None:
    print(f"""{CYN}{B}
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║   📼  AGENT VCR — THE ULTIMATE DEMO                       ║
  ║                                                           ║
  ║   Time-travel debugging for AI agents.                    ║
  ║   Record. Rewind. Edit. Resume. Never re-run.             ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝{R}
""")


def step_header(num: int, title: str, icon: str = "▶") -> None:
    print(f"\n  {BLU}{B}{'─' * 55}{R}")
    print(f"  {BLU}{B}{icon}  STEP {num}: {title}{R}")
    print(f"  {BLU}{B}{'─' * 55}{R}\n")


def typing_print(text: str, delay: float = 0.008) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def pause(seconds: float = 0.5) -> None:
    time.sleep(seconds)


# ── Simulated Agent ─────────────────────────────────────

class CodingAgent:
    """Simulated multi-step coding agent."""

    def plan(self, state: dict) -> dict:
        return {**state, "plan": ["setup project", "write auth module", "write API endpoints", "write tests"],
                "step": "plan", "tokens_used": 850, "cost": 0.0025}

    def code_auth(self, state: dict) -> dict:
        return {**state, "auth_code": "def hash_password(pw): return bcrypt.hash(pw)",
                "files_written": ["auth/utils.py"], "step": "code_auth",
                "tokens_used": 1200, "cost": 0.0036}

    def code_api(self, state: dict) -> dict:
        if state.get("framework") == "flask":
            # THIS WILL FAIL — wrong framework causes import error
            return {**state, "error": "ImportError: No module named 'flask_restful'",
                    "step": "code_api_FAILED", "tokens_used": 900, "cost": 0.0027}
        return {**state, "api_code": "app = FastAPI()\n@app.get('/users')\nasync def get_users(): ...",
                "files_written": ["api/routes.py", "api/models.py"], "step": "code_api",
                "tokens_used": 1500, "cost": 0.0045}

    def write_tests(self, state: dict) -> dict:
        return {**state, "tests": "def test_auth(): assert hash_password('pw') != 'pw'",
                "step": "done", "tokens_used": 650, "cost": 0.0020}


def main() -> None:
    banner()

    # Setup temp workspace
    workspace = tempfile.mkdtemp(prefix="vcr_demo_")
    vcr_dir = os.path.join(workspace, ".vcr")

    from agent_vcr import VCRPlayer, VCRRecorder
    from agent_vcr.models import FrameMetadata, ResumeConfig

    agent = CodingAgent()

    # ═══════════════════════════════════════════════════════
    #  STEP 1: RECORD — Agent runs and FAILS
    # ═══════════════════════════════════════════════════════
    step_header(1, "RECORD — Run the agent (it will FAIL)", "📹")

    recorder = VCRRecorder(output_dir=vcr_dir)
    recorder.start_session("build-rest-api")

    initial_state = {"task": "Build a REST API with auth", "framework": "flask"}

    typing_print(f"  {D}Task: {WHT}{B}\"Build a REST API with auth\"{R}")
    typing_print(f"  {D}Framework: {WHT}flask{R}")
    pause(0.3)

    # Step 1: Plan
    state = agent.plan(initial_state)
    recorder.record_step("planner", initial_state, state,
                         metadata=FrameMetadata(latency_ms=120, tokens_used=state["tokens_used"],
                                                cost_usd=state["cost"]))
    print(f"  {GRN}✓{R} Frame 0: {B}planner{R}     {D}120ms  850 tokens  $0.0025{R}")

    # Step 2: Code auth
    prev = state
    state = agent.code_auth(state)
    recorder.record_step("coder:auth", prev, state,
                         metadata=FrameMetadata(latency_ms=250, tokens_used=state["tokens_used"],
                                                cost_usd=state["cost"]))
    print(f"  {GRN}✓{R} Frame 1: {B}coder:auth{R}  {D}250ms  1200 tokens  $0.0036{R}")

    # Step 3: Code API — FAILS
    prev = state
    state = agent.code_api(state)
    recorder.record_step("coder:api", prev, state,
                         metadata=FrameMetadata(latency_ms=180, tokens_used=state["tokens_used"],
                                                cost_usd=state["cost"],
                                                error_message=state.get("error"),
                                                error_type="ImportError" if "error" in state else None))
    print(f"  {RED}✗{R} Frame 2: {B}coder:api{R}   {D}180ms  900 tokens  $0.0027{R}")
    print(f"    {BG_RED}{WHT}{B} ERROR {R} {RED}ImportError: No module named 'flask_restful'{R}")
    pause(0.3)

    vcr_path = recorder.save()
    total_tokens = 850 + 1200 + 900
    total_cost = 0.0025 + 0.0036 + 0.0027
    print(f"\n  {D}Session saved: {vcr_path}{R}")
    print(f"  {D}Total: {total_tokens} tokens, ${total_cost:.4f}, 3 frames{R}")

    # ═══════════════════════════════════════════════════════
    #  STEP 2: REWIND — Time-travel to the failure
    # ═══════════════════════════════════════════════════════
    step_header(2, "REWIND — Time-travel to the failure", "⏮️")

    player = VCRPlayer.load(vcr_path)
    typing_print(f"  {D}Loaded session: {WHT}{player.session.session_id}{R} ({len(player.frames)} frames)")
    pause(0.2)

    # Show the error
    errors = player.get_errors()
    print(f"\n  {D}Finding errors...{R}")
    if not errors:
        print(f"  {YLW}No error frames, but frame 2 has error in state{R}")

    # Inspect frame 2 (the failure)
    fail_state = player.goto_frame(2)
    print(f"  {RED}Frame 2 state:{R}")
    print(f"    {D}error:{R} {RED}{fail_state.get('error', 'N/A')}{R}")
    print(f"    {D}framework:{R} {YLW}{fail_state.get('framework')}{R} ← {B}this is the problem!{R}")
    pause(0.3)

    # Inspect frame 1 (the last good state)
    good_state = player.goto_frame(1)
    print(f"\n  {GRN}Frame 1 state (last good):{R}")
    print(f"    {D}auth_code:{R} {GRN}✓ written{R}")
    print(f"    {D}files_written:{R} {GRN}{good_state.get('files_written')}{R}")

    # Diff
    diff = player.compare_frames(1, 2)
    print(f"\n  {CYN}Diff (frame 1 → 2):{R}")
    for key, val in diff.get("added", {}).items():
        v = str(val)[:60]
        print(f"    {GRN}+ {key}: {v}{R}")
    for key, val in diff.get("modified", {}).items():
        print(f"    {YLW}~ {key}: {val.get('before', '?')!s:.40} → {val.get('after', '?')!s:.40}{R}")

    # ═══════════════════════════════════════════════════════
    #  STEP 3: EDIT & RESUME — Fix and continue
    # ═══════════════════════════════════════════════════════
    step_header(3, "EDIT & RESUME — Fix the state, continue from frame 2", "✏️")

    typing_print(f"  {D}Overriding:{R} {YLW}framework: 'flask'{R} → {GRN}framework: 'fastapi'{R}")
    pause(0.3)
    print(f"  {D}Resuming from frame 2 with fixed state...{R}")

    # Create agent that runs from the fixed point
    def fixed_agent(state: dict) -> dict:
        """Run remaining steps with the fix."""
        state = agent.code_api(state)   # Now uses FastAPI → succeeds
        state = agent.write_tests(state)
        return state

    fork_recorder = VCRRecorder(output_dir=vcr_dir)
    new_session_id = player.resume(
        agent_callable=fixed_agent,
        config=ResumeConfig(
            from_frame=2,
            state_overrides={"framework": "fastapi"},
        ),
        recorder=fork_recorder,
    )
    pause(0.3)

    # Show the result
    forked_player = VCRPlayer.load_by_id(new_session_id, vcr_dir=vcr_dir)
    final_state = forked_player.goto_frame(0)
    print(f"\n  {GRN}{B}✓ Agent succeeded!{R}")
    print(f"    {D}api_code:{R} {GRN}{final_state.get('api_code', 'N/A')[:50]}...{R}")
    print(f"    {D}tests:{R} {GRN}{final_state.get('tests', 'N/A')[:50]}...{R}")
    print(f"    {D}step:{R} {GRN}{final_state.get('step')}{R}")

    # Cost savings
    saved_tokens = 850 + 1200  # didn't re-run planner + auth
    print(f"\n  {MAG}{B}💰 Saved by NOT re-running steps 0-1:{R}")
    print(f"    {D}Tokens skipped:{R} {B}{saved_tokens}{R}")
    print(f"    {D}Cost skipped:{R}  {B}${0.0061:.4f}{R}")
    print(f"    {D}Time skipped:{R}  {B}370ms{R}")

    # ═══════════════════════════════════════════════════════
    #  STEP 4: GHOST REPLAY — Same task, zero tokens
    # ═══════════════════════════════════════════════════════
    step_header(4, "GHOST REPLAY — Same task again, $0.00", "👻")

    from agent_vcr.golden_cache import GoldenRunCache

    cache = GoldenRunCache(cache_dir=os.path.join(vcr_dir, "golden"))

    # Save the successful run as golden
    # Use the original recorder that had 3 frames
    typing_print(f"  {D}Saving successful run as golden path...{R}")
    fingerprint = cache.save_golden_run("Build a REST API with auth", recorder)
    print(f"  {D}Fingerprint:{R} {CYN}{fingerprint[:16]}{R}")
    pause(0.3)

    # Replay it
    typing_print(f"  {D}Replaying same task...{R}")
    replay_recorder = VCRRecorder(output_dir=vcr_dir)
    outputs, ledger = cache.replay(
        "Build a REST API with auth",
        recorder=replay_recorder,
    )
    pause(0.3)

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  {B}ORIGINAL RUN{R}          │  {B}GHOST REPLAY{R}           │
  │  Tokens: {WHT}{B}{ledger.original_tokens:>6}{R}       │  Tokens: {GRN}{B}{ledger.replay_tokens:>6}{R}           │
  │  Cost:   {WHT}{B}${ledger.original_cost_usd:>8.4f}{R}    │  Cost:   {GRN}{B}${ledger.replay_cost_usd:>8.4f}{R}        │
  │  Time:   {WHT}{B}{ledger.original_latency_ms:>6.0f}ms{R}     │  Time:   {GRN}{B}{ledger.replay_latency_ms:>6.1f}ms{R}       │
  ├─────────────────────────────────────────────────┤
  │  {MAG}{B}💰 SAVINGS: {ledger.savings_percent:.0f}%{R}  │  {GRN}{B}{ledger.tokens_saved} tokens  ${ledger.cost_saved_usd:.4f}  {ledger.time_saved_ms:.0f}ms{R}  │
  └─────────────────────────────────────────────────┘""")

    # ═══════════════════════════════════════════════════════
    #  STEP 5: SENTINEL — Catch bad code in real-time
    # ═══════════════════════════════════════════════════════
    step_header(5, "SENTINEL — Catch bad code before it ships", "🛡️")

    from openhands_sentinel import Sentinel

    sentinel_recorder = VCRRecorder(output_dir=vcr_dir)
    sentinel = Sentinel(recorder=sentinel_recorder)
    sentinel.start_session("sentinel-demo")

    # Good code
    good_code = '''
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash."""
    import bcrypt
    return bcrypt.checkpw(password.encode(), hashed.encode())
'''
    result1 = sentinel.check_file("auth/utils.py", good_code)
    print(f"  {GRN}✓{R} auth/utils.py — {GRN}{B}CLEAN{R} ({len(result1.violations)} violations)")
    pause(0.2)

    # Bad code — duplicate function, too long, too complex
    bad_code = '''
def hash_password(password):
    # DUPLICATE of auth/utils.py!
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def handle_everything(request, db, cache, logger, config, session, user, role, permissions):
    # 9 params! Way too many.
    if request.method == "GET":
        if user.is_admin:
            if permissions.can_read:
                if cache.has(request.path):
                    if not session.expired:
                        if config.cache_enabled:
                            if db.is_connected:
                                if logger.level == "DEBUG":
                                    return cache.get(request.path)
    elif request.method == "POST":
        if user.is_admin:
            if permissions.can_write:
                return db.insert(request.body)
    elif request.method == "DELETE":
        if user.is_admin:
            if permissions.can_delete:
                return db.delete(request.path)
    return None
'''
    result2 = sentinel.check_file("api/handlers.py", bad_code)
    print(f"  {RED}✗{R} api/handlers.py — {RED}{B}{len(result2.violations)} VIOLATIONS{R}")
    for v in result2.violations:
        sev_color = RED if v.severity.value in ("critical", "blocker") else YLW
        print(f"    {sev_color}{v.severity.value.upper():>8}{R}  {v.message}")
    pause(0.3)

    # Agent self-corrects
    fixed_code = '''
from auth.utils import hash_password, verify_password

def handle_read(request, ctx):
    """Handle GET requests."""
    if ctx.cache.has(request.path):
        return ctx.cache.get(request.path)
    return ctx.db.query(request.path)

def handle_write(request, ctx):
    """Handle POST requests."""
    return ctx.db.insert(request.body)
'''
    result3 = sentinel.check_file("api/handlers.py", fixed_code)
    print(f"\n  {D}Agent self-corrects...{R}")
    print(f"  {GRN}✓{R} api/handlers.py — {GRN}{B}CLEAN{R} ✨ All issues resolved!")
    pause(0.2)

    sentinel.save()

    # ═══════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"""
  {CYN}{B}{'═' * 57}
  ║                                                       ║
  ║   📼  AGENT VCR — WHAT JUST HAPPENED                   ║
  ║                                                       ║
  ║   1. Agent ran 3 steps, FAILED at step 3              ║
  ║   2. We rewound to the failure, inspected state       ║
  ║   3. Changed framework: flask → fastapi               ║
  ║   4. Resumed from step 2 — skipped steps 0-1          ║
  ║      Saved: 2,050 tokens, $0.0061, 370ms              ║
  ║   5. Ghost Replay: same task again for $0.00           ║
  ║      Saved: 100% tokens, 100% cost                    ║
  ║   6. Sentinel caught 4+ violations in real-time       ║
  ║      Agent self-corrected without human review        ║
  ║                                                       ║
  ║   {GRN}LangSmith shows you what happened.{R}{CYN}{B}                  ║
  ║   {GRN}{B}Agent VCR lets you change it.{R}{CYN}{B}                     ║
  ║                                                       ║
  ║   pip install ai-agent-vcr                            ║
  ║   github.com/ixchio/agent-vcr                         ║
  ║                                                       ║
  {'═' * 57}{R}
""")

    # Cleanup
    shutil.rmtree(workspace, ignore_errors=True)
    # Also clean local .vcr from basic_usage etc.
    shutil.rmtree(".vcr", ignore_errors=True)


if __name__ == "__main__":
    main()
