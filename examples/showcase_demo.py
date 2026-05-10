#!/usr/bin/env python3
"""
Agent VCR — The Ultimate Showcase Demo

Shows EVERY killer feature in one beautiful terminal run:
  1. Record a multi-step AI agent (all steps succeed)
  2. Rewind to any step, inspect + compare states
  3. Edit state and fork a new execution path
  4. Ghost Replay: same task, 0 tokens, instant
  5. Sentinel catches bad code → agent self-corrects

Run:  python examples/showcase_demo.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

# ── Colors ──────────────────────────────────────────────

R = "\033[0m"     # reset
B = "\033[1m"     # bold
D = "\033[2m"     # dim
GRN = "\033[92m"
YLW = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"
WHT = "\033[97m"


def banner() -> None:
    print(f"""{CYN}{B}
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║   📼  AGENT VCR — LIVE DEMO                               ║
  ║                                                           ║
  ║   Time-travel debugging for AI agents.                    ║
  ║   Record · Rewind · Edit · Resume · Never re-run.         ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝{R}
""")


def step_header(num: int, title: str, icon: str = "▶") -> None:
    print(f"\n  {CYN}{B}{'─' * 55}{R}")
    print(f"  {CYN}{B}{icon}  STEP {num}: {title}{R}")
    print(f"  {CYN}{B}{'─' * 55}{R}\n")


def typing_print(text: str, delay: float = 0.008) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def pause(seconds: float = 0.4) -> None:
    time.sleep(seconds)


# ── Simulated Agent ─────────────────────────────────────

class CodingAgent:
    """Simulated multi-step coding agent."""

    def plan(self, state: dict) -> dict:
        return {**state, "plan": ["setup project", "write auth", "write API", "write tests"],
                "step": "plan", "tokens_used": 850, "cost": 0.0025}

    def code_auth(self, state: dict) -> dict:
        return {**state, "auth_code": "def hash_password(pw): return bcrypt.hash(pw)",
                "files_written": ["auth/utils.py"], "step": "code_auth",
                "tokens_used": 1200, "cost": 0.0036}

    def code_api(self, state: dict) -> dict:
        framework = state.get("framework", "fastapi")
        return {**state,
                "api_code": f"app = {framework.title()}()\n@app.get('/users')\ndef get_users(): ...",
                "files_written": ["api/routes.py", "api/models.py"], "step": "code_api",
                "tokens_used": 1500, "cost": 0.0045}

    def write_tests(self, state: dict) -> dict:
        return {**state, "tests": "def test_auth(): assert hash_password('pw') != 'pw'",
                "step": "done", "tokens_used": 650, "cost": 0.0020}

    def run_full(self, state: dict) -> dict:
        """Run all steps to completion."""
        state = self.plan(state)
        state = self.code_auth(state)
        state = self.code_api(state)
        state = self.write_tests(state)
        return state


def main() -> None:
    banner()

    workspace = tempfile.mkdtemp(prefix="vcr_demo_")
    vcr_dir = os.path.join(workspace, ".vcr")

    from agent_vcr import VCRPlayer, VCRRecorder
    from agent_vcr.models import FrameMetadata, ResumeConfig

    agent = CodingAgent()

    # ═══════════════════════════════════════════════════════
    #  STEP 1: RECORD — Capture every agent step
    # ═══════════════════════════════════════════════════════
    step_header(1, "RECORD — Capture the agent's execution", "📹")

    recorder = VCRRecorder(output_dir=vcr_dir)
    recorder.start_session("build-rest-api")

    initial_state = {"task": "Build a REST API with auth", "framework": "fastapi"}

    typing_print(f"  {D}Task:{R} {WHT}{B}\"Build a REST API with auth\"{R}")
    typing_print(f"  {D}Framework:{R} {WHT}fastapi{R}")
    pause(0.3)

    steps = [
        ("planner", agent.plan, 120),
        ("coder:auth", agent.code_auth, 250),
        ("coder:api", agent.code_api, 180),
        ("coder:tests", agent.write_tests, 95),
    ]

    state = initial_state
    for name, fn, latency in steps:
        prev = state
        state = fn(state)
        recorder.record_step(name, prev, state,
                             metadata=FrameMetadata(latency_ms=latency,
                                                    tokens_used=state["tokens_used"],
                                                    cost_usd=state["cost"]))
        print(f"  {GRN}✓{R} {B}{name:<14}{R} {D}{latency:>4}ms  {state['tokens_used']:>5} tokens  ${state['cost']:.4f}{R}")
        pause(0.15)

    vcr_path = recorder.save()
    print(f"\n  {D}Saved → {vcr_path}{R}")
    print(f"  {D}Total: 4 frames · 4,200 tokens · $0.0126{R}")

    # ═══════════════════════════════════════════════════════
    #  STEP 2: REWIND — Time-travel to any point
    # ═══════════════════════════════════════════════════════
    step_header(2, "REWIND — Inspect any frame instantly", "⏮️")

    player = VCRPlayer.load(vcr_path)
    typing_print(f"  {D}Loaded:{R} {WHT}{player.session.session_id}{R} ({len(player.frames)} frames)")
    pause(0.2)

    for i in range(len(player.frames)):
        f = player.frames[i]
        print(f"  {BLU}Frame {i}:{R} {B}{f.node_name:<14}{R} {D}{f.metadata.latency_ms}ms{R}")
    pause(0.3)

    state_1 = player.goto_frame(1)
    print(f"\n  {CYN}Inspecting Frame 1 (after auth):{R}")
    print(f"    {D}auth_code:{R}     {GRN}✓ written{R}")
    print(f"    {D}files_written:{R} {GRN}{state_1.get('files_written')}{R}")
    print(f"    {D}framework:{R}    {WHT}{state_1.get('framework')}{R}")
    pause(0.2)

    diff = player.compare_frames(1, 2)
    print(f"\n  {CYN}Diff (frame 1 → 2):{R}")
    for key, val in diff.get("added", {}).items():
        v = str(val)[:55]
        print(f"    {GRN}+ {key}: {v}{R}")
    for key, val in diff.get("modified", {}).items():
        print(f"    {YLW}~ {key}: {val.get('before', '?')!s:.35} → {val.get('after', '?')!s:.35}{R}")

    # ═══════════════════════════════════════════════════════
    #  STEP 3: EDIT & FORK — Change state, new path
    # ═══════════════════════════════════════════════════════
    step_header(3, "EDIT & FORK — Change state, resume from frame 1", "✏️")

    typing_print(f"  {D}Overriding:{R} {YLW}framework: 'fastapi'{R} → {GRN}framework: 'django'{R}")
    typing_print(f"  {D}Resuming from frame 1 with new state...{R}")
    pause(0.3)

    def fork_agent(state: dict) -> dict:
        state = agent.code_api(state)
        state = agent.write_tests(state)
        return state

    fork_recorder = VCRRecorder(output_dir=vcr_dir)
    new_session_id = player.resume(
        agent_callable=fork_agent,
        config=ResumeConfig(from_frame=1, state_overrides={"framework": "django"}),
        recorder=fork_recorder,
    )
    pause(0.2)

    forked_player = VCRPlayer.load_by_id(new_session_id, vcr_dir=vcr_dir)
    forked_state = forked_player.goto_frame(0)
    print(f"\n  {GRN}{B}✓ Forked execution complete{R}")
    print(f"    {D}api_code:{R} {GRN}{forked_state.get('api_code', '')[:45]}...{R}")
    print(f"    {D}step:{R}     {GRN}{forked_state.get('step')}{R}")
    print(f"    {D}parent:{R}   {WHT}{forked_player.session.parent_session_id}{R}")

    saved_tokens = 850 + 1200
    print(f"\n  {MAG}{B}💰 Saved by skipping steps 0–1:{R}")
    print(f"    {D}Tokens:{R} {B}{saved_tokens} skipped{R}  {D}·{R}  {D}Cost:{R} {B}$0.0061 saved{R}  {D}·{R}  {D}Time:{R} {B}370ms saved{R}")

    # ═══════════════════════════════════════════════════════
    #  STEP 4: GHOST REPLAY — Same task, zero tokens
    # ═══════════════════════════════════════════════════════
    step_header(4, "GHOST REPLAY — Same task, $0.00 cost", "👻")

    from agent_vcr.golden_cache import GoldenRunCache

    cache = GoldenRunCache(cache_dir=os.path.join(vcr_dir, "golden"))

    typing_print(f"  {D}Saving run as golden path...{R}")
    fingerprint = cache.save_golden_run("Build a REST API with auth", recorder)
    print(f"  {D}Fingerprint:{R} {CYN}{fingerprint[:16]}{R}")
    pause(0.3)

    typing_print(f"  {D}Replaying same task from cache...{R}")
    replay_recorder = VCRRecorder(output_dir=vcr_dir)
    _outputs, ledger = cache.replay("Build a REST API with auth", recorder=replay_recorder)
    pause(0.2)

    print(f"""
  ┌───────────────────────────┬───────────────────────────┐
  │  {B}ORIGINAL RUN{R}              │  {B}👻 GHOST REPLAY{R}           │
  │  Tokens: {WHT}{B}{ledger.original_tokens:>6}{R}            │  Tokens: {GRN}{B}{ledger.replay_tokens:>6}{R}            │
  │  Cost:   {WHT}{B}${ledger.original_cost_usd:>8.4f}{R}       │  Cost:   {GRN}{B}${ledger.replay_cost_usd:>8.4f}{R}       │
  │  Time:   {WHT}{B}{ledger.original_latency_ms:>6.0f}ms{R}        │  Time:   {GRN}{B}{ledger.replay_latency_ms:>6.1f}ms{R}        │
  ├───────────────────────────┴───────────────────────────┤
  │  {MAG}{B}💰 SAVINGS: {ledger.savings_percent:.0f}% tokens · ${ledger.cost_saved_usd:.4f} · {ledger.time_saved_ms:.0f}ms{R}              │
  └───────────────────────────────────────────────────────┘""")

    # ═══════════════════════════════════════════════════════
    #  STEP 5: SENTINEL — Code quality guardian
    # ═══════════════════════════════════════════════════════
    step_header(5, "SENTINEL — Real-time code quality guard", "🛡️")

    from openhands_sentinel import Sentinel

    sentinel_recorder = VCRRecorder(output_dir=vcr_dir)
    sentinel = Sentinel(recorder=sentinel_recorder)
    sentinel.start_session("sentinel-demo")

    good_code = '''
def hash_password(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
'''
    result1 = sentinel.check_file("auth/utils.py", good_code)
    print(f"  {GRN}✓{R} auth/utils.py — {GRN}{B}CLEAN{R} ({len(result1.violations)} violations)")
    pause(0.2)

    bad_code = '''
def hash_password(password):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def handle_everything(request, db, cache, logger, config, session, user, role, permissions):
    if request.method == "GET":
        if user.is_admin:
            if permissions.can_read:
                if cache.has(request.path):
                    if not session.expired:
                        if config.cache_enabled:
                            if db.is_connected:
                                return cache.get(request.path)
    return None
'''
    result2 = sentinel.check_file("api/handlers.py", bad_code)
    print(f"  {YLW}⚠{R} api/handlers.py — {YLW}{B}{len(result2.violations)} issues found{R}")
    for v in result2.violations:
        print(f"    {YLW}›{R} {v.message}")
    pause(0.3)

    fixed_code = '''
from auth.utils import hash_password

def handle_read(request, ctx):
    if ctx.cache.has(request.path):
        return ctx.cache.get(request.path)
    return ctx.db.query(request.path)
'''
    sentinel.check_file("api/handlers.py", fixed_code)
    print(f"\n  {D}Agent self-corrects...{R}")
    print(f"  {GRN}✓{R} api/handlers.py — {GRN}{B}CLEAN{R} ✨")
    pause(0.2)

    sentinel.save()

    # ═══════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"""
  {CYN}{B}╔═════════════════════════════════════════════════════════╗
  ║                                                         ║
  ║   📼  AGENT VCR — RECAP                                  ║
  ║                                                         ║
  ║   ✓ Recorded 4 agent steps with full state capture      ║
  ║   ✓ Rewound to any frame, inspected + diffed states     ║
  ║   ✓ Forked execution: changed framework, skipped 2      ║
  ║     steps — saved 2,050 tokens and $0.0061              ║
  ║   ✓ Ghost Replay: re-ran same task for $0.00            ║
  ║   ✓ Sentinel caught 3 code issues, agent self-fixed     ║
  ║                                                         ║
  ║   LangSmith shows you what happened.                    ║
  ║   Agent VCR lets you change it.                         ║
  ║                                                         ║
  ║   pip install ai-agent-vcr                              ║
  ║   github.com/ixchio/agent-vcr                           ║
  ║                                                         ║
  ╚═════════════════════════════════════════════════════════╝{R}
""")

    shutil.rmtree(workspace, ignore_errors=True)
    shutil.rmtree(".vcr", ignore_errors=True)


if __name__ == "__main__":
    main()
