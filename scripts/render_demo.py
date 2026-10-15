#!/usr/bin/env python3
"""Render the Agent VCR product demo from a real ACID transaction.

The script creates a disposable Git repository, executes the public
``ACIDWorkspace`` and ``GoldenRunCache`` APIs, verifies byte-for-byte recovery,
and uses the measured timings and hashes in the rendered assets.

Outputs:
    docs/demo.mp4          1920x1080, 30 fps, 18 seconds
    docs/demo.gif          palette-optimized GitHub preview under 5 MiB
    docs/demo.svg          static poster/fallback for the same story
    docs/demo-metrics.json machine-readable evidence captured during rendering
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION_SECONDS = 18

BG = "#070A0F"
SURFACE = "#0D121B"
SURFACE_2 = "#111925"
SURFACE_3 = "#172231"
BORDER = "#263447"
TEXT = "#F5F7FA"
MUTED = "#8C9AAD"
DIM = "#536174"
GREEN = "#43D59B"
GREEN_2 = "#84F0C2"
RED = "#FF5C6C"
RED_2 = "#FF99A3"
AMBER = "#F4C56A"
BLUE = "#65A9FF"
PURPLE = "#A88CFF"


@dataclass(frozen=True)
class DemoMetrics:
    """Evidence captured from the real transaction used by the renderer."""

    generated_at: str
    savepoint_commit: str
    rollback_ms: float
    replay_ms: float
    ci_ms: float
    billing_sha256: str
    env_sha256: str
    restored_billing_sha256: str
    restored_env_sha256: str
    hallucinated_file_removed: bool
    escrow_secret_absent_from_git: bool
    escrow_secret_absent_from_recording: bool
    ghost_steps_replayed: int
    ghost_steps_rerun: int
    ghost_tokens: int
    ghost_cost_usd: float
    ci_passed: int
    ci_risk: str


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_real_metrics() -> DemoMetrics:
    """Exercise ACID rollback and Ghost Replay in a disposable repository."""
    from agent_vcr.datasets import Expectation, ReplayDataset
    from agent_vcr.golden_cache import GoldenRunCache
    from agent_vcr.integrations.openhands import ACIDWorkspace
    from agent_vcr.models import FrameMetadata
    from agent_vcr.recorder import VCRRecorder
    from agent_vcr.regression import run_regression

    with tempfile.TemporaryDirectory(prefix="agent-vcr-demo-") as temp_dir:
        workspace = Path(temp_dir)
        source_dir = workspace / "src"
        source_dir.mkdir()
        billing = source_dir / "billing.py"
        env_file = workspace / ".env"
        hallucinated = workspace / "hallucinated_v2.py"

        original_billing = (
            "from decimal import Decimal\n\n"
            "def charge_total(subtotal: Decimal, tax: Decimal) -> Decimal:\n"
            "    return (subtotal + tax).quantize(Decimal('0.01'))\n"
        )
        original_env = "STRIPE_KEY=sk_test_demo_restored\n"
        billing.write_text(original_billing)
        env_file.write_text(original_env)
        (workspace / ".gitignore").write_text(".env\n.vcr/\n")

        _run(["git", "init", "-b", "main"], cwd=workspace)
        _run(["git", "config", "user.name", "Agent VCR Demo"], cwd=workspace)
        _run(["git", "config", "user.email", "demo@agent-vcr.local"], cwd=workspace)
        _run(["git", "add", ".gitignore", "src/billing.py"], cwd=workspace)
        _run(["git", "commit", "-m", "baseline billing"], cwd=workspace)

        billing_hash = _sha256(billing)
        env_hash = _sha256(env_file)
        recorder = VCRRecorder(output_dir=workspace / ".vcr", auto_save=False)
        acid = ACIDWorkspace(str(workspace), recorder=recorder)
        acid.begin("fix-billing")
        frame = acid.savepoint(
            {"label": "step-2-tested", "tests": "passed"},
            node_name="tests_passed",
        )
        savepoint_commit = str(frame.metadata.custom["acid_commit_hash"])

        billing.write_text(
            "def charge_total(subtotal, tax):\n"
            "    return subtotal * 100  # hallucinated currency conversion\n"
        )
        env_file.write_text("STRIPE_KEY=BROKEN\n")
        hallucinated.write_text("def charge_everything_twice():\n    return True\n")

        rollback_started = time.perf_counter()
        acid.rollback(to_frame_index=0)
        rollback_ms = (time.perf_counter() - rollback_started) * 1000

        restored_billing_hash = _sha256(billing)
        restored_env_hash = _sha256(env_file)
        if restored_billing_hash != billing_hash:
            raise RuntimeError("Tracked billing source was not restored by ACID rollback")
        if restored_env_hash != env_hash:
            raise RuntimeError("Escrowed .env was not restored byte-for-byte")
        if hallucinated.exists():
            raise RuntimeError("Hallucinated file survived ACID rollback")

        git_objects = _run(
            ["git", "cat-file", "--batch-all-objects", "--batch"], cwd=workspace
        ).stdout
        secret_absent_from_git = original_env.encode() not in git_objects
        acid_recording = acid.recorder.save()
        secret_absent_from_recording = original_env.encode() not in acid_recording.read_bytes()
        if not secret_absent_from_git or not secret_absent_from_recording:
            raise RuntimeError("Escrowed .env content leaked into a durable artifact")
        acid.abort()

        recordings_dir = workspace / "recordings"
        golden = VCRRecorder(output_dir=recordings_dir, auto_save=False)
        golden.start_session("billing-golden")
        golden.record_step(
            "planner",
            {"task": "fix billing"},
            {"plan": "validate decimal totals", "tests": "pending"},
            FrameMetadata(
                model="demo-model",
                latency_ms=620,
                tokens_used=1098,
                cost_usd=0.041,
            ),
        )
        golden.record_step(
            "tool:update_billing",
            {"plan": "validate decimal totals", "tests": "pending"},
            {"plan": "validate decimal totals", "tests": "passed"},
            FrameMetadata(
                model="demo-model",
                latency_ms=410,
                tokens_used=744,
                cost_usd=0.042,
            ),
        )
        golden.record_checkpoint(
            {"plan": "validate decimal totals", "tests": "passed"},
            label="step-2-tested",
        )
        golden_path = golden.save()

        cache = GoldenRunCache(cache_dir=str(workspace / "golden"))
        datasets_dir = workspace / "datasets"
        dataset = ReplayDataset.create("billing", datasets_dir=datasets_dir)
        identity: dict[str, Any] = {
            "model": "demo-model",
            "code_commit": savepoint_commit,
            "tools": [{"name": "update_billing", "schema": {"type": "object"}}],
        }
        dataset.add_case(
            "fix billing",
            golden,
            name="billing",
            identity=identity,
            expect=Expectation(
                max_cost_usd=0.10,
                max_tokens=2000,
                require_no_errors=True,
                expect_state={"tests": "passed"},
                require_checkpoints=["step-2-tested"],
            ),
            golden_cache=cache,
        )

        replay_recorder = VCRRecorder(
            output_dir=workspace / "replays",
            auto_save=False,
        )
        replay_started = time.perf_counter()
        _, ledger = cache.replay(
            "fix billing",
            identity=identity,
            recorder=replay_recorder,
        )
        replay_ms = (time.perf_counter() - replay_started) * 1000

        ci_started = time.perf_counter()
        report = run_regression(
            dataset,
            {"billing": str(golden_path)},
            cache,
        )
        ci_ms = (time.perf_counter() - ci_started) * 1000
        if report.failed or report.degraded or report.missing:
            raise RuntimeError("Authentic Agent CI fixture did not pass")

        return DemoMetrics(
            generated_at=datetime.now(timezone.utc).isoformat(),
            savepoint_commit=savepoint_commit,
            rollback_ms=round(rollback_ms, 2),
            replay_ms=round(replay_ms, 2),
            ci_ms=round(ci_ms, 2),
            billing_sha256=billing_hash,
            env_sha256=env_hash,
            restored_billing_sha256=restored_billing_hash,
            restored_env_sha256=restored_env_hash,
            hallucinated_file_removed=not hallucinated.exists(),
            escrow_secret_absent_from_git=secret_absent_from_git,
            escrow_secret_absent_from_recording=secret_absent_from_recording,
            ghost_steps_replayed=ledger.steps_replayed,
            ghost_steps_rerun=ledger.steps_rerun,
            ghost_tokens=ledger.replay_tokens,
            ghost_cost_usd=ledger.replay_cost_usd,
            ci_passed=report.passed,
            ci_risk=report.risk.value,
        )


def _font_path(name: str) -> Path:
    candidates = {
        "sans": [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        ],
        "sans_bold": [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        ],
        "mono": [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")],
        "mono_bold": [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")],
    }
    for candidate in candidates[name]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No local font found for {name}")


class Fonts:
    """Cache local font instances so frame rendering stays fast."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, name: str, size: int) -> ImageFont.FreeTypeFont:
        key = (name, size)
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(str(_font_path(name)), size)
        return self._cache[key]


FONTS = Fonts()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ease(value: float) -> float:
    value = _clamp(value)
    return value * value * (3 - 2 * value)


def _hex_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha


def _gradient_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    pixels = image.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            blue_glow = max(0.0, 1.0 - math.hypot(x - 1560, y - 100) / 1150)
            green_glow = max(0.0, 1.0 - math.hypot(x - 150, y - 980) / 1050)
            pixels[x, y] = (
                7 + int(blue_glow * 4),
                10 + int(green_glow * 6),
                15 + int(blue_glow * 10 + green_glow * 3),
            )
    return image


def _build_base_canvas() -> Image.Image:
    base = _gradient_background().convert("RGBA")
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((66, 130, 1854, 1026), radius=28, fill=(0, 0, 0, 185))
    shadow = shadow.filter(ImageFilter.GaussianBlur(28))
    base.alpha_composite(shadow)

    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((60, 118, 1860, 1008), radius=24, fill=SURFACE, outline=BORDER, width=2)
    draw.rounded_rectangle((60, 118, 1860, 192), radius=24, fill=SURFACE_2)
    draw.rectangle((60, 168, 1860, 192), fill=SURFACE_2)
    draw.line((60, 192, 1860, 192), fill=BORDER, width=2)
    draw.line((1120, 192, 1120, 1008), fill=BORDER, width=2)

    for center_x, color in ((94, "#FF5F57"), (126, "#FEBB2E"), (158, "#28C840")):
        draw.ellipse((center_x - 8, 147 - 8, center_x + 8, 147 + 8), fill=color)

    draw.text((204, 134), "acid/fix-billing", font=FONTS.get("mono", 24), fill=MUTED)
    draw.rounded_rectangle((70, 31, 135, 72), radius=9, fill=TEXT)
    draw.text((82, 40), "VCR", font=FONTS.get("mono_bold", 25), fill=BG)
    draw.text((160, 29), "Agent VCR", font=FONTS.get("sans_bold", 31), fill=TEXT)
    draw.text(
        (160, 67),
        "STATE + FILESYSTEM TRANSACTIONS FOR CODING AGENTS",
        font=FONTS.get("mono", 17),
        fill=MUTED,
    )
    draw.text((1160, 137), "LIVE WORKSPACE INSPECTOR", font=FONTS.get("mono_bold", 20), fill=TEXT)
    return base


BASE_CANVAS = _build_base_canvas()


def _pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    color: str,
    background_alpha: int = 28,
    font_size: int = 17,
) -> int:
    font = FONTS.get("mono_bold", font_size)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 26
    x, y = xy
    draw.rounded_rectangle(
        (x, y, x + width, y + font_size + 20),
        radius=8,
        fill=_hex_rgba(color, background_alpha),
        outline=_hex_rgba(color, 105),
        width=1,
    )
    draw.text((x + 13, y + 8), text, font=font, fill=color)
    return width


def _scene_at(t: float) -> tuple[int, str, str, float]:
    if t < 5:
        return 1, "SAVEPOINT CREATED", GREEN, t / 5
    if t < 10:
        return 2, "AGENT MUTATION DETECTED", RED, (t - 5) / 5
    if t < 14:
        return 3, "ACID ROLLBACK", GREEN, (t - 10) / 4
    return 4, "ZERO-COST GHOST REPLAY", PURPLE, (t - 14) / 4


def _draw_header(draw: ImageDraw.ImageDraw, t: float) -> None:
    scene, label, color, _ = _scene_at(t)
    text = f"0{scene}  /  {label}"
    font = FONTS.get("mono_bold", 18)
    text_width = draw.textbbox((0, 0), text, font=font)[2]
    draw.text((1840 - text_width, 42), text, font=font, fill=color)

    start_x = 1260
    segment_width = 135
    gap = 12
    for index in range(4):
        x = start_x + index * (segment_width + gap)
        draw.rounded_rectangle((x, 82, x + segment_width, 88), radius=3, fill=SURFACE_3)
        segment_start = (0, 5, 10, 14)[index]
        segment_end = (5, 10, 14, 18)[index]
        fill = _clamp((t - segment_start) / (segment_end - segment_start))
        if fill > 0:
            segment_color = (GREEN, RED, GREEN, PURPLE)[index]
            draw.rounded_rectangle(
                (x, 82, x + int(segment_width * fill), 88),
                radius=3,
                fill=segment_color,
            )


def _reveal(text: str, progress: float) -> str:
    return text[: int(len(text) * _clamp(progress))]


def _terminal_lines(t: float, metrics: DemoMetrics) -> list[tuple[str, str, bool]]:
    commit = metrics.savepoint_commit[:8]
    billing_hash = metrics.billing_sha256[:10]
    env_hash = metrics.env_sha256[:10]
    if t < 5:
        progress = t / 5
        lines = [
            ("$ python agent.py", TEXT, False),
            ('workspace = ACIDWorkspace(".")', BLUE, False),
            ('workspace.begin("fix-billing")', BLUE, False),
            ("BRANCH    acid/fix-billing", GREEN, True),
            ("workspace.savepoint(", BLUE, False),
            ('    {"label": "step-2-tested"},', TEXT, False),
            ('    node_name="tests_passed",', TEXT, False),
            (")", BLUE, False),
            (f"SAVEPOINT step-2-tested  {commit}", GREEN_2, True),
        ]
        revealed: list[tuple[str, str, bool]] = []
        for index, (text, color, bold) in enumerate(lines):
            local = (progress * 1.35) - index * 0.105
            revealed.append((_reveal(text, local / 0.22), color, bold))
        return revealed

    if t < 10:
        progress = (t - 5) / 5
        lines = [
            ("agent step 3", MUTED, True),
            ('apply "simplified billing fix"', TEXT, False),
            ("WRITE   src/billing.py        CORRUPTED", RED_2, True),
            ("CREATE  hallucinated_v2.py    UNTRACKED", RED_2, True),
            ("WRITE   .env                   OVERWRITTEN", RED_2, True),
            ("         STRIPE_KEY=BROKEN", AMBER, False),
            ("TEST    billing_flow           FAILED", RED, True),
            ("REPOSITORY INTEGRITY           COMPROMISED", RED, True),
        ]
        revealed = []
        for index, (text, color, bold) in enumerate(lines):
            local = (progress * 1.25) - index * 0.12
            revealed.append((_reveal(text, local / 0.2), color, bold))
        return revealed

    if t < 14:
        progress = (t - 10) / 4
        lines = [
            ("workspace.rollback(to_frame_index=0)", BLUE, False),
            ("# target: step-2-tested", DIM, False),
            (f"RESET    Git savepoint         {commit}", GREEN_2, True),
            ("DELETE   hallucinated_v2.py", GREEN_2, True),
            (f"RESTORE  src/billing.py        sha256:{billing_hash}", GREEN_2, True),
            (f"RESTORE  .env                  sha256:{env_hash}", GREEN_2, True),
            ("         source: private local escrow", MUTED, False),
            (f"ROLLBACK COMPLETE               {metrics.rollback_ms:.2f} ms", GREEN, True),
        ]
        revealed = []
        for index, (text, color, bold) in enumerate(lines):
            local = (progress * 1.35) - index * 0.13
            revealed.append((_reveal(text, local / 0.2), color, bold))
        return revealed

    progress = (t - 14) / 4
    lines = [
        ("$ vcr ci run --dataset billing \\", TEXT, False),
        ("    --candidates .vcr/candidates", TEXT, False),
        ("DATASET billing                 1 case", MUTED, True),
        (f"GHOST CACHE HIT                 {metrics.ghost_steps_replayed} steps", PURPLE, True),
        ("billing                         PASS", GREEN_2, True),
        ("0 LLM calls  /  0 tokens  /  $0.00", GREEN, True),
        (f"REPLAY {metrics.replay_ms:.2f} ms  /  RISK {metrics.ci_risk}", TEXT, True),
        ("VERIFIED RUN REUSED. NO MODEL CALL.", PURPLE, True),
    ]
    revealed = []
    for index, (text, color, bold) in enumerate(lines):
        local = (progress * 1.35) - index * 0.13
        revealed.append((_reveal(text, local / 0.2), color, bold))
    return revealed


def _draw_terminal(draw: ImageDraw.ImageDraw, t: float, metrics: DemoMetrics) -> None:
    draw.text((100, 225), "TERMINAL", font=FONTS.get("mono_bold", 18), fill=DIM)
    _pill(draw, (920, 216), "ACID", color=GREEN, font_size=15)
    draw.line((96, 266, 1083, 266), fill=BORDER, width=1)

    y = 304
    for text, color, bold in _terminal_lines(t, metrics):
        if not text:
            y += 54
            continue
        font = FONTS.get("mono_bold" if bold else "mono", 27)
        draw.text((102, y), text, font=font, fill=color)
        y += 57

    cursor_alpha = int(90 + 150 * abs(math.sin(t * math.pi * 2)))
    draw.rectangle((103, 930, 119, 962), fill=_hex_rgba(GREEN, cursor_alpha))
    draw.text(
        (139, 933),
        "recording locally  /  no hosted account",
        font=FONTS.get("mono", 18),
        fill=DIM,
    )


def _metric_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    value_color: str,
) -> None:
    draw.rounded_rectangle((x, y, x + width, y + 88), radius=12, fill=SURFACE_2, outline=BORDER)
    draw.text((x + 18, y + 14), label, font=FONTS.get("mono", 14), fill=DIM)
    draw.text((x + 18, y + 43), value, font=FONTS.get("mono_bold", 21), fill=value_color)


def _file_row(
    draw: ImageDraw.ImageDraw,
    y: int,
    name: str,
    status: str,
    color: str,
    *,
    dimmed: bool = False,
) -> None:
    x = 1158
    row_color = SURFACE if dimmed else SURFACE_2
    draw.rounded_rectangle((x, y, 1822, y + 52), radius=10, fill=row_color)
    draw.rectangle((x + 16, y + 17, x + 30, y + 35), fill=_hex_rgba(color, 210))
    name_color = DIM if dimmed else TEXT
    draw.text((x + 45, y + 13), name, font=FONTS.get("mono", 19), fill=name_color)
    font = FONTS.get("mono_bold", 14)
    status_width = draw.textbbox((0, 0), status, font=font)[2]
    draw.text((1798 - status_width, y + 16), status, font=font, fill=color)


def _timeline_node(
    draw: ImageDraw.ImageDraw,
    y: int,
    number: str,
    title: str,
    detail: str,
    color: str,
    *,
    active: bool = False,
) -> None:
    center_x = 1180
    radius = 13 if active else 9
    if active:
        draw.ellipse(
            (center_x - 20, y - 20, center_x + 20, y + 20),
            fill=_hex_rgba(color, 35),
        )
    draw.ellipse(
        (center_x - radius, y - radius, center_x + radius, y + radius),
        fill=color,
    )
    draw.text((1210, y - 18), number, font=FONTS.get("mono_bold", 14), fill=color)
    draw.text((1255, y - 22), title, font=FONTS.get("sans_bold", 20), fill=TEXT)
    draw.text((1255, y + 5), detail, font=FONTS.get("mono", 14), fill=MUTED)


def _draw_inspector(draw: ImageDraw.ImageDraw, t: float, metrics: DemoMetrics) -> None:
    scene, _, color, progress = _scene_at(t)
    status = ("PROTECTED", "DRIFT", "RESTORING", "VERIFIED")[scene - 1]
    _pill(draw, (1740, 130), status, color=color, font_size=13)

    if scene == 1:
        tokens = int(1842 * _ease(progress))
        cost = 0.083 * _ease(progress)
        token_color = TEXT
        cost_color = GREEN
    elif scene == 2:
        tokens = int(1842 + (2934 - 1842) * _ease(progress))
        cost = 0.083 + (0.126 - 0.083) * _ease(progress)
        token_color = RED_2
        cost_color = RED_2
    elif scene == 3:
        tokens = 2934
        cost = 0.126
        token_color = MUTED
        cost_color = MUTED
    else:
        tokens = int(2934 * (1 - _ease(min(progress * 1.6, 1))))
        cost = 0.126 * (1 - _ease(min(progress * 1.6, 1)))
        token_color = GREEN if tokens == 0 else MUTED
        cost_color = GREEN if cost < 0.0005 else MUTED

    _metric_card(draw, 1158, 218, 200, "TOKENS", f"{tokens:,}", token_color)
    _metric_card(draw, 1372, 218, 205, "LLM COST", f"${cost:.3f}", cost_color)
    _metric_card(draw, 1591, 218, 231, "BRANCH", "acid/fix-billing", BLUE)

    draw.text((1158, 330), "LIVE FILE TREE", font=FONTS.get("mono_bold", 16), fill=DIM)
    if scene == 1:
        _file_row(draw, 364, "src/billing.py", "CLEAN", GREEN)
        _file_row(draw, 424, ".env", "ESCROWED", PURPLE)
        _file_row(draw, 484, "tests/test_billing.py", "PASS", GREEN)
    elif scene == 2:
        _file_row(draw, 364, "src/billing.py", "CORRUPTED", RED)
        _file_row(draw, 424, ".env", "OVERWRITTEN", RED)
        _file_row(draw, 484, "hallucinated_v2.py", "NEW", RED)
        _file_row(draw, 544, "tests/test_billing.py", "FAIL", RED)
    elif scene == 3:
        restored = progress > 0.48
        _file_row(
            draw,
            364,
            "src/billing.py",
            "RESTORED" if restored else "RESETTING",
            GREEN if restored else AMBER,
        )
        _file_row(
            draw,
            424,
            ".env",
            "BYTE-EXACT" if progress > 0.68 else "FROM ESCROW",
            GREEN if progress > 0.68 else PURPLE,
        )
        if progress < 0.68:
            _file_row(draw, 484, "hallucinated_v2.py", "DELETING", RED, dimmed=True)
        else:
            _file_row(draw, 484, "hallucinated_v2.py", "DELETED", GREEN, dimmed=True)
        _file_row(draw, 544, "tests/test_billing.py", "PASS", GREEN)
    else:
        _file_row(draw, 364, "src/billing.py", "VERIFIED", GREEN)
        _file_row(draw, 424, ".env", "ESCROWED", PURPLE)
        _file_row(draw, 484, "tests/test_billing.py", "PASS", GREEN)

    timeline_y = 620 if scene in (2, 3) else 570
    draw.text((1158, timeline_y), "SAVEPOINT TIMELINE", font=FONTS.get("mono_bold", 16), fill=DIM)
    line_x = 1180
    draw.line((line_x, timeline_y + 52, line_x, timeline_y + 245), fill=BORDER, width=3)
    _timeline_node(
        draw,
        timeline_y + 55,
        "01",
        "Transaction begin",
        "acid/fix-billing",
        GREEN,
    )
    _timeline_node(
        draw,
        timeline_y + 130,
        "02",
        "step-2-tested",
        metrics.savepoint_commit[:8],
        GREEN,
        active=scene in (1, 3, 4),
    )
    if scene == 2:
        _timeline_node(
            draw,
            timeline_y + 205,
            "03",
            "Uncommitted mutation",
            "integrity drift detected",
            RED,
            active=True,
        )
    elif scene == 3:
        _timeline_node(
            draw,
            timeline_y + 205,
            "RB",
            "Rollback",
            f"completed in {metrics.rollback_ms:.2f} ms",
            GREEN,
            active=True,
        )
    elif scene == 4:
        _timeline_node(
            draw,
            timeline_y + 205,
            "CI",
            "Ghost Replay",
            f"{metrics.ghost_steps_replayed} cached steps / $0.00",
            PURPLE,
            active=True,
        )

    integrity_y = 932
    draw.text((1158, integrity_y), "INTEGRITY", font=FONTS.get("mono_bold", 14), fill=DIM)
    hash_color = RED if scene == 2 else GREEN
    hash_text = "DRIFT" if scene == 2 else metrics.env_sha256[:12]
    draw.text(
        (1282, integrity_y - 1),
        f".env sha256:{hash_text}",
        font=FONTS.get("mono", 14),
        fill=hash_color,
    )


def _draw_scene_pulse(frame: Image.Image, t: float) -> None:
    scene, _, color, progress = _scene_at(t)
    if scene not in (2, 3, 4):
        return
    pulse = max(0.0, 1.0 - progress * 4.2)
    if pulse <= 0:
        return
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (1121, 193, 1859, 1007),
        radius=22,
        fill=_hex_rgba(color, int(34 * pulse)),
        outline=_hex_rgba(color, int(185 * pulse)),
        width=4,
    )
    frame.alpha_composite(overlay)


def render_frame(t: float, metrics: DemoMetrics) -> Image.Image:
    frame = BASE_CANVAS.copy()
    draw = ImageDraw.Draw(frame, "RGBA")
    _draw_header(draw, t)
    _draw_terminal(draw, t, metrics)
    _draw_inspector(draw, t, metrics)
    _draw_scene_pulse(frame, t)

    if t < 0.6:
        opacity = int(255 * (1 - _ease(t / 0.6)))
        frame.alpha_composite(Image.new("RGBA", frame.size, (7, 10, 15, opacity)))
    if t > 17.2:
        opacity = int(155 * _ease((t - 17.2) / 0.8))
        overlay = Image.new("RGBA", frame.size, (7, 10, 15, opacity))
        frame.alpha_composite(overlay)
        final_draw = ImageDraw.Draw(frame, "RGBA")
        reveal = _ease((t - 17.2) / 0.45)
        line = "THE UNDO BUTTON FOR CODING AGENTS"
        font = FONTS.get("sans_bold", 32)
        width = final_draw.textbbox((0, 0), line, font=font)[2]
        final_draw.text(
            ((WIDTH - width) / 2, 505),
            line,
            font=font,
            fill=_hex_rgba(TEXT, int(255 * reveal)),
        )
    return frame.convert("RGB")


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required to render the demo")
    return executable


def render_mp4(path: Path, metrics: DemoMetrics) -> None:
    ffmpeg = _ffmpeg_executable()
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Unable to open ffmpeg frame pipe")
    try:
        for frame_number in range(FPS * DURATION_SECONDS):
            frame = render_frame(frame_number / FPS, metrics)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding demo.mp4")


def render_gif(mp4_path: Path, gif_path: Path) -> None:
    ffmpeg = _ffmpeg_executable()
    profiles = [
        (960, 12, 128, 5),
        (900, 10, 112, 5),
        (800, 10, 96, 6),
    ]
    maximum_bytes = 5 * 1024 * 1024
    for width, fps, colors, dither_scale in profiles:
        filter_graph = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
            f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
            f"[b][p]paletteuse=dither=bayer:bayer_scale={dither_scale}:diff_mode=rectangle"
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4_path),
                "-filter_complex",
                filter_graph,
                "-loop",
                "0",
                str(gif_path),
            ],
            check=True,
        )
        if gif_path.stat().st_size < maximum_bytes:
            return
    raise RuntimeError(f"Optimized GIF is still over 5 MiB: {gif_path.stat().st_size}")


def render_svg(path: Path, metrics: DemoMetrics) -> None:
    commit = escape(metrics.savepoint_commit[:8])
    rollback = f"{metrics.rollback_ms:.2f} ms"
    replay = f"{metrics.replay_ms:.2f} ms"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080" role="img" aria-labelledby="title desc">
<title id="title">Agent VCR rollback and Ghost Replay</title>
<desc id="desc">A dark split-screen showing a recovered billing workspace and a zero-cost replay.</desc>
<defs>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="20" stdDeviation="26" flood-opacity=".45"/></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#070A0F"/><stop offset="1" stop-color="#0B1420"/></linearGradient>
</defs>
<rect width="1920" height="1080" fill="url(#bg)"/>
<rect x="70" y="31" width="65" height="41" rx="9" fill="#F5F7FA"/>
<text x="82" y="59" fill="#070A0F" font-family="monospace" font-size="25" font-weight="700">VCR</text>
<text x="160" y="57" fill="#F5F7FA" font-family="sans-serif" font-size="31" font-weight="700">Agent VCR</text>
<text x="160" y="84" fill="#8C9AAD" font-family="monospace" font-size="17">STATE + FILESYSTEM TRANSACTIONS FOR CODING AGENTS</text>
<rect x="60" y="118" width="1800" height="890" rx="24" fill="#0D121B" stroke="#263447" stroke-width="2" filter="url(#shadow)"/>
<rect x="60" y="118" width="1800" height="74" rx="24" fill="#111925"/>
<rect x="60" y="168" width="1800" height="24" fill="#111925"/>
<line x1="1120" y1="192" x2="1120" y2="1008" stroke="#263447" stroke-width="2"/>
<text x="102" y="244" fill="#536174" font-family="monospace" font-size="18" font-weight="700">TERMINAL</text>
<g font-family="monospace" font-size="27">
  <text x="102" y="326" fill="#65A9FF">workspace.rollback(to_frame_index=0)</text>
  <text x="102" y="390" fill="#84F0C2">RESET    Git savepoint         {commit}</text>
  <text x="102" y="454" fill="#84F0C2">DELETE   hallucinated_v2.py</text>
  <text x="102" y="518" fill="#84F0C2">RESTORE  src/billing.py        sha256:{metrics.billing_sha256[:10]}</text>
  <text x="102" y="582" fill="#84F0C2">RESTORE  .env                  sha256:{metrics.env_sha256[:10]}</text>
  <text x="102" y="646" fill="#8C9AAD">         source: private local escrow</text>
  <text x="102" y="730" fill="#43D59B" font-weight="700">ROLLBACK COMPLETE               {rollback}</text>
  <text x="102" y="830" fill="#F5F7FA">$ vcr ci run --dataset billing --candidates .vcr/candidates</text>
  <text x="102" y="894" fill="#A88CFF" font-weight="700">GHOST CACHE HIT  /  0 LLM calls  /  $0.00  /  {replay}</text>
</g>
<text x="1160" y="158" fill="#F5F7FA" font-family="monospace" font-size="20" font-weight="700">LIVE WORKSPACE INSPECTOR</text>
<g font-family="monospace">
  <text x="1160" y="254" fill="#536174" font-size="16">LIVE FILE TREE</text>
  <rect x="1158" y="282" width="664" height="58" rx="10" fill="#111925"/>
  <text x="1204" y="319" fill="#F5F7FA" font-size="20">src/billing.py</text><text x="1690" y="319" fill="#43D59B" font-size="15" font-weight="700">RESTORED</text>
  <rect x="1158" y="354" width="664" height="58" rx="10" fill="#111925"/>
  <text x="1204" y="391" fill="#F5F7FA" font-size="20">.env</text><text x="1670" y="391" fill="#43D59B" font-size="15" font-weight="700">BYTE-EXACT</text>
  <rect x="1158" y="426" width="664" height="58" rx="10" fill="#0D121B"/>
  <text x="1204" y="463" fill="#536174" font-size="20">hallucinated_v2.py</text><text x="1710" y="463" fill="#43D59B" font-size="15" font-weight="700">DELETED</text>
  <text x="1160" y="560" fill="#536174" font-size="16">VERIFIED OUTCOME</text>
  <rect x="1158" y="590" width="664" height="260" rx="16" fill="#111925" stroke="#263447"/>
  <text x="1194" y="644" fill="#43D59B" font-size="17" font-weight="700">ACID ROLLBACK</text>
  <text x="1194" y="692" fill="#F5F7FA" font-size="28" font-weight="700">{rollback}</text>
  <text x="1194" y="754" fill="#A88CFF" font-size="17" font-weight="700">GHOST REPLAY</text>
  <text x="1194" y="802" fill="#F5F7FA" font-size="28" font-weight="700">0 calls  /  $0.00  /  PASS</text>
  <text x="1160" y="944" fill="#43D59B" font-size="18" font-weight="700">PROTECTED  /  VERIFIED  /  LOCAL-FIRST</text>
</g>
</svg>
"""
    path.write_text(svg)


def write_metrics(path: Path, metrics: DemoMetrics) -> None:
    path.write_text(json.dumps(asdict(metrics), indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs",
        help="Directory for demo.mp4, demo.gif, demo.svg, and metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = capture_real_metrics()

    mp4_path = output_dir / "demo.mp4"
    gif_path = output_dir / "demo.gif"
    svg_path = output_dir / "demo.svg"
    metrics_path = output_dir / "demo-metrics.json"

    write_metrics(metrics_path, metrics)
    render_mp4(mp4_path, metrics)
    render_gif(mp4_path, gif_path)
    render_svg(svg_path, metrics)

    print(json.dumps(asdict(metrics), indent=2))
    print(f"MP4: {mp4_path} ({mp4_path.stat().st_size:,} bytes)")
    print(f"GIF: {gif_path} ({gif_path.stat().st_size:,} bytes)")
    print(f"SVG: {svg_path} ({svg_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
