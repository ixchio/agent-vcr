"""Produce candidate recordings for Agent CI from golden runs (demo helper).

For each case in a dataset, Ghost-replays the golden run (FULL mode, zero
LLM cost) into ``<candidates-dir>/<case-name>.vcr`` — i.e. "current code
behaves exactly like the baseline". With ``--break-case NAME`` the named
candidate drops its final frame, simulating a regression for demos.

Usage:
    python examples/agent_ci_candidates.py --dataset refunds
    python examples/agent_ci_candidates.py --dataset refunds --break-case "refund order"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_vcr.datasets import ReplayDataset
from agent_vcr.golden_cache import GoldenRunCache
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets-dir", default=".vcr/datasets")
    parser.add_argument("--golden-dir", default=".vcr/golden")
    parser.add_argument("--candidates-dir", default=".vcr/candidates")
    parser.add_argument("--break-case", default=None)
    args = parser.parse_args()

    ds = ReplayDataset.load(args.dataset, datasets_dir=args.datasets_dir)
    cache = GoldenRunCache(cache_dir=args.golden_dir)
    out = Path(args.candidates_dir)
    out.mkdir(parents=True, exist_ok=True)

    for case in ds.cases():
        info = cache.get_golden_info(case.task, case.identity)
        if info is None:
            raise SystemExit(f"golden missing for case {case.name!r}")
        player = VCRPlayer.load(info["file"])
        frames = player.frames
        if case.name == args.break_case:
            frames = frames[:-1]  # simulate a truncated/regressed run

        rec = VCRRecorder(output_dir=str(out), auto_save=False)
        rec.start_session(session_id=case.name)
        for i, frame in enumerate(frames):
            from agent_vcr.models import StateSerializer

            rec.record_step(
                node_name=frame.node_name,
                input_state=player.get_input_state(i),
                output_state=StateSerializer.deserialize(frame.output_state),
                metadata=frame.metadata,
                frame_type=frame.frame_type,
            )
        path = rec.save()
        # Rename to <case-name>.vcr so `vcr ci run` picks it up.
        target = out / f"{case.name}.vcr"
        if path != target:
            target.write_bytes(path.read_bytes())
            path.unlink()
        print(f"candidate: {target}")


if __name__ == "__main__":
    main()
