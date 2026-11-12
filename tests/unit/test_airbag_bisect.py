"""`vcr bisect`: find the agent action that broke the tests, in temporary checkouts.

Real git, real subprocesses, temporary directories only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from agent_vcr import cli
from agent_vcr.airbag.bisect import bisect, timeline
from agent_vcr.airbag.core import Airbag, Policy, ToolCall

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

PY = sys.executable
#: Passes while calc.add still adds.
ADD_TEST = [PY, "-c", "from calc import add; raise SystemExit(0 if add(2, 2) == 4 else 1)"]


@pytest.fixture(autouse=True)
def _airbag_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "airbag-home"))
    for var in ("AGENT_VCR_AIRBAG", "AGENT_VCR_AIRBAG_MODE", "AGENT_VCR_AIRBAG_ON_IRREVERSIBLE"):
        monkeypatch.delenv(var, raising=False)
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return temp


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=env, encoding="utf-8", errors="replace").stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    _git(proj, "init", "-q", "-b", "main")
    _git(proj, "config", "user.email", "u@example.com")
    _git(proj, "config", "user.name", "u")
    (proj / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (proj / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    _git(proj, "add", "-A")
    _git(proj, "commit", "-qm", "init")
    return proj


def _agent(bag: Airbag, proj: Path, command: str) -> None:
    bag.before(ToolCall(agent="claude", tool_name="Bash", kind="shell", cwd=str(proj),
                        command=command))
    subprocess.run(command, shell=True, cwd=proj, check=True)


def _break_it(proj: Path) -> Airbag:
    """Five actions; the third one (snapshot #3) breaks add()."""
    bag = Airbag(proj, policy=Policy())
    _agent(bag, proj, "echo notes > NOTES.md")
    _agent(bag, proj, "echo 'X = 1' > util.py")
    _agent(bag, proj, "sed -i.bak 's/a + b/a - b/' calc.py && rm calc.py.bak")
    _agent(bag, proj, "echo more >> NOTES.md")
    _agent(bag, proj, "rm util.py")
    return bag


def _refs(bag: Airbag) -> dict[int, str]:
    return {s.seq: s.commit for s in bag.store.snapshots()}


def test_finds_the_breaking_action_without_touching_project_or_refs(
        project: Path, _airbag_home: Path) -> None:
    bag = _break_it(project)
    before_files = {p.name: p.read_bytes() for p in project.iterdir() if p.is_file()}
    before_refs = _refs(bag)
    user_head = _git(project, "rev-parse", "HEAD")

    report = bisect(bag, ADD_TEST)

    assert report.status == "found"
    assert report.culprit is not None and "sed" in report.culprit["command"]
    assert report.culprit["agent"] == "claude"
    assert (report.before, report.after) == (3, 4)
    assert report.changes == [("M", "calc.py")]
    assert report.rewind == "vcr rewind 3"
    assert report.test_runs <= 4  # 6 states: both ends + log2 search
    # Read-only: same files, same old refs (one extra snapshot of "now"), same user repo.
    assert {p.name: p.read_bytes() for p in project.iterdir() if p.is_file()} == before_files
    after_refs = _refs(bag)
    assert {k: v for k, v in after_refs.items() if k in before_refs} == before_refs
    assert len(after_refs) - len(before_refs) <= 1
    assert _git(project, "rev-parse", "HEAD") == user_head
    assert not list(_airbag_home.iterdir())  # temporary checkouts removed


def test_rewind_command_goes_to_just_before_the_culprit(project: Path) -> None:
    bag = _break_it(project)
    report = bisect(bag, ADD_TEST)
    assert report.before is not None
    bag.rewind(report.before)
    assert "a + b" in (project / "calc.py").read_text(encoding="utf-8")
    assert (project / "util.py").exists()  # the state right before action #3


def test_no_actions(project: Path) -> None:
    report = bisect(Airbag(project, policy=Policy()), ADD_TEST)
    assert report.status == "no-actions"
    assert report.runs == []


def test_already_failing_at_the_good_end(project: Path) -> None:
    bag = _break_it(project)
    report = bisect(bag, [PY, "-c", "raise SystemExit(3)"])
    assert report.status == "good-fails"
    assert report.culprit is None


def test_passing_at_the_bad_end(project: Path) -> None:
    bag = _break_it(project)
    report = bisect(bag, [PY, "-c", "pass"])
    assert report.status == "bad-passes"
    assert report.test_runs == 1


def test_good_and_bad_bounds(project: Path) -> None:
    bag = _break_it(project)
    report = bisect(bag, ADD_TEST, good=2, bad=5)
    assert report.status == "found"
    assert (report.good, report.bad, report.before, report.after) == (2, 5, 3, 4)
    assert bisect(bag, ADD_TEST, good=4, bad=5).status == "good-fails"
    assert bisect(bag, ADD_TEST, good=99).status == "invalid"
    assert bisect(bag, ADD_TEST, good=4, bad=2).status == "invalid"


def test_timeout_counts_as_failure(project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    _agent(bag, project, "echo a > a.txt")
    _agent(bag, project, "touch slow")
    _agent(bag, project, "echo b > b.txt")
    slow_test = [PY, "-c", "import os, time\nif os.path.exists('slow'): time.sleep(30)"]
    report = bisect(bag, slow_test, timeout=1.0)
    assert report.status == "found"
    assert report.culprit is not None and report.culprit["command"] == "touch slow"
    assert report.timeouts >= 1
    assert any(r.outcome == "timeout" for r in report.runs)


def test_copy_ignored_symlinks_dependencies(project: Path) -> None:
    deps = project / "node_modules" / "left-pad"
    deps.mkdir(parents=True)
    (deps / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    bag = _break_it(project)
    needs_deps = [PY, "-c", "import os\nfrom calc import add\n"
                  "ok = os.path.exists('node_modules/left-pad/index.js') and add(2, 2) == 4\n"
                  "raise SystemExit(0 if ok else 1)"]
    assert bisect(bag, needs_deps).status == "good-fails"  # deps are not in snapshots
    report = bisect(bag, needs_deps, copy_ignored=True)
    assert report.status == "found" and report.before == 3
    assert (deps / "index.js").read_text(
        encoding="utf-8") == "module.exports = 1\n"  # cleanup kept the real dir


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell background job")
def test_background_child_holding_stdout_is_not_a_timeout(project: Path) -> None:
    bag = _break_it(project)
    start = time.monotonic()
    # Passes immediately but leaves `sleep 30` running with the test's stdout.
    report = bisect(bag, ["sleep 30 & echo started"], timeout=10.0)
    assert report.status == "bad-passes"
    assert report.runs[0].outcome == "pass" and "started" in report.runs[0].output
    assert report.timeouts == 0
    assert time.monotonic() - start < 9


def test_copy_ignored_links_dependencies_not_build_output(project: Path) -> None:
    (project / ".gitignore").write_text("node_modules/\ndist/\n", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "dist").mkdir()
    (project / "dist" / "app.js").write_text("built\n", encoding="utf-8")
    bag = _break_it(project)
    # The test wipes dist/ in its checkout: it must never reach the real project's dist/.
    wipe = [PY, "-c", "import os, shutil\n"
            "assert os.path.islink('node_modules')\n"
            "shutil.rmtree('dist', ignore_errors=True)\n"
            "os.makedirs('dist', exist_ok=True)\n"
            "open('dist/app.js', 'w').write('clobbered')\n"
            "from calc import add\nraise SystemExit(0 if add(2, 2) == 4 else 1)"]
    report = bisect(bag, wipe, copy_ignored=True)
    assert report.status == "found" and report.before == 3
    assert (project / "dist" / "app.js").read_text(encoding="utf-8") == "built\n"

    probe = [PY, "-c", "import os\nraise SystemExit(0 if os.path.islink('dist') else 1)"]
    assert bisect(bag, probe, copy_ignored=True).status == "good-fails"
    assert bisect(bag, probe, extra_links=["dist"]).status == "bad-passes"  # named explicitly
    with pytest.raises(ValueError):
        bisect(bag, probe, extra_links=["../elsewhere"])


def test_cli_copy_ignored_with_extra_dir(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (project / ".gitignore").write_text("node_modules/\ndist/\n", encoding="utf-8")
    (project / "dist").mkdir()
    _break_it(project)
    probe = [PY, "-c", "import os\nraise SystemExit(0 if os.path.islink('dist') else 1)"]
    with pytest.raises(SystemExit) as exc:
        cli.main(["bisect", "--project-dir", str(project), "--json", "--copy-ignored",
                  "--copy-ignored", "dist", "--", *probe])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "bad-passes"
    with pytest.raises(SystemExit) as exc:
        cli.main(["bisect", "--project-dir", str(project), "--copy-ignored", "../x", "--",
                  *probe])
    assert exc.value.code == 2


def test_single_string_runs_through_the_shell(project: Path) -> None:
    bag = _break_it(project)
    report = bisect(bag, ["test -f calc.py && grep -q 'a + b' calc.py"])
    assert report.status == "found" and report.before == 3


def test_checkout_runs_with_snapshot_env_and_outside_project(project: Path) -> None:
    bag = _break_it(project)
    probe = [PY, "-c", "import os, sys\n"
             f"sys.exit(0 if not os.getcwd().startswith({str(project)!r}) and "
             "os.environ.get('VCR_BISECT_SNAPSHOT') and 'GIT_DIR' not in os.environ else 1)"]
    assert bisect(bag, probe).status == "bad-passes"  # every run passed the probe


def test_timeline_includes_restores(project: Path) -> None:
    bag = _break_it(project)
    bag.undo()
    kinds = [step.kind for step in timeline(bag)]
    assert kinds[-1] == "restore" and kinds.count("action") == 5


def test_cli_json_and_exit_codes(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _break_it(project)
    with pytest.raises(SystemExit) as exc:
        cli.main(["bisect", "--project-dir", str(project), "--json", "--", *ADD_TEST])
    assert exc.value.code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "found" and data["before"] == 3
    assert data["changes"] == [{"status": "M", "path": "calc.py"}]
    assert data["rewind"].startswith("vcr rewind 3")

    with pytest.raises(SystemExit) as exc:
        cli.main(["bisect", "--project-dir", str(project), "--", PY, "-c", "pass"])
    assert exc.value.code == 1
    assert "passes now" in capsys.readouterr().out


def test_cli_human_report(project: Path, capsys: pytest.CaptureFixture[str],
                          monkeypatch: pytest.MonkeyPatch) -> None:
    _break_it(project)
    monkeypatch.chdir(project)
    cli.main(["bisect", "--", *ADD_TEST])
    out = capsys.readouterr().out
    assert "First bad action: sed" in out
    assert "~ calc.py" in out
    assert "Go back to just before it:  vcr rewind 3" in out
    assert "--project-dir" not in out


def test_cli_usage_errors(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["bisect", "--project-dir", str(project)])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["bisect", "--project-dir", str(project), "--good", "x", "--", "true"])
    assert exc.value.code == 2
