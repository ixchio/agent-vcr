## What and why

<!-- What does this change, and what problem does it solve? Link issues with "Fixes #123". -->

## How it was tested

<!-- Commands you ran and what you checked. For agent/framework integrations, say whether
     you tested against the real tool (version) or against documented sample payloads. -->

## Checklist

- [ ] `make verify` passes (ruff, mypy, unit + e2e tests, benchmarks, build)
- [ ] New behavior has a focused test; bug fixes have a regression test
- [ ] Docs and `CHANGELOG.md` (Unreleased) are updated for user-visible changes
- [ ] No secrets, tokens or private paths in code, tests or recordings
- [ ] Airbag / rollback changes: restore never deletes or overwrites data it did not snapshot
