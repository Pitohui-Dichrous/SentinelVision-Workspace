# SentinelVision workspace rules

- Read `CODEX_HANDOFF_CN.md` and `GIT_GUIDE_CN.md` before continuing work from a new Codex account or computer.
- Treat `RESULTS` as manually approved production models. Never deploy a training output automatically.
- Keep datasets, portable runtimes, downloaded tools, model binaries and training outputs out of Git. The root `.gitignore` defines the protected boundary.
- Before editing, inspect the repository status with the portable Git executable in `TOOLS/Git/cmd/git.exe` when available.
- Preserve unrelated user changes. Commit only the files that belong to the requested task unless the user explicitly requests a full workspace snapshot.
- After an implemented change, run proportionate checks. Create a Git commit only when the user explicitly requests it; otherwise leave verified changes uncommitted. Push only when the user explicitly requests synchronization.
- All paths must remain relative to the workspace root; never hard-code the removable drive letter in application code.
- New detection models must remain opt-in through the dynamic `RESULTS` scan and model check boxes.
- Preserve the compatibility baseline for safety-pipeline research. Temporal PPE processing must remain explicitly switchable and must not silently replace old persisted behavior.
- Keep runtime events, experiment traces, hard samples and health logs below `.runtime` so they remain outside Git.
