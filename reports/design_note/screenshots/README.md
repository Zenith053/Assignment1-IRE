# Codabench screenshots

Drop four PNGs here, then rebuild the note (`make q6`, or
`python src/report/build_design_note.py`). Until a file exists, the note
renders a labelled placeholder box in its place, so the build never breaks.

| file | what to capture |
|---|---|
| `mind_submission.png` | the MIND submission's result row on Codabench (status + score) |
| `mind_leaderboard.png` | the MIND leaderboard with our entry visible |
| `ebnerd_submission.png` | the EB-NeRD submission's result row |
| `ebnerd_leaderboard.png` | the EB-NeRD leaderboard with our entry visible |

The numeric scores go in `CODABENCH_SCORES` at the top of
`src/report/build_design_note.py` (one dict per dataset, `None` until filled).
