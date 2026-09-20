# Codabench screenshots

These four are in place and embedded in the note. If a submission is rescored,
replace the PNG and rebuild (`make q6`, or
`python src/report/build_design_note.py`). Any file that is missing renders as a
labelled placeholder box instead, so the build never breaks.

| file | what to capture |
|---|---|
| `mind_submission.png` | the MIND submission's result row on Codabench (status + score) |
| `mind_leaderboard.png` | the MIND leaderboard with our entry visible |
| `ebnerd_submission.png` | the EB-NeRD submission's result row |
| `ebnerd_leaderboard.png` | the EB-NeRD leaderboard with our entry visible |

`ebnerd_detailed_results.png` is also kept here: it is the per-day metric
breakdown behind the EB-NeRD entry, quoted in the note's Q5 section but not
embedded, for length.

The numeric scores live in `CODABENCH_SCORES` at the top of
`src/report/build_design_note.py`, with the per-day AUCs in
`EBNERD_PER_DAY_AUC` just below. Those are the only numbers in the note
transcribed from a screenshot rather than read from a results file.
