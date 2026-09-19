# tools/

`sweep_pooling_k.py` sweeps `k` for the semantic scorer's top-k similarity
pooling on the val split, and writes `reports/sweep_pooling_k_<dataset>_val.json`
(AUC per k, paired bootstrap CIs, and a breakdown by history-length quartile):

```bash
make sweep     # both datasets
.venv/bin/python tools/sweep_pooling_k.py --config config/mind.yaml \
    --out reports/sweep_pooling_k_mind_val.json
```

k=1 is max pooling and k >= |history| is exactly mean pooling, so the sweep also
emits a `mean_pool_reference` that its largest k must reproduce. Analysis is in
`reports/design_note.md` (§4, *Pooling the history*).

`evaluate_official.py` is Microsoft's scorer, vendored unmodified from
<https://github.com/msnews/MIND/blob/master/evaluate.py> so results can be
reproduced without network access.

Run it against a MIND submission whose split has public labels
(MINDlarge_test does not):

```bash
mkdir -p /tmp/b/ref /tmp/b/res /tmp/out

# truth.txt: "impid [labels]" - no spaces inside the brackets, parse_line splits
# the line on whitespace and json-decodes the second field.
python - <<'PY'
src = 'data/raw/mind/MINDsmall_dev/behaviors.tsv'
with open(src) as fh, open('/tmp/b/ref/truth.txt', 'w') as out:
    for line in fh:
        parts = line.rstrip('\n').split('\t')
        labels = [t.rpartition('-')[2] for t in parts[4].split()]
        out.write(parts[0] + ' [' + ','.join(labels) + ']\n')
PY

cp reports/submissions/prediction_mind_test.txt /tmp/b/res/prediction.txt
.venv/bin/python tools/evaluate_official.py /tmp/b /tmp/out
cat /tmp/out/scores.txt
```

Row order matters: the scorer zips truth and prediction line by line and
rejects any impression-id mismatch.
