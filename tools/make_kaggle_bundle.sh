#!/usr/bin/env bash
# Pack exactly what tools/run_q3.sh needs to train on another machine (Kaggle/Colab):
# code, configs, finished run reports (so they are skipped), and the processed
# data + NRMS caches. No raw data, no checkpoints. ~300 MB.
#
#   tools/make_kaggle_bundle.sh [out.zip]
#
# The data is licensed (MIND, EB-NeRD): keep any upload of this bundle private.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-q3_kaggle_bundle.zip}"
STAGE="$(mktemp -d)/ire-team"
mkdir -p "$STAGE"

cp -R src tools config requirements.txt "$STAGE"/
mkdir -p "$STAGE/reports/q3"
cp reports/q3/*.json "$STAGE/reports/q3/" 2>/dev/null || true

for ds in mind ebnerd; do
  mkdir -p "$STAGE/data/processed/$ds" "$STAGE/data/feature_store/$ds"
  cp data/processed/$ds/articles.parquet data/processed/$ds/split_meta.json "$STAGE/data/processed/$ds/"
  for split in train val test; do
    mkdir -p "$STAGE/data/processed/$ds/$split"
    cp data/processed/$ds/$split/impressions.parquet "$STAGE/data/processed/$ds/$split/"
  done
  cp data/feature_store/$ds/{user_profiles.parquet,nrms_title_tokens.npz,nrms_word_embeddings.npz} \
     "$STAGE/data/feature_store/$ds/"
done

find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$STAGE" -name .DS_Store -delete
rm -f "$OUT"
(cd "$(dirname "$STAGE")" && zip -qr - ire-team) > "$OUT"
echo "$OUT: $(du -h "$OUT" | cut -f1)"
echo "finished runs included (skipped on the remote queue):"
ls "$STAGE/reports/q3/"
