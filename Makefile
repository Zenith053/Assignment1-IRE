PY := .venv/bin/python
CONFIGS := config/mind.yaml config/ebnerd.yaml

.PHONY: all data download clean split features retrieval eval submission sweep test ebnerd-testset q3 q3-train q3-tables

# One-command rebuild from raw files (Q1.5).
all: data retrieval eval submission

data: download clean split features

download:
	$(PY) src/data/download.py

# clean + split run once per dataset; the modules are dataset-agnostic and
# take their differences entirely from the config file.
clean:
	@for cfg in $(CONFIGS); do $(PY) src/data/clean.py --config $$cfg || exit 1; done

split:
	@for cfg in $(CONFIGS); do $(PY) src/data/split.py --config $$cfg || exit 1; done

features:
	@for cfg in $(CONFIGS); do $(PY) src/data/feature_store.py --config $$cfg || exit 1; done

retrieval:
	@for cfg in $(CONFIGS); do $(PY) src/retrieval/bm25.py --config $$cfg || exit 1; done

eval:
	@for cfg in $(CONFIGS); do $(PY) src/eval/harness.py --config $$cfg || exit 1; done

submission:
	@for cfg in $(CONFIGS); do $(PY) src/submission/generate_predictions.py --config $$cfg || exit 1; done

# Pooling ablation: k for top-k similarity pooling. Not in `all` - it reproduces a
# design decision rather than any reported result.
sweep:
	@for cfg in $(CONFIGS); do \
	  ds=$$(basename $$cfg .yaml); \
	  $(PY) tools/sweep_pooling_k.py --config $$cfg \
	    --out reports/sweep_pooling_k_$${ds}_val.json || exit 1; \
	done

# Q3: NRMS baseline, popularity/freshness-aware NRMS, ablation (A2).
# q3-train is ~11 h on an M4 (resumable; skips finished runs); q3-tables ~3 min.
q3: q3-train q3-tables

q3-train:
	@for cfg in $(CONFIGS); do $(PY) src/baseline/news_data.py --config $$cfg || exit 1; done
	tools/run_q3.sh

q3-tables:
	$(PY) src/baseline/q3_tables.py

test:
	.venv/bin/pytest tests/ -v

# Opt-in: 1.5 GB download, only needed for a scored RecSys 2024 submission.
ebnerd-testset:
	$(PY) src/data/download.py --id ebnerd_testset
