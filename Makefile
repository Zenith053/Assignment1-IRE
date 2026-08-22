PY := .venv/bin/python
CONFIGS := config/mind.yaml config/ebnerd.yaml

.PHONY: all data download clean split features retrieval eval submission test ebnerd-testset

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

test:
	.venv/bin/pytest tests/ -v

# Opt-in: 1.5 GB download, only needed for a scored RecSys 2024 submission.
ebnerd-testset:
	$(PY) src/data/download.py --id ebnerd_testset
