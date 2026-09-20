PY := .venv/bin/python
CONFIGS := config/mind.yaml config/ebnerd.yaml

.PHONY: all data download clean split features retrieval eval submission sweep test ebnerd-testset q3 q3-train q3-tables q4 q4-check q4-memory q4-latency q4-cost q4-scale q4-report q1 q2 q5 q6

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

# Q1: behavioural features live in the feature store + feature builders; this target
# rebuilds the store and runs the boundary (no-future-click) and feature unit tests.
q1: features
	.venv/bin/pytest tests/test_no_leakage.py tests/test_rerank_features.py -q

# Q2: GBDT/MLP re-rankers, both universes; EB-NeRD also runs Q9's with/without
# serving-time-unavailable arm (MIND declares no such columns). ~5 min total.
q2:
	$(PY) src/rerank/evaluate_reranker.py --config config/mind.yaml --sample 10000 --retrieve-sample 3000
	$(PY) src/rerank/evaluate_reranker.py --config config/ebnerd.yaml --sample 10000 --retrieve-sample 3000 --leaky

# Q5: extended evaluation of the two-stage pipeline (all metrics, slices, CIs),
# plus the A1 single-stage harness both datasets report against.
q5:
	@for cfg in $(CONFIGS); do $(PY) src/eval/evaluate_twostage.py --config $$cfg --sample 3500 || exit 1; done
	$(PY) src/eval/harness.py --config config/ebnerd.yaml --split test --out reports/a2/eval_ebnerd_small_test.json

# Q6: the design note (LaTeX + figures + PDF) from the stored results.
q6:
	$(PY) src/submission/codabench_analysis.py
	$(PY) src/baseline/freshness_curve.py
	$(PY) src/report/build_design_note.py

# Q3: NRMS baseline, popularity/freshness-aware NRMS, ablation (A2).
# q3-train is ~11 h on an M4 (resumable; skips finished runs); q3-tables ~3 min.
q3: q3-train q3-tables

q3-train:
	@for cfg in $(CONFIGS); do $(PY) src/baseline/news_data.py --config $$cfg || exit 1; done
	tools/run_q3.sh

q3-tables:
	$(PY) src/baseline/q3_tables.py

# Q4: serving & scale analysis (A2). Needs the Q3 checkpoints named in config/serving.yaml.
# Order matters: cost and scale read the memory and latency results. ~12 min on an M4;
# run on AC power with other apps closed, since background load inflates p99.
q4: q4-check q4-memory q4-latency q4-cost q4-scale q4-report

q4-check:
	$(PY) src/serving/check_setup.py
	@for cfg in $(CONFIGS); do $(PY) src/serving/pipeline.py --config $$cfg --check || exit 1; done

q4-memory:
	@for cfg in $(CONFIGS); do $(PY) src/serving/measure_memory.py --config $$cfg || exit 1; done

q4-latency:
	@for cfg in $(CONFIGS); do $(PY) src/serving/benchmark_latency.py --config $$cfg || exit 1; done

q4-cost:
	$(PY) src/serving/cost_model.py

q4-scale:
	@for cfg in $(CONFIGS); do $(PY) src/serving/scale_10x.py --config $$cfg || exit 1; done

q4-report:
	$(PY) src/serving/q4_report.py

test:
	.venv/bin/pytest tests/ -v

# Opt-in: 1.5 GB download, only needed for a scored RecSys 2024 submission.
ebnerd-testset:
	$(PY) src/data/download.py --id ebnerd_testset
