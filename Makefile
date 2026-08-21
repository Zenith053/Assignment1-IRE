.PHONY: all download clean split features retrieval eval submission

all: download clean split features retrieval eval submission

download:
	python src/data/download.py

clean:
	python src/data/clean.py

split:
	python src/data/split.py

features:
	python src/data/feature_store.py

retrieval:
	python src/retrieval/bm25.py

eval:
	python src/eval/metrics.py

submission:
	python src/submission/generate_predictions.py
