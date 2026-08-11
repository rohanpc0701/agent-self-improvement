.PHONY: demo demo-continuous test install

install:
	pip3 install -r requirements.txt

demo:
	bash scripts/demo.sh

demo-continuous:
	VIEWER_LOG=fixtures/demo_continuous.jsonl bash scripts/demo.sh

test:
	python3 -m pytest -q
