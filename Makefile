.PHONY: demo demo-continuous gif test install

install:
	pip3 install -r requirements.txt

demo:
	bash scripts/demo.sh

demo-continuous:
	VIEWER_LOG=fixtures/demo_continuous.jsonl bash scripts/demo.sh

gif:
	python3 scripts/generate_demo_gif.py

test:
	python3 -m pytest -q
