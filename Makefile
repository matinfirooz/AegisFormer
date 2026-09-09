.PHONY: install test lint smoke train robust pretrain eval profile demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:
	python scripts/smoke_test.py

train:
	aegisformer train --config configs/baseline.yaml

robust:
	aegisformer train --config configs/robust.yaml

pretrain:
	aegisformer pretrain --config configs/simclr.yaml

profile:
	aegisformer profile --config configs/baseline.yaml

demo:
	python demo/app.py --checkpoint checkpoints/best.pt
