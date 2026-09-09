# Contributing

Contributions are welcome. Please open an issue before a large change.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check src tests scripts
```

Please add tests for new attacks, metrics, datasets, or model variants.
