# Dev Conventions

- Python 3.11+, type hints everywhere
- All inter-stage types live in `src/recsys/contracts/` — never define pipeline types elsewhere
- `src/recsys/data/schema.py` has column name constants — don't hardcode CSV column strings
- Use `python -m pytest tests/ -v` to run tests
- Use `pip install -e ".[dev]"` for editable install
- KuaiRec data goes in `data/raw/` (gitignored)
