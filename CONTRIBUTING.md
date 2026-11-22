# Contributing

Thanks for helping. Keep changes small and focused.

1. `make install` creates `.venv` and installs the package with dev extras.
2. Before you open a PR, run `make format && make lint && make test`.
3. Any change to model or pairing logic needs a test in `tests/`.
4. Data: commit new snapshots under `data/snapshots/<source>/<date>/` and update `data/latest.json`
   (`cs2pickem fetch --refresh` does both). Never commit `.cache/`.
5. When you scrape, keep the rate limits and caching in `src/cs2pickem/data/` and respect each site's terms.
   Do not raise request rates.
6. Do not commit personal paths, credentials or tokens.
7. A new event should be a config-only change (`configs/<event>/`) whenever possible.

Code style: ruff (`pyproject.toml`), with type hints and small pure functions.
The simulators are performance-sensitive; check `cs2pickem simulate -n 100000` timing before and after any change to them.
