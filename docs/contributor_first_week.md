# Contributor First Week Path

## Day 1

- Read:
  - `README.md`
  - `docs/architecture.md`
  - `docs/pooling_patch.md`
- Run:

```bash
make install
make lint
pytest -q tests
```

## Day 2-3

- Pick one roadmap issue labeled `good first issue`.
- Add/adjust tests before implementation.
- Keep changes focused to one plugin or one shared subsystem.

## Day 4-5

- Open PR with:
  - short rationale
  - test evidence
  - benchmark evidence if performance-related

## Quality bar

- No silent broad exception swallowing.
- No benchmark claims without reproducibility metadata.
- No changes to pooling patch behavior without compatibility tests.
