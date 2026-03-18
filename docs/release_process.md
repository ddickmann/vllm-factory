# Release Process

## Versioning

- Use semantic versioning tags: `vMAJOR.MINOR.PATCH`
- Keep `CHANGELOG.md` updated before cutting a tag.

## Release Checklist

1. Ensure CI is green on `main`.
2. Update `CHANGELOG.md` under `Unreleased`.
3. Merge release PR into `main`.
4. Create tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

5. GitHub Actions `Release` workflow publishes release notes from the tag.

## Branch Policy

- `main` is protected.
- Pull requests are required for all changes.
- Required checks:
  - `quality / Lint`
  - `quality / Import and syntax smoke checks`
  - `quality / Run CPU-safe tests`
