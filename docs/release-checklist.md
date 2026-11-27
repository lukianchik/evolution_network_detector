# Release checklist

- Verify Ruff, unit tests, MCP transport and clean Docker Compose startup.
- Verify local experiments and archive recovery without API credentials.
- Record the dataset, split, fixed rule, evaluation protocol and limitations.
- Audit tracked files, Git metadata and screenshots for credentials and private information.
- Exclude `.env`, datasets, quotas and run directories from distribution.
- Do not claim live Groq or Shinka quality improvements without live tests; do not claim deployment readiness without independent validation.

Run the checks in the README in an isolated Compose project. Keep the official test data outside search. Never delete working volumes as part of release checks.
