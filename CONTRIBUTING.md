# Contributing

Use the local installation in the README and install Ruff. Reproduce a change with a fixed search seed and a fresh output directory. For a scoring or selection change, describe the dataset, split, evaluation budget and effect on existing runs.

Run `python -m ruff check .` and `python -m unittest discover -s tests`. Test integration changes with mock responses. Include a regression test when changing a metric, candidate validation or an artifact format. Keep old history readable.

Use the internal validation split for development and compare paired seeds under the same budget. Record the selected candidate before running a final test. Keep credentials, downloaded datasets, run directories and locally generated reports in ignored locations. Use an isolated Compose project for integration checks. [Release checklist](docs/release-checklist.md)
