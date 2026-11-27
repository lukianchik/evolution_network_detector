# Network Evolution Lab v1.0.0

Initial public release.

## Highlights

- Run adaptive Local and Hybrid searches through one CPU engine. The search ranks interpretable rules with Pareto selection and a fixed optimization profile.
- Inspect candidate rules, diffs, metrics, Pareto fronts and previous runs in the local panel.
- Verify checkpoints over MCP Streamable HTTP. Compose isolates the metrics service from provider credentials and the official test CSV.
- Enable optional Groq proposals with an explicit attempt budget. The optional ShinkaEvolve CLI has been exercised with a mocked provider.

The frozen UNSW-NB15 leader r69 achieved F1 0.8785, FPR 0.1911, recall 0.9055 and precision 0.8531 on 82,332 official test rows. This split was previously analyzed during development and is not a blind holdout. The benchmark measures research performance on the available data. [Evaluation protocol](evaluation.md)
